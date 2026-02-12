//
//  YYJSONParser.swift
//  swift-transformers
//
//  High-performance JSON parsing using yyjson.
//

import Foundation
import yyjson

/// A high-performance JSON parser using yyjson.
///
/// This parser provides significantly faster JSON parsing compared to Foundation's
/// JSONSerialization, especially for large files like tokenizer.json (10+ MB).
enum YYJSONParser {
    /// Error types for yyjson parsing failures.
    enum ParseError: Error, LocalizedError {
        case readFailed(code: UInt32, message: String, position: Int)
        case nullDocument

        var errorDescription: String? {
            switch self {
            case .readFailed(let code, let message, let position):
                return "yyjson read failed (code \(code)) at position \(position): \(message)"
            case .nullDocument:
                return "yyjson returned null document"
            }
        }
    }

    /// Parses JSON data directly into a Config object.
    ///
    /// This is the most efficient path as it goes directly from yyjson to Config
    /// without intermediate Foundation object creation.
    ///
    /// - Parameter data: The JSON data to parse
    /// - Returns: A Config object
    /// - Throws: ParseError if parsing fails
    static func parseToConfig(_ data: Data) throws -> Config {
        guard !data.isEmpty else {
            throw ParseError.readFailed(code: 0, message: "empty data", position: 0)
        }

        return try data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) -> Config in
            guard let baseAddress = buffer.baseAddress else {
                throw ParseError.readFailed(code: 0, message: "empty buffer", position: 0)
            }

            var err = yyjson_read_err()
            let doc = yyjson_read_opts(
                UnsafeMutableRawPointer(mutating: baseAddress).assumingMemoryBound(to: CChar.self),
                buffer.count,
                0,
                nil,
                &err
            )

            guard let doc = doc else {
                let message = err.msg.map { String(cString: $0) } ?? "unknown error"
                throw ParseError.readFailed(code: err.code, message: message, position: err.pos)
            }

            defer { yyjson_doc_free(doc) }

            guard let root = yyjson_doc_get_root(doc) else {
                throw ParseError.nullDocument
            }

            return convertToConfig(root)
        }
    }

    // MARK: - Direct Config conversion

    private static func convertToConfig(_ val: UnsafeMutablePointer<yyjson_val>) -> Config {
        if yyjson_is_null(val) {
            return Config()
        } else if yyjson_is_bool(val) {
            return Config(yyjson_get_bool(val))
        } else if yyjson_is_uint(val) {
            let uintVal = yyjson_get_uint(val)
            if uintVal > UInt64(Int.max) {
                return Config(Float(uintVal))
            }
            return Config(Int(uintVal))
        } else if yyjson_is_sint(val) {
            let sintVal = yyjson_get_sint(val)
            if sintVal < Int64(Int.min) || sintVal > Int64(Int.max) {
                return Config(Float(sintVal))
            }
            return Config(Int(sintVal))
        } else if yyjson_is_real(val) {
            return Config(Float(yyjson_get_real(val)))
        } else if yyjson_is_str(val) {
            guard let str = yyjson_get_str(val) else { return Config("") }
            return Config(String(cString: str))
        } else if yyjson_is_arr(val) {
            return convertArrayToConfig(val)
        } else if yyjson_is_obj(val) {
            return convertObjectToConfig(val)
        } else {
            return Config()
        }
    }

    private static func convertObjectToConfig(_ obj: UnsafeMutablePointer<yyjson_val>) -> Config {
        let size = yyjson_obj_size(obj)
        var result: [BinaryDistinctString: Config] = Dictionary(minimumCapacity: Int(size))

        var iter = yyjson_obj_iter()
        yyjson_obj_iter_init(obj, &iter)

        while let key = yyjson_obj_iter_next(&iter) {
            guard let keyPtr = yyjson_get_str(key),
                let val = yyjson_obj_iter_get_val(key)
            else {
                continue
            }

            let keyString = String(cString: keyPtr)
            result[BinaryDistinctString(keyString)] = convertToConfig(val)
        }

        return Config(result)
    }

    private static func convertArrayToConfig(_ arr: UnsafeMutablePointer<yyjson_val>) -> Config {
        let size = yyjson_arr_size(arr)
        var result: [Config] = []
        result.reserveCapacity(Int(size))

        var iter = yyjson_arr_iter()
        yyjson_arr_iter_init(arr, &iter)

        while let val = yyjson_arr_iter_next(&iter) {
            result.append(convertToConfig(val))
        }

        return Config(result)
    }
}
