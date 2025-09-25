#if canImport(CoreML)
import CoreML

/// A container for model weights loaded from various tensor file formats.
///
/// `Weights` provides a unified interface for accessing model parameters stored
/// in different formats such as Safetensors, GGUF, and MLX files.
public struct Weights {
    /// Errors that can occur during weight file loading and processing.
    enum WeightsError: LocalizedError {
        /// The weight file format is not supported.
        case notSupported(message: String)
        /// The weight file is invalid or corrupted.
        case invalidFile

        var errorDescription: String? {
            switch self {
            case let .notSupported(message):
                String(localized: "The weight format '\(message)' is not supported by this application.", comment: "Error when weight format is not supported")
            case .invalidFile:
                String(localized: "The weights file is invalid or corrupted.", comment: "Error when weight file is invalid")
            }
        }
    }

    private let dictionary: [String: MLMultiArray]

    init(_ dictionary: [String: MLMultiArray]) {
        self.dictionary = dictionary
    }

    /// Accesses the weight tensor for the given parameter name.
    ///
    /// - Parameter key: The parameter name to look up
    /// - Returns: The weight tensor as an `MLMultiArray`, or `nil` if not found
    subscript(key: String) -> MLMultiArray? { dictionary[key] }

    /// Loads weights from a file URL.
    ///
    /// Supports Safetensors format with automatic format detection. GGUF and MLX
    /// formats are currently recognized but not yet supported.
    ///
    /// - Parameter fileURL: The URL of the weights file to load
    /// - Returns: A `Weights` instance containing the loaded model parameters
    /// - Throws: `WeightsError.notSupported` for unsupported formats,
    ///           `WeightsError.invalidFile` for corrupted files,
    ///           or other errors during file loading
    public static func from(fileURL: URL) throws -> Weights {
        guard ["safetensors", "gguf", "mlx"].contains(fileURL.pathExtension)
        else { throw WeightsError.notSupported(message: "\(fileURL.pathExtension)") }

        let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        switch ([UInt8](data.subdata(in: 0..<4)), [UInt8](data.subdata(in: 4..<6))) {
        case ([0x47, 0x47, 0x55, 0x46], _): throw WeightsError.notSupported(message: "gguf")
        case ([0x93, 0x4E, 0x55, 0x4D], [0x50, 0x59]): throw WeightsError.notSupported(message: "mlx")
        default: return try Safetensor.from(data: data)
        }
    }
}

/// Internal implementation for loading Safetensors format files.
enum Safetensor {
    typealias Error = Weights.WeightsError

    /// Header structure for Safetensors files containing tensor metadata.
    enum Header {
        /// Offset information for individual tensors within the file.
        struct Offset: Decodable {
            /// The start and end byte offsets for the tensor data.
            let dataOffsets: [Int]?
            /// The data type string identifier.
            let dtype: String?
            /// The shape dimensions of the tensor.
            let shape: [Int]?

            /// Converts the string data type to CoreML's `MLMultiArrayDataType`.
            ///
            /// - Returns: The corresponding CoreML data type
            /// - Throws: `WeightsError.notSupported` for unsupported data types
            ///
            /// - Note: Unsupported types include "I8", "U8", "I16", "U16", "BF16"
            var dataType: MLMultiArrayDataType? {
                get throws {
                    switch dtype {
                    case "I32", "U32": .int32
                    case "F16": .float16
                    case "F32": .float32
                    case "F64", "U64": .float64
                    default: throw Error.notSupported(message: "\(dtype ?? "empty")")
                    }
                }
            }
        }

        /// Parses the header from raw JSON data.
        ///
        /// - Parameter data: The JSON header data
        /// - Returns: A dictionary mapping tensor names to their offset information
        /// - Throws: An error if the JSON cannot be decoded
        static func from(data: Data) throws -> [String: Offset?] {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            return try decoder.decode([String: Offset?].self, from: data)
        }
    }

    /// Loads weights from Safetensors format data.
    ///
    /// - Parameter data: The complete file data in Safetensors format
    /// - Returns: A `Weights` instance containing all tensors from the file
    /// - Throws: `WeightsError.invalidFile` for malformed files or other parsing errors
    static func from(data: Data) throws -> Weights {
        let headerSize: Int = data.subdata(in: 0..<8).withUnsafeBytes { $0.load(as: Int.self) }
        guard headerSize < data.count else { throw Error.invalidFile }
        let header = try Header.from(data: data.subdata(in: 8..<(headerSize + 8)))

        var dict = [String: MLMultiArray]()
        for (key, point) in header {
            guard let offsets = point?.dataOffsets, offsets.count == 2,
                let shape = point?.shape as? [NSNumber],
                let dType = try point?.dataType
            else { continue }

            let strides = shape.dropFirst().reversed().reduce(into: [1]) { acc, a in
                acc.insert(acc[0].intValue * a.intValue as NSNumber, at: 0)
            }
            let start = 8 + offsets[0] + headerSize
            let end = 8 + offsets[1] + headerSize
            let tensorData = data.subdata(in: start..<end) as NSData
            let ptr = UnsafeMutableRawPointer(mutating: tensorData.bytes)
            dict[key] = try MLMultiArray(dataPointer: ptr, shape: shape, dataType: dType, strides: strides)
        }

        return Weights(dict)
    }
}
#endif // canImport(CoreML)
