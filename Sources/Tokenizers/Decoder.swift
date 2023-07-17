//
//  Decoder.swift
//  
//
//  Created by Pedro Cuenca on 17/7/23.
//

import Foundation
import Hub

public protocol Decoder {
    func decode(tokens: [String]) -> [String]
    func callAsFunction(tokens: [String]) -> [String]
    
    init(config: Config)
}

extension Decoder {
    func callAsFunction(tokens: [String]) -> [String] {
        return decode(tokens: tokens)
    }
}

enum DecoderType: String {
    case Sequence
//    case WordPiece
//    case Metaspace
//    case ByteLevel
    case Replace
    case ByteFallback
    case Fuse
    case Strip
    case Unknown = ""
}

struct DecoderFactory {
    static func fromConfig(config: Config) -> Decoder {
        let type = DecoderType(rawValue: config.type?.stringValue ?? "")
        switch type {
        case .Sequence    : return DecoderSequence(config: config)
        case .Replace     : return ReplaceDecoder(config: config)
        case .ByteFallback: return ByteFallbackDecoder(config: config)
        case .Fuse        : return FuseDecoder(config: config)
        case .Strip       : return StripDecoder(config: config)
        default           : fatalError("Unsupported decoder type \(String(describing: type))")
        }
    }
}

class DecoderSequence: Decoder {
    let decoders: [Decoder]
    
    required public init(config: Config) {
        guard let configs = config.decoders?.arrayValue else { fatalError("No decoders in Sequence") }
        decoders = configs.map { DecoderFactory.fromConfig(config: $0) }
    }
    
    func decode(tokens: [String]) -> [String] {
        decoders.reduce(tokens) { current, decoder in
            decoder(tokens: current)
        }
    }
}

class ReplaceDecoder: Decoder {
    let pattern: StringReplacePattern?
    
    required public init(config: Config) {
        self.pattern = StringReplacePattern.from(config: config)
    }
    
    func decode(tokens: [String]) -> [String] {
        guard let pattern = pattern else { return tokens }
        return tokens.map { pattern.replace($0) }
    }
}

class ByteFallbackDecoder: Decoder {
    required public init(config: Config) {}
    
    func decode(tokens: [String]) -> [String] {
        var newTokens: [String] = []
        var byteTokens: [Int] = []

        func parseByte(_ token: String) -> Int? {
            guard token.count == 6 && token.hasPrefix("<0x") && token.hasSuffix(">") else {
                return nil
            }
            let startIndex = token.index(token.startIndex, offsetBy: 3)
            let endIndex = token.index(token.startIndex, offsetBy: 5)
            return Int(token[startIndex..<endIndex], radix: 16)
        }
        
        for token in tokens {
            if let byte = parseByte(token) {
                byteTokens.append(byte)
            } else {
                if !byteTokens.isEmpty {
                    // decode as utf8 and append
                    let codeUnits = byteTokens.map { UTF8.CodeUnit($0) }
                    newTokens.append(String(decoding: codeUnits, as: UTF8.self))
                    byteTokens.removeAll()
                }
                newTokens.append(token)
            }
        }
        return newTokens
    }
}

class FuseDecoder: Decoder {
    required public init(config: Config) {}
    
    func decode(tokens: [String]) -> [String] {
        [tokens.joined(separator: "")]
    }
}

class StripDecoder: Decoder {
    let content: String
    let start: Int
    let stop: Int
    
    required public init(config: Config) {
        guard let content = config.content?.stringValue else { fatalError("Incorrect StripDecoder configuration: can't parse `content`.") }
        guard let start = config.start?.intValue else { fatalError("Incorrect StripDecoder configuration: can't parse `start`.") }
        guard let stop = config.stop?.intValue else { fatalError("Incorrect StripDecoder configuration: can't parse `stop`.") }
        self.content = content
        self.start = start
        self.stop = stop
    }
    
    func decode(tokens: [String]) -> [String] {
        tokens.map { token in
            token.trimmingFromStart(upto: start).trimmingFromEnd(upto: stop)
        }
    }
}

// We could use firstIndex(where:), lastIndex(where:) for possibly better efficiency (and do both ends at once)
public extension String {
    func trimmingFromStart(character: Character = " ", upto: Int) -> String {
        var result = self
        var trimmed = 0
        while trimmed < upto && result.first == character {
            result.removeFirst()
            trimmed += 1
        }
        return result
    }

    func trimmingFromEnd(character: Character = " ", upto: Int) -> String {
        var result = self
        var trimmed = 0
        while trimmed < upto && result.last == character {
            result.removeLast()
            trimmed += 1
        }
        return result
    }
}
