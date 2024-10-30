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
    case WordPiece
    case ByteLevel
    case Replace
    case ByteFallback
    case Fuse
    case Strip
    case Metaspace
    case Unknown = ""
}

struct DecoderFactory {
    static func fromConfig(config: Config?, addedTokens: Set<String>? = nil) -> Decoder? {
        // TODO: not sure if we need to include `addedTokens` in all the decoder initializers (and the protocol)
        guard let config = config else { return nil }
        guard let typeName = config.type?.stringValue else { return nil }
        let type = DecoderType(rawValue: typeName)
        switch type {
        case .Sequence    : return DecoderSequence(config: config)
        case .ByteLevel   : return ByteLevelDecoder(config: config, addedTokens: addedTokens)
        case .Replace     : return ReplaceDecoder(config: config)
        case .ByteFallback: return ByteFallbackDecoder(config: config)
        case .Fuse        : return FuseDecoder(config: config)
        case .Strip       : return StripDecoder(config: config)
        case .Metaspace   : return MetaspaceDecoder(config: config)
        case .WordPiece   : return WordPieceDecoder(config: config)
        default           : fatalError("Unsupported Decoder type: \(typeName)")
        }
    }
}

class WordPieceDecoder: Decoder {
    let prefix: String
    let cleanup: Bool

    required public init(config: Config) {
        guard let prefix = config.prefix?.stringValue else { fatalError("Missing `prefix` configuration for WordPieceDecoder.") }
        self.prefix = prefix
        self.cleanup = config.cleanup?.boolValue ?? false
    }

    func decode(tokens: [String]) -> [String] {
        var newTokens = [String]()
        newTokens.reserveCapacity(tokens.count)
        for (index, token) in tokens.enumerated() {
            var decodedToken = token
            if index != 0 {
                if decodedToken.hasPrefix(prefix) {
                    decodedToken = String(decodedToken.dropFirst(prefix.count))
                } else {
                    decodedToken = " \(decodedToken)"
                }
            }
            if cleanup {
                decodedToken = cleanUpTokenization(decodedToken)
            }
            newTokens.append(decodedToken)
        }
        return newTokens
    }

    private func cleanUpTokenization(_ token: String) -> String {
        return token.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

class DecoderSequence: Decoder {
    let decoders: [Decoder]
    
    required public init(config: Config) {
        guard let configs = config.decoders?.arrayValue else { fatalError("No decoders in Sequence") }
        decoders = configs.compactMap { DecoderFactory.fromConfig(config: $0) }
    }
    
    func decode(tokens: [String]) -> [String] {
        decoders.reduce(tokens) { current, decoder in
            decoder(tokens: current)
        }
    }
}

class ByteLevelDecoder: Decoder {
    let addedTokens: Set<String>
    
    required public init(config: Config) {
        self.addedTokens = []
    }
    
    init(config: Config, addedTokens: Set<String>?) {
        self.addedTokens = addedTokens ?? []
    }
    
    func decode(tokens: [String]) -> [String] {
        var subTexts: [String] = []
        var currentSubText: [String] = []
        
        func convertTokensToString(_ tokens: [String]) -> String {
            let text = tokens.joined(separator: "")
            
            let utfCodepoints = text.map { byteDecoder[String($0)]! }
            return String(decoding: utfCodepoints, as: UTF8.self)
        }
        
        for token in tokens {
            if addedTokens.contains(token) {
                if !currentSubText.isEmpty {
                    subTexts.append(convertTokensToString(currentSubText))
                    currentSubText = []
                }
                subTexts.append(token)
            } else {
                currentSubText.append(token)
            }
        }
        
        if !currentSubText.isEmpty {
            subTexts.append(convertTokensToString(currentSubText))
        }
        
        return subTexts
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

class MetaspaceDecoder: Decoder {
    let addPrefixSpace: Bool
    let replacement: String
    
    required public init(config: Config) {
        addPrefixSpace = config.addPrefixSpace?.boolValue ?? false
        replacement = config.replacement?.stringValue ?? "_"
    }

    func decode(tokens: [String]) -> [String] {
        var replaced = tokens.map { token in
            token.replacingOccurrences(of: replacement, with: " ")
        }
        if addPrefixSpace && replaced.first?.starts(with: " ") ?? false {
            replaced[0].removeFirst()
        }
        return replaced
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
