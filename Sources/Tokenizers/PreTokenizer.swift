//
//  PreTokenizer.swift
//  
//
//  Created by Pedro Cuenca on 18/7/23.
//

import Foundation
import Hub

public protocol PreTokenizer {
    func preTokenize(text: String) -> [String]
    func preTokenize(texts: [String]) -> [String]
    func callAsFunction(texts: [String]) -> [String]
    func callAsFunction(text: String) -> [String]

    init(config: Config)
}

extension PreTokenizer {
    func preTokenize(texts: [String]) -> [String] {
        texts.flatMap { preTokenize(text: $0) }
    }

    func callAsFunction(texts: [String]) -> [String] {
        return preTokenize(texts: texts)
    }
    
    func callAsFunction(text: String) -> [String] {
        return preTokenize(text: text)
    }
    
}

enum PreTokenizerType: String {
    case Sequence
    case ByteLevel
    // Several more to be supported
    case Unknown = ""
}

struct PreTokenizerFactory {
    static func fromConfig(config: Config?) -> PreTokenizer? {
        guard let config = config else { return nil }
        guard let typeName = config.type?.stringValue else { return nil }
        let type = PreTokenizerType(rawValue: typeName)
        switch type {
        case .Sequence : return PreTokenizerSequence(config: config)
        case .ByteLevel: return ByteLevelPreTokenizer(config: config)
        default       : fatalError("Unsupported PreTokenizer type: \(typeName)")
        }
    }
}

class PreTokenizerSequence: PreTokenizer {
    let preTokenizers: [PreTokenizer]
    
    required public init(config: Config) {
        guard let configs = config.pretokenizers?.arrayValue else { fatalError("No pretokenizers in Sequence") }
        preTokenizers = configs.compactMap { PreTokenizerFactory.fromConfig(config: $0) }
    }
    
    public func preTokenize(text: String) -> [String] {
        preTokenizers.reduce([text]) { current, preTokenizer in
            preTokenizer(texts: current)
        }
    }
}

class ByteLevelPreTokenizer: PreTokenizer {
    let addPrefixSpace: Bool
    let trimOffsets: Bool
    let useRegex: Bool
    let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
    
    required public init(config: Config) {
        addPrefixSpace = config.addPrefixSpace?.boolValue ?? false
        trimOffsets = config.trimOffsets?.boolValue ?? true
        useRegex = config.useRegex?.boolValue ?? true
    }
    
    public func preTokenize(text: String) -> [String] {
        // Split on whitespace and punctuation
//        let tokens: [String]
//        if useRegex {
//            tokens = text.ranges(of: RE).map { String(text[$0]) }
//        } else {
//            tokens = [text]
//        }
        let tokens = useRegex ? text.ranges(of: RE).map({ String(text[$0]) }) : [text]
        return tokens.map { token in
            if addPrefixSpace && !token.hasPrefix(" ") {
                return " " + token
            }
            return token
        }.map { token in
            return Array(token.utf8).map { byteEncoder[$0]! }.joined()
        }
    }
}

extension String {
    func ranges(of string: String, options: CompareOptions = .regularExpression) -> [Range<Index>] {
        var result: [Range<Index>] = []
        var start = startIndex
        while let range = range(of: string, options: options, range: start..<endIndex) {
            result.append(range)
            start = range.lowerBound < range.upperBound ? range.upperBound : index(range.lowerBound, offsetBy: 1, limitedBy: endIndex) ?? endIndex
        }
        return result
    }
}
