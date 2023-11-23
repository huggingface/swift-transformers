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
    case Punctuation
    case Digits
    case Split
    case Whitespace
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
        case .Punctuation: return PunctuationPreTokenizer(config: config)
        case .Digits: return DigitsPreTokenizer(config: config)
        case .Split: return SplitPreTokenizer(config: config)
        case .Whitespace: return WhitespacePreTokenizer(config: config)
        default       : fatalError("Unsupported PreTokenizer type: \(typeName)")
        }
    }
}

class PreTokenizerSequence: PreTokenizer {
    let preTokenizers: [PreTokenizer]
    
    required init(config: Config) {
        guard let configs = config.pretokenizers?.arrayValue else { fatalError("No pretokenizers in Sequence") }
        preTokenizers = configs.compactMap { PreTokenizerFactory.fromConfig(config: $0) }
    }
    
    func preTokenize(text: String) -> [String] {
        preTokenizers.reduce([text]) { current, preTokenizer in
            preTokenizer(texts: current)
        }
    }
}

class WhitespacePreTokenizer: PreTokenizer {
    let re: String

    required init(config: Config) {
        re = #"\S+"#
    }

    func preTokenize(text: String) -> [String] {
        return text.ranges(of: re).map { String(text[$0]) }
    }
}

class ByteLevelPreTokenizer: PreTokenizer {
    let addPrefixSpace: Bool
    let trimOffsets: Bool
    let useRegex: Bool
    let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
    
    required init(config: Config) {
        addPrefixSpace = config.addPrefixSpace?.boolValue ?? false
        trimOffsets = config.trimOffsets?.boolValue ?? true
        useRegex = config.useRegex?.boolValue ?? true
    }
    
    func preTokenize(text: String) -> [String] {
        // Split on whitespace and punctuation
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

class PunctuationPreTokenizer: PreTokenizer {
    let PUNCTUATION_REGEX = #"\p{P}\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"#
    let re: String

    required init(config: Config) {
        re = "[^\(PUNCTUATION_REGEX)]+|[\(PUNCTUATION_REGEX)]+"
    }

    func preTokenize(text: String) -> [String] {
        // Ref: https://github.com/xenova/transformers.js/blob/27920d84831e323275b38f0b5186644b7936e1a2/src/tokenizers.js#L1138
        return text.ranges(of: re).map { String(text[$0]) }
    }
}

class DigitsPreTokenizer: PreTokenizer {
    let re: String

    required init(config: Config) {
        let individualDigits = config.individualDigits?.boolValue ?? false
        re = "[^\\d]+|\\d\(individualDigits ? "" : "+")"
    }

    func preTokenize(text: String) -> [String] {
        return text.ranges(of: re).map { String(text[$0]) }
    }
}

class SplitPreTokenizer: PreTokenizer {
    let pattern: StringSplitPattern?
    let invert: Bool

    required init(config: Config) {
        pattern = StringSplitPattern.from(config: config)
        invert = config.invert?.boolValue ?? false
    }

    func preTokenize(text: String) -> [String] {
        guard let pattern = pattern else { return [text] }
        return pattern.split(text, invert: invert)
    }
}

enum StringSplitPattern {
    case regexp(regexp: String)
    case string(pattern: String)
}

extension StringSplitPattern {
    func split(_ text: String, invert: Bool = true) -> [String] {
        switch self {
        case .regexp(let regexp):
            return text.split(by: regexp, includeSeparators: !invert)
        case .string(let substring):
            return text.split(by: substring, options: [], includeSeparators: !invert)
        }
    }
}

extension StringSplitPattern {
    static func from(config: Config) -> StringSplitPattern? {
        if let pattern = config.pattern?.String?.stringValue {
            return StringSplitPattern.string(pattern: pattern)
        }
        if let pattern = config.pattern?.Regex?.stringValue {
            return StringSplitPattern.regexp(regexp: pattern)
        }
        return nil
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
        
    func split(by string: String, options: CompareOptions = .regularExpression, includeSeparators: Bool = false, omittingEmptySubsequences: Bool = true) -> [String] {
        var result: [String] = []
        var start = startIndex
        while let range = range(of: string, options: options, range: start..<endIndex) {
            // Prevent empty strings
            if omittingEmptySubsequences && start < range.lowerBound {
                result.append(String(self[start..<range.lowerBound]))
            }
            if includeSeparators {
                result.append(String(self[range]))
            }
            start = range.upperBound
        }
        
        result.append(String(self[start...]))
        return result
    }

}
