//
//  Normalizer.swift
//  
//
//  Created by Pedro Cuenca on 17/7/23.
//

import Foundation
import Hub

public protocol Normalizer {
    func normalize(text: String) -> String
    func callAsFunction(text: String) -> String
    
    init(config: Config)
}

extension Normalizer {
    func callAsFunction(text: String) -> String {
        return normalize(text: text)
    }
}

enum NormalizerType: String {
    case Sequence
    case Prepend
    case Replace
    case Unknown = ""
}

struct NormalizerFactory {
    static func fromConfig(config: Config?) -> Normalizer? {
        guard let config = config else { return nil }
        guard let typeName = config.type?.stringValue else { return nil }
        let type = NormalizerType(rawValue: typeName)
        switch type {
        case .Sequence: return NormalizerSequence(config: config)
        case .Prepend : return PrependNormalizer(config: config)
        case .Replace : return ReplaceNormalizer(config: config)
        default       : fatalError("Unsupported Normalizer type: \(typeName)")
        }
    }
}

class NormalizerSequence: Normalizer {
    let normalizers: [Normalizer]
    
    required public init(config: Config) {
        guard let configs = config.normalizers?.arrayValue else { fatalError("No normalizers in Sequence") }
        normalizers = configs.compactMap { NormalizerFactory.fromConfig(config: $0) }
    }
    
    public func normalize(text: String) -> String {
        normalizers.reduce(text) { current, normalizer in
            normalizer(text: current)
        }
    }
}

class PrependNormalizer: Normalizer {
    let prepend: String
    
    required public init(config: Config) {
        prepend = config.prepend?.stringValue ?? ""
    }
    
    public func normalize(text: String) -> String {
        return prepend + text
    }
}

class ReplaceNormalizer: Normalizer {
    let pattern: StringReplacePattern?
    
    required public init(config: Config) {
        self.pattern = StringReplacePattern.from(config: config)
    }
    
    public func normalize(text: String) -> String {
        guard let pattern = pattern else { return text }
        return pattern.replace(text)
    }
}

enum StringReplacePattern {
    case regexp(regexp: NSRegularExpression, replacement: String)
    case string(pattern: String, replacement: String)
}

extension StringReplacePattern {
    func replace(_ text: String) -> String {
        switch self {
        case .regexp(let regexp, let replacement):
            let range = NSRange(text.startIndex..., in: text)
            let replaced = regexp.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: replacement)
            return replaced
        case .string(let toReplace, let replacement):
            return text.replacingOccurrences(of: toReplace, with: replacement)
        }
    }
}

extension StringReplacePattern {
    static func from(config: Config) -> StringReplacePattern? {
        guard let replacement = config.content?.stringValue else { return nil }
        if let pattern = config.pattern?.String?.stringValue {
            return StringReplacePattern.string(pattern: pattern, replacement: replacement)
        }
        if let pattern = config.pattern?.Regex?.stringValue {
            guard let regexp = try? NSRegularExpression(pattern: pattern, options: []) else {
                fatalError("Cannot build regexp from \(pattern)")
            }
            return StringReplacePattern.regexp(regexp: regexp, replacement: replacement)
        }
        return nil
    }
}
