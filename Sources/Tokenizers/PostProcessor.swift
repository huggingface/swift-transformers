//
//  PostProcessor.swift
//  
//
//  Created by Pedro Cuenca on 17/7/23.
//

import Foundation
import Hub

public protocol PostProcessor {
    func postProcess(tokens: [String], tokensPair: [String]?) -> [String]
    func callAsFunction(tokens: [String], tokensPair: [String]?) -> [String]
    
    init(config: Config)
}

extension PostProcessor {
    func callAsFunction(tokens: [String], tokensPair: [String]? = nil) -> [String] {
        return postProcess(tokens: tokens, tokensPair: tokensPair)
    }
}

enum PostProcessorType: String {
    case TemplateProcessing
    case ByteLevel
    case RobertaProcessing
}

struct PostProcessorFactory {
    static func fromConfig(config: Config?) -> PostProcessor? {
        guard let config = config else { return nil }
        guard let typeName = config.type?.stringValue else { return nil }
        let type = PostProcessorType(rawValue: typeName)
        switch type {
        case .TemplateProcessing: return TemplateProcessing(config: config)
        case .ByteLevel         : return ByteLevelPostProcessor(config: config)
        case .RobertaProcessing : return RobertaProcessing(config: config)
        default                 : fatalError("Unsupported PostProcessor type: \(typeName)")
        }
    }
}

class TemplateProcessing: PostProcessor {
    let single: [Config]
    let pair: [Config]
    
    required public init(config: Config) {
        guard let single = config.single?.arrayValue else { fatalError("Missing `single` processor configuration") }
        guard let pair = config.pair?.arrayValue else { fatalError("Missing `pair` processor configuration") }
        
        self.single = single
        self.pair = pair
    }
    
    func postProcess(tokens: [String], tokensPair: [String]? = nil) -> [String] {
        let config = tokensPair == nil ? single : pair
                
        var toReturn: [String] = []
        for item in config {
            if let specialToken = item.SpecialToken {
                toReturn.append(specialToken.id!.stringValue!)
            } else if let sequence = item.Sequence {
                if sequence.id?.stringValue == "A" {
                    toReturn += tokens
                } else if sequence.id?.stringValue == "B" {
                    toReturn += tokensPair!
                }
            }
        }
        return toReturn
    }
}

class ByteLevelPostProcessor: PostProcessor {
    required public init(config: Config) {}
    func postProcess(tokens: [String], tokensPair: [String]? = nil) -> [String] { tokens }
}

class RobertaProcessing: PostProcessor {
    private let sep: (UInt, String)
    private let cls: (UInt, String)
    /// Trim all remaining space, or leave one space character if `addPrefixSpace` is `true`.
    private let trimOffset: Bool
    /// Keep one space character on each side. Depends on `trimOffsets` being `true`.
    private let addPrefixSpace: Bool

    required public init(config: Config) {
        guard let sep = config.sep?.tokenValue else { fatalError("Missing `sep` processor configuration") }
        guard let cls = config.cls?.tokenValue else { fatalError("Missing `cls` processor configuration") }
        self.sep = sep
        self.cls = cls
        self.trimOffset = config.trimOffset?.boolValue ?? true
        self.addPrefixSpace = config.addPrefixSpace?.boolValue ?? true
    }
    
    func postProcess(tokens: [String], tokensPair: [String]?) -> [String] {
        var outTokens = tokens
        var tokensPair = tokensPair
        if trimOffset {
            if addPrefixSpace {
                outTokens = outTokens.map({ trimExtraSpaces(token: $0) })
                tokensPair = tokensPair?.map({ trimExtraSpaces(token: $0) })
           } else {
                outTokens = outTokens.map({ $0.trimmingCharacters(in: .whitespaces) })
                tokensPair = tokensPair?.map({ $0.trimmingCharacters(in: .whitespaces) })
            }
        }

        outTokens = [self.cls.1] + outTokens + [self.sep.1]
        if let tokensPair = tokensPair, !tokensPair.isEmpty {
            // Yes, it adds another `sep`.
            // https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/roberta/hub_interface.py#L58-L65
            outTokens += [self.sep.1] + tokensPair + [self.sep.1]
        }

        return outTokens
    }

    private func trimExtraSpaces(token: String) -> String {
        let prefixOffset = findPrefixIndex(text: token)
        let suffixOffset = findSuffixIndex(text: token)
        let prefixIndex = token.index(token.startIndex, offsetBy: prefixOffset)
        let suffixIndex = token.index(token.startIndex, offsetBy: token.count - suffixOffset)
        return String(token[prefixIndex..<suffixIndex])
    }

    private func findPrefixIndex(text: String) -> Int {
        guard !text.isEmpty, text.first!.isWhitespace else { return 0 }
        return text.prefix(while: { $0.isWhitespace }).count - 1
    }

    private func findSuffixIndex(text: String) -> Int {
        guard !text.isEmpty, text.last!.isWhitespace else { return 0 }
        return text.reversed().prefix(while: { $0.isWhitespace }).count - 1
    }
}
