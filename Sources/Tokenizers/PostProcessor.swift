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
//    case RobertaProcessing
}

struct PostProcessorFactory {
    static func fromConfig(config: Config?) -> PostProcessor? {
        guard let config = config else { return nil }
        guard let typeName = config.type?.stringValue else { return nil }
        let type = PostProcessorType(rawValue: typeName)
        switch type {
        case .TemplateProcessing: return TemplateProcessing(config: config)
        case .ByteLevel         : return ByteLevelPostProcessor(config: config)
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
