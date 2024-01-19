//
//  Tokenizers.swift
//  
//
//  Created by Pedro Cuenca on 6/5/23.
//

import Foundation

#warning("this file to be replaced by TokenizerFactory and data downloaded from the hub")
public struct Architecture {
    public var name: String
    public var tokenizerClass: TokenizingModel.Type
    
    /// These will be hardcoded for now. We could:
    /// - Retrieve them from the compiled Core ML model, storing them as metadata fields
    /// - The configuration file from the Hub
    public var padTokenId: Int? = nil
    public var bosTokenId: Int? = nil
    public var eosTokenId: Int? = nil
}

extension Architecture {
    static let bert: Architecture = Architecture(name: "bert", tokenizerClass: BertTokenizer.self)
    static let gpt: Architecture = Architecture(name: "gpt", tokenizerClass: GPT2Tokenizer.self, bosTokenId: 50256, eosTokenId: 50256)
    static let rw: Architecture = Architecture(name: "rw", tokenizerClass: GPT2Tokenizer.self, bosTokenId: 11, eosTokenId: 11)
}

public enum SupportedArchitecture: String, CaseIterable {
    case bert
    case gpt
    case rw = "RefinedWebModel"
    
    var architecture: Architecture {
        switch self {
        case .bert: return Architecture.bert
        case .gpt: return Architecture.gpt
        case .rw: return Architecture.rw
        }
    }
}

extension Architecture {
    public static func from(modelType: String) -> Architecture? {
        for arch in SupportedArchitecture.allCases {
            if modelType.contains(arch.rawValue) {
                return arch.architecture
            }
        }
        return nil
    }
}
