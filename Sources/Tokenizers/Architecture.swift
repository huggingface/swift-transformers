//
//  Tokenizers.swift
//  
//
//  Created by Pedro Cuenca on 6/5/23.
//

import Foundation

public struct Architecture {
    public var name: String
    public var tokenizerClass: Tokenizer.Type
    
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
}

public enum SupportedArchitecture: String, CaseIterable {
    case bert
    case gpt
    
    var architecture: Architecture {
        switch self {
        case .bert: return Architecture.bert
        case .gpt: return Architecture.gpt
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
