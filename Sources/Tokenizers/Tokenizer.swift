//
//  Tokenizer.swift
//  
//
//  Created by Pedro Cuenca on 6/5/23.
//

import Hub
import Foundation

enum TokenizerError : Error {
    case missingConfig
    case missingTokenizerClassInConfig
    case unsupportedTokenizer(String)
    case missingVocab
    
    case tooLong(String)
}

public protocol Tokenizer {
    func tokenize(text: String) -> [String]
    func encode(text: String) -> [Int]
    func decode(tokens: [Int]) -> String
    
    init(tokenizerConfig: Config, tokenizerData: Config) throws

    // Alias for `encode`
    func callAsFunction(_ text: String) -> [Int]
}

public extension Tokenizer {
    func callAsFunction(_ text: String) -> [Int] {
        encode(text: text)
    }
}

/// Unfortunately factory methods can't be added to protocols yet; not sure if there's a better pattern
/// What I'd like: `Tokenizer.from(pretrained: "")`
public struct AutoTokenizer {
    static let knownTokenizers: [String : Tokenizer.Type] = [
        "BertTokenizer"   : BertTokenizer.self,
        "GPT2Tokenizer"   : GPT2Tokenizer.self,
        "FalconTokenizer" : FalconTokenizer.self,
        "LlamaTokenizer"  : LlamaTokenizer.self,
        "CodeGenTokenizer": CodeGenTokenizer.self,
        "WhisperTokenizer": WhisperTokenizer.self,

        // Default
        "PreTrainedTokenizer": BPETokenizer.self
    ]

    public static func from(tokenizerConfig: Config, tokenizerData: Config) throws -> Tokenizer {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass?.stringValue else {
            throw TokenizerError.missingTokenizerClassInConfig
        }
        
        // Some tokenizer_class entries use a Fast suffix
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        guard let tokenizerClass = AutoTokenizer.knownTokenizers[tokenizerName] else {
            throw TokenizerError.unsupportedTokenizer(tokenizerName)
        }
        
        return try tokenizerClass.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }
    
    public static func from(pretrained model: String) async throws -> Tokenizer {
        let config = LanguageModelConfigurationFromHub(modelName: model)
        guard let tokenizerConfig = try await config.tokenizerConfig else { throw TokenizerError.missingConfig }
        let tokenizerData = try await config.tokenizerData

        return try from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }
}
