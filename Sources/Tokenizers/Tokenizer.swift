//
//  Tokenizer.swift
//  
//
//  Created by Pedro Cuenca on 6/5/23.
//

import Hub
import Foundation

enum TokenizerError : Error {
    case missingTokenizerClassInConfig
    case unsupportedTokenizer(String)
    case missingVocab
    
    case tooLong(String)
}

public protocol Tokenizer {
    func tokenize(text: String) -> [String]
    func encode(text: String) -> [Int]
    func decode(tokens: [Int]) -> String
    
    init(tokenizerData: Config) throws
}

public struct TokenizerFactory {
    static let knownTokenizers: [String : Tokenizer.Type] = [
        "BertTokenizer"  : BertTokenizer.self,
        "GPT2Tokenizer"  : GPT2Tokenizer.self,
        "FalconTokenizer": FalconTokenizer.self,
        "LlamaTokenizer" : LlamaTokenizer.self,
    ]

    public static func from(tokenizerConfig: Config, tokenizerData: Config) throws -> Tokenizer {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass?.stringValue else {
            throw TokenizerError.missingTokenizerClassInConfig
        }
        
        // Some tokenizer_class entries use a Fast suffix
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        guard let tokenizerClass = TokenizerFactory.knownTokenizers[tokenizerName] else {
            throw TokenizerError.unsupportedTokenizer(tokenizerName)
        }
        
        return try tokenizerClass.init(tokenizerData: tokenizerData)
    }

    public static func fallbackTokenizerConfig(for modelType: String) -> Config? {
        guard let url = Bundle.module.url(forResource: "\(modelType)_tokenizer_config", withExtension: "json") else { return nil }
        do {
            let data = try Data(contentsOf: url)
            let parsed = try JSONSerialization.jsonObject(with: data, options: [])
            guard let dictionary = parsed as? [String: Any] else { return nil }
            return Config(dictionary)
        } catch {
            return nil
        }
    }
}
