//
//  Tokenizer.swift
//  
//
//  Created by Pedro Cuenca on 6/5/23.
//

import Hub

enum TokenizerError : Error {
    case missingTokenizerClassInConfig
    case unsupportedTokenizer(String)
    case missingVocab(String)
    
    case tooLong(String)
}

public protocol Tokenizer {
    func tokenize(text: String) -> [String]
    func encode(text: String) -> [Int]
    func decode(tokens: [Int]) -> String
    
    init(vocab: [String: Int], merges: [String]?)
}

public struct TokenizerFactory {
    static let knownTokenizers: [String : Tokenizer.Type] = [
        "BertTokenizer"  : BertTokenizer.self,
        "GPT2Tokenizer"  : GPT2Tokenizer.self,
        "FalconTokenizer": FalconTokenizer.self,
        "LlamaTokenizer" : LlamaTokenizer.self,
    ]

    //TODO: some models (gpt2) still don't have a tokenizer_config.json
    public static func from(tokenizerConfig: Config, tokenizerData: Config) throws -> Tokenizer {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass?.stringValue else {
            throw TokenizerError.missingTokenizerClassInConfig
        }
        
        // Some tokenizer_class entries use a Fast suffix
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        guard let tokenizerClass = TokenizerFactory.knownTokenizers[tokenizerName] else {
            throw TokenizerError.unsupportedTokenizer(tokenizerName)
        }
        
        guard let vocab = tokenizerData.model?.vocab?.dictionary as? [String: Int] else {
            throw TokenizerError.missingVocab(tokenizerName)
        }
        let merges = tokenizerData.model?.merges?.value as? [String]

        return tokenizerClass.init(vocab: vocab, merges: merges)
    }
}
