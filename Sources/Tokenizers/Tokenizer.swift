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

public protocol Tokenizing {
    func tokenize(text: String) -> [String]
    func encode(text: String) -> [Int]
    func decode(tokens: [Int]) -> String
    
    // Alias for `encode`
    func callAsFunction(_ text: String) -> [Int]
    
    func convertTokenToId(_ token: String) -> Int?
    func convertTokensToIds(_ tokens: [String]) -> [Int?]
    
    func convertIdToToken(_ id: Int) -> String?
    func convertIdsToTokens(_ ids: [Int]) -> [String?]
    
    var unknownToken: String? { get }
    var unknownTokenId: Int? { get }
}

/// A tokenizer that is set up with Hub configuration data
public protocol PreTrainedTokenizer: Tokenizing {
    init(tokenizerConfig: Config, tokenizerData: Config) throws
}

public extension Tokenizing {
    func callAsFunction(_ text: String) -> [Int] {
        encode(text: text)
    }
}

public extension Tokenizing {
    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        return tokens.map { convertTokenToId($0) }
    }
    
    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        return ids.map { convertIdToToken($0) }
    }
}

public class Tokenizer {
    let model: Tokenizing
    
    static let knownTokenizers: [String : PreTrainedTokenizer.Type] = [
        "BertTokenizer"   : BertTokenizer.self,
        "GPT2Tokenizer"   : GPT2Tokenizer.self,
        "FalconTokenizer" : FalconTokenizer.self,
        "LlamaTokenizer"  : LlamaTokenizer.self,
        "CodeGenTokenizer": CodeGenTokenizer.self,
        "WhisperTokenizer": WhisperTokenizer.self,
        "T5Tokenizer"     : T5Tokenizer.self,

        // Default
        "PreTrainedTokenizer": BPETokenizer.self
    ]

    public static func from(tokenizerConfig: Config, tokenizerData: Config) throws -> Tokenizing {
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
    
    public static func from(pretrained model: String) async throws -> Tokenizing {
        let config = LanguageModelConfigurationFromHub(modelName: model)
        guard let tokenizerConfig = try await config.tokenizerConfig else { throw TokenizerError.missingConfig }
        let tokenizerData = try await config.tokenizerData

        return try from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    required public init(tokenizerConfig: Config, tokenizerData: Config) throws {
        model = try Tokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }
}

extension Tokenizer: Tokenizing {
    public func tokenize(text: String) -> [String] { model.tokenize(text: text) }
    public func encode(text: String) -> [Int] { model.encode(text: text) }
    public func decode(tokens: [Int]) -> String { model.decode(tokens: tokens) }
    public func convertTokenToId(_ token: String) -> Int? { model.convertTokenToId(token) }
    public func convertIdToToken(_ id: Int) -> String? { model.convertIdToToken(id) }
    public var unknownToken: String? { model.unknownToken }
    public var unknownTokenId: Int? { model.unknownTokenId }
}

public typealias AutoTokenizer = Tokenizer
