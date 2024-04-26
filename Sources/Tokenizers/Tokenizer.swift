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
    case malformedVocab

    case tooLong(String)
}

public protocol TokenizingModel {
    func tokenize(text: String) -> [String]

    // Alias for `tokenize`
    func callAsFunction(_ text: String) -> [String]

    func convertTokenToId(_ token: String) -> Int?
    func convertTokensToIds(_ tokens: [String]) -> [Int?]

    func convertIdToToken(_ id: Int) -> String?
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    var bosToken: String? { get }
    var bosTokenId: Int? { get }
    var eosToken: String? { get }
    var eosTokenId: Int? { get }
    var unknownToken: String? { get }
    var unknownTokenId: Int? { get }
}

public extension TokenizingModel {
    func callAsFunction(_ text: String) -> [String] {
        tokenize(text: text)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        return tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        return ids.map { convertIdToToken($0) }
    }
}

/// A tokenizer model that is set up with Hub configuration data
public protocol PreTrainedTokenizerModel: TokenizingModel {
    init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String : Int]) throws
}

struct TokenizerModel {
    static let knownTokenizers: [String : PreTrainedTokenizerModel.Type] = [
        "BertTokenizer"      : BertTokenizer.self,
        "CodeGenTokenizer"   : CodeGenTokenizer.self,
        "CodeLlamaTokenizer" : CodeLlamaTokenizer.self,
        "FalconTokenizer"    : FalconTokenizer.self,
        "GemmaTokenizer"     : GemmaTokenizer.self,
        "GPT2Tokenizer"      : GPT2Tokenizer.self,
        "LlamaTokenizer"     : LlamaTokenizer.self,
        "T5Tokenizer"        : T5Tokenizer.self,
        "WhisperTokenizer"   : WhisperTokenizer.self,
        "CohereTokenizer"    : CohereTokenizer.self,
        "PreTrainedTokenizer": BPETokenizer.self
    ]

    static func unknownToken(from tokenizerConfig: Config) -> String? {
        return tokenizerConfig.unkToken?.content?.stringValue ?? tokenizerConfig.unkToken?.stringValue
    }

    public static func from(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String : Int]) throws -> TokenizingModel {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass?.stringValue else {
            throw TokenizerError.missingTokenizerClassInConfig
        }

        // Some tokenizer_class entries use a Fast suffix
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        guard let tokenizerClass = TokenizerModel.knownTokenizers[tokenizerName] else {
            throw TokenizerError.unsupportedTokenizer(tokenizerName)
        }

        return try tokenizerClass.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)
    }
}

public protocol Tokenizer {
    func tokenize(text: String) -> [String]

    /// Main entry point
    func encode(text: String) -> [Int]
    func callAsFunction(_ text: String) -> [Int]

    /// Decode
    func decode(tokens: [Int]) -> String

    func convertTokenToId(_ token: String) -> Int?
    func convertTokensToIds(_ tokens: [String]) -> [Int?]

    func convertIdToToken(_ id: Int) -> String?
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    var bosToken: String? { get }
    var bosTokenId: Int? { get }
    var eosToken: String? { get }
    var eosTokenId: Int? { get }
    var unknownToken: String? { get }
    var unknownTokenId: Int? { get }
}

public extension Tokenizer {
    func callAsFunction(_ text: String) -> [Int] {
        encode(text: text)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        return tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        return ids.map { convertIdToToken($0) }
    }
}

public class PreTrainedTokenizer: Tokenizer {
    let model: TokenizingModel

    public var bosToken: String? { model.bosToken }
    public var bosTokenId: Int? { model.bosTokenId }
    public var eosToken: String? { model.eosToken }
    public var eosTokenId: Int? { model.eosTokenId }
    public var unknownToken: String? { model.unknownToken }
    public var unknownTokenId: Int? { model.unknownTokenId }

    private let addedTokens: Set<String>
    private let specialTokens: [String: Int]
    private let addedTokensRegex: NSRegularExpression?

    private let preTokenizer: PreTokenizer?
    private let normalizer: Normalizer?
    private let postProcessor: PostProcessor?
    private let decoder: Decoder?

    private let cleanUpTokenizationSpaces: Bool

    required public init(tokenizerConfig: Config, tokenizerData: Config) throws {
        var addedTokens: [String : Int] = [:]
        var specialTokens: [String : Int] = [:]
        for addedToken in tokenizerData.addedTokens?.arrayValue ?? [] {
            guard let id = addedToken.id?.intValue else { continue /* malformed: token with no id */ }
            guard let content = addedToken.content?.stringValue else { continue /* malformed: token with no content */ }
            addedTokens[content] = id

            if addedToken.special?.boolValue ?? false {
                specialTokens[content] = id
            }
        }

        let addedTokensRegexString = (tokenizerData.addedTokens?.arrayValue ?? []).compactMap { addedToken in
               guard let content = addedToken.content?.stringValue else { return nil }
               let prefix = (addedToken.lstrip?.boolValue ?? false ? #"\s*"# : "")
               let suffix = (addedToken.rstrip?.boolValue ?? false ? #"\s*"# : "")
               let token = NSRegularExpression.escapedPattern(for: content)
               return "\(prefix)(\(token))\(suffix)"
        }.joined(separator: "|")
        addedTokensRegex = try? NSRegularExpression(pattern: addedTokensRegexString, options: [])

        // TODO: specialTokens are stored but never used
        self.specialTokens = specialTokens
        self.addedTokens = Set(addedTokens.keys)

        self.preTokenizer = PreTokenizerFactory.fromConfig(config: tokenizerData.preTokenizer)
        self.normalizer = NormalizerFactory.fromConfig(config: tokenizerData.normalizer)
        self.postProcessor = PostProcessorFactory.fromConfig(config: tokenizerData.postProcessor)
        self.decoder = DecoderFactory.fromConfig(config: tokenizerData.decoder)
        self.cleanUpTokenizationSpaces = tokenizerConfig.cleanUpTokenizationSpaces?.boolValue ?? true

        model = try TokenizerModel.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)
    }

    func preTokenize(_ text: String) -> [String] {
        guard let preTokenizer = preTokenizer else { return [text] }
        return preTokenizer(text: text)
    }

    func normalize(_ text: String) -> String {
        guard let normalizer = normalizer else { return text }
        return normalizer(text: text)
    }

    func postProcess(_ tokens: [String]) -> [String] {
        guard let postProcessor = postProcessor else { return tokens }
        return postProcessor(tokens: tokens)
    }

    func decodeTokens(_ tokens: [String]) -> [String] {
        guard let tokenDecoder = decoder else { return tokens }
        return tokenDecoder(tokens: tokens)
    }

    /// Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms
    func cleanUp(text: String) -> String {
        guard cleanUpTokenizationSpaces else { return text }

        return text.replacingOccurrences(of: " .", with: ".")
            .replacingOccurrences(of: " ?", with: "?")
            .replacingOccurrences(of: " !", with: "!")
            .replacingOccurrences(of: " ,", with: ",")
            .replacingOccurrences(of: " ' ", with: "'")
            .replacingOccurrences(of: " n't", with: "n't")
            .replacingOccurrences(of: " 'm", with: "'m")
            .replacingOccurrences(of: " 's", with: "'s")
            .replacingOccurrences(of: " 've", with: "'ve")
            .replacingOccurrences(of: " 're", with: "'re")
    }

    public func tokenize(text: String) -> [String] {
        // Take care of special tokens first
        let sections: [String]
        if let regex = self.addedTokensRegex {
            sections = text.split(by: regex)
        } else {
            sections = [text]
        }
        return sections.map { x in
            if addedTokens.contains(x) { return [x] }
            return preTokenize(normalize(x)).flatMap { model($0) }
        }.flatMap { $0 }
    }

    /// Main entry point
    public func encode(text: String) -> [Int] {
        return postProcess(tokenize(text: text)).map { model.convertTokenToId($0)! }
    }

    /// Decode
    public func decode(tokens: [Int]) -> String {
        // IDs to tokens
        let tokenStrings = tokens.map { model.convertIdToToken($0)! }
        let decoded = decodeTokens(tokenStrings)
        // At this point we should have a single String
        return cleanUp(text: decoded.joined(separator: ""))
    }

    public func convertTokenToId(_ token: String) -> Int? {
        model.convertTokenToId(token)
    }

    public func convertIdToToken(_ id: Int) -> String? {
        model.convertIdToToken(id)
    }
}

// MARK: - Building

public struct AutoTokenizer {}

extension AutoTokenizer {
    public static func from(tokenizerConfig: Config, tokenizerData: Config) throws -> Tokenizer {
        return try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    public static func from(
        pretrained model: String,
        hubApi: HubApi = .shared
    ) async throws -> Tokenizer {
        let config = LanguageModelConfigurationFromHub(modelName: model, hubApi: hubApi)
        guard let tokenizerConfig = try await config.tokenizerConfig else { throw TokenizerError.missingConfig }
        let tokenizerData = try await config.tokenizerData

        return try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }
    
    public static func from(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) async throws -> Tokenizer {
        let config = LanguageModelConfigurationFromHub(modelFolder: modelFolder, hubApi: hubApi)
        guard let tokenizerConfig = try await config.tokenizerConfig else { throw TokenizerError.missingConfig }
        let tokenizerData = try await config.tokenizerData
        
        return try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }
}

// MARK: - Tokenizer model classes

class GPT2Tokenizer     : BPETokenizer {}
class FalconTokenizer   : BPETokenizer {}
class LlamaTokenizer    : BPETokenizer {}
class CodeGenTokenizer  : BPETokenizer {}
class WhisperTokenizer  : BPETokenizer {}
class GemmaTokenizer    : BPETokenizer {}
class CodeLlamaTokenizer: BPETokenizer {}
class CohereTokenizer   : BPETokenizer {}

class T5Tokenizer       : UnigramTokenizer {}
