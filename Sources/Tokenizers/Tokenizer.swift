//
//  Tokenizer.swift
//
//
//  Created by Pedro Cuenca on 6/5/23.
//

import Hub
import Foundation

public enum TokenizerError: Error {
    case missingConfig
    case missingTokenizerClassInConfig
    case unsupportedTokenizer(String)
    case missingVocab
    case malformedVocab
    case chatTemplate(String)
    case tooLong(String)
    case mismatchedConfig(String)
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

    var fuseUnknownTokens: Bool { get }
}

// Helper - possibly to be moved somewhere else
public func addedTokenAsString(_ addedToken: Config?) -> String? {
    guard let addedToken = addedToken else { return nil }
    if let stringValue = addedToken.stringValue {
        return stringValue
    }
    // This is possibly a serialization of the AddedToken class
    // TODO: support lstrip, rstrip, normalized, etc.
    return addedToken.content?.stringValue
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
        "DistilbertTokenizer": BertTokenizer.self,
        "DistilBertTokenizer": BertTokenizer.self,
        "CodeGenTokenizer"   : CodeGenTokenizer.self,
        "CodeLlamaTokenizer" : CodeLlamaTokenizer.self,
        "FalconTokenizer"    : FalconTokenizer.self,
        "GemmaTokenizer"     : GemmaTokenizer.self,
        "GPT2Tokenizer"      : GPT2Tokenizer.self,
        "LlamaTokenizer"     : LlamaTokenizer.self,
        "T5Tokenizer"        : T5Tokenizer.self,
        "WhisperTokenizer"   : WhisperTokenizer.self,
        "CohereTokenizer"    : CohereTokenizer.self,
        "Qwen2Tokenizer"     : Qwen2Tokenizer.self,
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

public enum ChatTemplateArgument {
    /// A Jinja template to use for the conversation. Normally it is not necessary to provide a template, since it will be read from the tokenizer config.
    case literal(String)
    /// For models whose tokenizer config includes multiple chat templates, the template can be specified by name. Normally this is not necessary.
    case name(String)
}

public protocol Tokenizer {
    func tokenize(text: String) -> [String]

    /// Main entry point
    func encode(text: String) -> [Int]
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int]

    /// Decode
    func decode(tokens: [Int]) -> String
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String

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

    /// The appropriate chat template is selected from the tokenizer config
    func applyChatTemplate(messages: [[String: String]]) throws -> [Int]

    /// The chat template is provided as a string literal or specified by name
    func applyChatTemplate(messages: [[String: String]], chatTemplate: ChatTemplateArgument) throws -> [Int]

    /// The chat template is provided as a string literal
    func applyChatTemplate(messages: [[String: String]], chatTemplate: String) throws -> [Int]

    func applyChatTemplate(
        messages: [[String: String]],
        /// A chat template can optionally be provided or specified by name when several templates are included in the tokenizer config. Normally this is not necessary.
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [[String: Any]]?
    ) throws -> [Int]
}

extension Tokenizer {
    public func applyChatTemplate(messages: [[String: String]]) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: nil, addGenerationPrompt: true, truncation: false, maxLength: nil, tools: nil)
    }

    public func applyChatTemplate(messages: [[String: String]], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: true, truncation: false, maxLength: nil, tools: nil)
    }

    public func applyChatTemplate(messages: [[String: String]], chatTemplate: String) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: .literal(chatTemplate), addGenerationPrompt: true, truncation: false, maxLength: nil, tools: nil)
    }
}

public extension Tokenizer {
    func callAsFunction(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    
    func decode(tokens: [Int]) -> String {
        decode(tokens: tokens, skipSpecialTokens: false)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        return tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        return ids.map { convertIdToToken($0) }
    }
}

// open because we have to subclass from `TokenizersTemplates`
open class PreTrainedTokenizer: Tokenizer {
    let model: TokenizingModel

    public var bosToken: String? { model.bosToken }
    public var bosTokenId: Int? { model.bosTokenId }
    public var eosToken: String? { model.eosToken }
    public var eosTokenId: Int? { model.eosTokenId }
    public var unknownToken: String? { model.unknownToken }
    public var unknownTokenId: Int? { model.unknownTokenId }
    public var fuseUnknownTokens: Bool { model.fuseUnknownTokens }

    let addedTokens: Set<String>
    let specialTokens: [String: Int]
    let addedTokensRegex: NSRegularExpression?

    let preTokenizer: PreTokenizer?
    let normalizer: Normalizer?
    let postProcessor: PostProcessor?
    let decoder: Decoder?
    public let tokenizerConfig: Config

    let cleanUpTokenizationSpaces: Bool

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

        // Convert to tuples for easier access, then sort by length (descending) to avoid early partial matches
        // (https://github.com/xenova/transformers.js/commit/c305c3824f628f1f02806a6310bd3b18b0f7f8f5)
        let unwrappedAddedTokens : [(content: String, prefix: Bool, suffix: Bool)] = (tokenizerData.addedTokens?.arrayValue ?? []).compactMap { addedToken in
            guard let content = addedToken.content?.stringValue else { return nil }
            let prefix = addedToken.lstrip?.boolValue ?? false
            let suffix = addedToken.rstrip?.boolValue ?? false
            return (content: content, prefix: prefix, suffix: suffix)
        }.sorted {
            $0.content.count > $1.content.count
        }

        // then concatenate into regular expression
        let addedTokensRegexString = unwrappedAddedTokens.map {
            let token = NSRegularExpression.escapedPattern(for: $0.content)
            let prefix = $0.prefix ? #"\s*"# : ""
            let suffix = $0.suffix ? #"\s*"# : ""
            return "\(prefix)(\(token))\(suffix)"
        }.joined(separator: "|")
        addedTokensRegex = try? NSRegularExpression(pattern: addedTokensRegexString, options: [])

        // TODO: specialTokens are stored but never used
        self.specialTokens = specialTokens
        self.addedTokens = Set(addedTokens.keys)

        self.preTokenizer = PreTokenizerFactory.fromConfig(config: tokenizerData.preTokenizer)
        self.normalizer = NormalizerFactory.fromConfig(config: tokenizerData.normalizer)
        self.postProcessor = PostProcessorFactory.fromConfig(config: tokenizerData.postProcessor)
        self.decoder = DecoderFactory.fromConfig(config: tokenizerData.decoder, addedTokens: self.addedTokens)
        self.cleanUpTokenizationSpaces = tokenizerConfig.cleanUpTokenizationSpaces?.boolValue ?? true
        self.tokenizerConfig = tokenizerConfig

        model = try TokenizerModel.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)
    }

    func preTokenize(_ text: String, options: PreTokenizerOptions) -> [String] {
        guard let preTokenizer = preTokenizer else { return [text] }
        return preTokenizer(text: text, options: options)
    }

    func normalize(_ text: String) -> String {
        guard let normalizer = normalizer else { return text }
        return normalizer(text: text)
    }

    func postProcess(_ tokens: [String], addSpecialTokens: Bool = true) -> [String] {
        guard let postProcessor = postProcessor else { return tokens }
        return postProcessor(tokens: tokens, addSpecialTokens: addSpecialTokens)
    }

    func decodeTokens(_ tokens: [String]) -> [String] {
        guard let tokenDecoder = decoder else { return tokens }
        return tokenDecoder(tokens: tokens)
    }

    /// Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms
    func cleanUp(text: String) -> String {
        guard cleanUpTokenizationSpaces else { return text }

        return text
            .replacingOccurrences(of: " .", with: ".")
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

    func fuseUnknown(_ tokens: [String]) -> [String] {
        guard fuseUnknownTokens else { return tokens }
        let (fused, _) = tokens.reduce((fused: [String](), previousIsUnknown: false)) { result, token in
            var (fused, previousIsUnknown) = result
            let isUnknown = model.convertTokenToId(token) == model.unknownTokenId
            if isUnknown {
                if !previousIsUnknown { fused.append(token) }
            } else {
                fused.append(token)
            }
            return (fused, isUnknown)
        }
        return fused
    }

    public func tokenize(text: String) -> [String] {
        // Take care of special tokens first
        let sections: [String]
        if let regex = self.addedTokensRegex {
            sections = text.split(by: regex)
        } else {
            sections = [text]
        }
        return sections.enumerated().map { section, x in
            if addedTokens.contains(x) { return [x] }
            return preTokenize(normalize(x), options: section == 0 ? [.firstSection] : []).flatMap { model($0) }
        }.flatMap { fuseUnknown($0) }
    }

    /// Main entry point
    public func encode(text: String, addSpecialTokens: Bool = true) -> [Int] {
        return postProcess(tokenize(text: text), addSpecialTokens: addSpecialTokens).map { model.convertTokenToId($0)! }
    }

    public func encode(text: String) -> [Int] {
        return encode(text: text, addSpecialTokens: true)
    }

    public func decode(tokens: [Int], skipSpecialTokens: Bool = false) -> String {
        // IDs to tokens
        let tokenStrings: [String]
        if skipSpecialTokens {
            let specialTokenIDs = Set(specialTokens.values)
            tokenStrings = tokens
                .filter { !specialTokenIDs.contains($0) }
                .compactMap { model.convertIdToToken($0) }
        } else {
            tokenStrings = tokens.compactMap { model.convertIdToToken($0) }
        }
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

    open func applyChatTemplate(messages: [[String: String]]) throws -> [Int] {
        try applyChatTemplate(messages: messages, addGenerationPrompt: true)
    }

    open func applyChatTemplate(messages: [[String: String]], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: true)
    }

    open func applyChatTemplate(messages: [[String: String]], chatTemplate: String) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: .literal(chatTemplate), addGenerationPrompt: true)
    }

    open func applyChatTemplate(
        messages: [[String: String]],
        chatTemplate: ChatTemplateArgument? = nil,
        addGenerationPrompt: Bool = false,
        truncation: Bool = false,
        maxLength: Int? = nil,
        /// A list of tools (callable functions) that will be accessible to the model. If the template does not
        /// support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
        /// giving the name, description and argument types for the tool. See the
        /// [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
        /// for more information.
        /// Note: tool calling is not supported yet, it will be available in a future update.
        tools: [[String: Any]]? = nil
    ) throws -> [Int] {
        throw TokenizerError.chatTemplate("Not implemented, you may want to use the `TokenizersWithTemplates` target.")
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
class Qwen2Tokenizer    : BPETokenizer {}

class T5Tokenizer       : UnigramTokenizer {}


// MARK: - PreTrainedTokenizer classes

// These need to be public to be visible from the wrapper factory

public let sentencePieceUnderline = "▁"

// Hack for Llama tokenizers, see https://github.com/huggingface/transformers/blob/bcb841f0073fcd7a4fb88ea8064313c17dcab04a/src/transformers/models/llama/tokenization_llama_fast.py#L181
// Return updated config, or nil
public func maybeUpdatePostProcessor(tokenizerConfig: Config, processorConfig: Config?) throws -> Config? {
    // If it's already a Template processor (instead of a ByteLevel one), assume it's correct
    let postProcessor = PostProcessorFactory.fromConfig(config: processorConfig)
    guard !(postProcessor is TemplateProcessing) else { return nil }

    let addBosToken = tokenizerConfig.addBosToken?.boolValue ?? false
    let bosToken = addedTokenAsString(tokenizerConfig.bosToken)
    if addBosToken && bosToken == nil {
        throw TokenizerError.mismatchedConfig("add_bos_token is True but bos_token is nil")
    }

    let addEosToken = tokenizerConfig.addEosToken?.boolValue ?? false
    let eosToken = addedTokenAsString(tokenizerConfig.eosToken)
    if addEosToken && eosToken == nil {
        throw TokenizerError.mismatchedConfig("add_eos_token is True but eos_token is nil")
    }

    // alt implementation
    var single: [[String : Any]] = []
    if addBosToken {
        single = single + [["SpecialToken": ["id": bosToken!, "type_id": 0]]]
    }
    single = single + [["Sequence": ["id": "A", "type_id": 0]]]
    if addEosToken {
        single = single + [["SpecialToken": ["id": eosToken!, "type_id": 0]]]
    }

    var pair: [[String : Any]] = single
    if addBosToken {
        pair = pair + [["SpecialToken": ["id": bosToken!, "type_id": 1]]]
    }
    pair = pair + [["Sequence": ["id": "B", "type_id": 1]]]
    if addEosToken {
        pair = pair + [["SpecialToken": ["id": eosToken!, "type_id": 1]]]
    }

    let postProcessorConfig = Config(["type": PostProcessorType.TemplateProcessing.rawValue, "single": single, "pair": pair])
    return postProcessorConfig
}
