//
//  Tokenizer.swift
//
//
//  Created by Pedro Cuenca on 6/5/23.
//

import Foundation
import Hub
import Jinja

/// A type alias for chat messages, represented as key-value pairs.
public typealias Message = [String: Any]

/// A type alias for tool specifications used in chat templating.
public typealias ToolSpec = [String: Any]

/// Errors that can occur during tokenizer operations.
public enum TokenizerError: LocalizedError {
    case missingConfig
    case missingTokenizerClassInConfig
    case unsupportedTokenizer(String)
    case missingVocab
    case malformedVocab
    case chatTemplate(String)
    case missingChatTemplate
    case tooLong(String)
    case mismatchedConfig(String)

    public var errorDescription: String? {
        switch self {
        case .missingConfig:
            String(localized: "Tokenizer configuration is missing.", comment: "Error when tokenizer config cannot be found")
        case .missingTokenizerClassInConfig:
            String(localized: "The tokenizer class is not specified in the configuration.", comment: "Error when tokenizer_class is missing in config")
        case let .unsupportedTokenizer(name):
            String(localized: "The tokenizer type '\(name)' is not supported.", comment: "Error when tokenizer type is not supported")
        case .missingVocab:
            String(localized: "Vocabulary file is missing from the tokenizer configuration.", comment: "Error when vocab file is missing")
        case .malformedVocab:
            String(localized: "The vocabulary file is malformed or corrupted.", comment: "Error when vocab file is malformed")
        case let .chatTemplate(message):
            String(localized: "Chat template error: \(message)", comment: "Error with chat template")
        case .missingChatTemplate:
            String(localized: "This tokenizer does not have a chat template, and no template was passed.")
        case let .tooLong(message):
            String(localized: "Input is too long: \(message)", comment: "Error when input exceeds maximum length")
        case let .mismatchedConfig(message):
            String(localized: "Tokenizer configuration mismatch: \(message)", comment: "Error when tokenizer configuration is inconsistent")
        }
    }
}

/// A protocol defining the core tokenization functionality.
///
/// This protocol defines the fundamental operations that any tokenization model must support,
/// including converting between text and tokens, and between tokens and their numeric IDs.
public protocol TokenizingModel {
    /// Tokenizes the input text into a sequence of tokens.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of tokens as strings
    func tokenize(text: String) -> [String]

    /// Alias for `tokenize` that allows the instance to be called as a function.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of tokens as strings
    func callAsFunction(_ text: String) -> [String]

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID of the token, or nil if the token is not in the vocabulary
    func convertTokenToId(_ token: String) -> Int?

    /// Converts multiple token strings to their corresponding numeric IDs.
    ///
    /// - Parameter tokens: An array of token strings to convert
    /// - Returns: An array of numeric IDs, with nil values for tokens not in the vocabulary
    func convertTokensToIds(_ tokens: [String]) -> [Int?]

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is not valid
    func convertIdToToken(_ id: Int) -> String?

    /// Converts multiple numeric token IDs back to their string representations.
    ///
    /// - Parameter ids: An array of numeric token IDs to convert
    /// - Returns: An array of token strings, with nil values for invalid IDs
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    /// The beginning-of-sequence token string, if defined.
    var bosToken: String? { get }

    /// The numeric ID of the beginning-of-sequence token, if defined.
    var bosTokenId: Int? { get }

    /// The end-of-sequence token string, if defined.
    var eosToken: String? { get }

    /// The numeric ID of the end-of-sequence token, if defined.
    var eosTokenId: Int? { get }

    /// The unknown token string used for out-of-vocabulary words.
    var unknownToken: String? { get }

    /// The numeric ID of the unknown token.
    var unknownTokenId: Int? { get }

    /// Whether consecutive unknown tokens should be fused together.
    var fuseUnknownTokens: Bool { get }
}

/// Helper - possibly to be moved somewhere else
func addedTokenAsString(_ addedToken: Config?) -> String? {
    guard let addedToken else { return nil }
    if let stringValue = addedToken.string() {
        return stringValue
    }
    // This is possibly a serialization of the AddedToken class
    // TODO: support lstrip, rstrip, normalized, etc.
    return addedToken.content.string()
}

public extension TokenizingModel {
    func callAsFunction(_ text: String) -> [String] {
        tokenize(text: text)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map { convertIdToToken($0) }
    }
}

/// A tokenizer model that can be initialized from Hugging Face Hub configuration data.
///
/// This protocol extends `TokenizingModel` with the ability to be created from configuration
/// files typically found in tokenizer repositories on the Hugging Face Hub.
public protocol PreTrainedTokenizerModel: TokenizingModel {
    /// Initializes a tokenizer model from configuration data.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration (typically from tokenizer_config.json)
    ///   - tokenizerData: The tokenizer data (typically from tokenizer.json)
    ///   - addedTokens: A dictionary mapping added token strings to their IDs
    /// - Throws: `TokenizerError` if the configuration is invalid or missing required data
    init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws
}

enum TokenizerModel {
    static let knownTokenizers: [String: PreTrainedTokenizerModel.Type] = [
        "BertTokenizer": BertTokenizer.self,
        "CodeGenTokenizer": BPETokenizer.self,
        "CodeLlamaTokenizer": BPETokenizer.self,
        "CohereTokenizer": BPETokenizer.self,
        "DistilbertTokenizer": BertTokenizer.self,
        "DistilBertTokenizer": BertTokenizer.self,
        "FalconTokenizer": BPETokenizer.self,
        "GemmaTokenizer": BPETokenizer.self,
        "GPT2Tokenizer": BPETokenizer.self,
        "LlamaTokenizer": BPETokenizer.self,
        "RobertaTokenizer": BPETokenizer.self,
        "T5Tokenizer": T5Tokenizer.self,
        "PreTrainedTokenizer": BPETokenizer.self,
        "Qwen2Tokenizer": BPETokenizer.self,
        "WhisperTokenizer": BPETokenizer.self,
        "XLMRobertaTokenizer": UnigramTokenizer.self,
    ]

    static func unknownToken(from tokenizerConfig: Config) -> String? {
        tokenizerConfig.unkToken.content.string() ?? tokenizerConfig.unkToken.string()
    }

    static func from(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int], strict: Bool = true) throws -> TokenizingModel {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass.string() else {
            throw TokenizerError.missingTokenizerClassInConfig
        }

        // Some tokenizer_class entries use a Fast suffix
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        // Fallback to BPETokenizer if class is not explicitly registered
        let tokenizerClass = TokenizerModel.knownTokenizers[tokenizerName] ?? BPETokenizer.self
        if TokenizerModel.knownTokenizers[tokenizerName] == nil {
            if strict {
                throw TokenizerError.unsupportedTokenizer(tokenizerName)
            } else {
                print("Warning: Tokenizer model class \(tokenizerName) is not registered, falling back to a standard BPE implementation.")
            }
        }
        return try tokenizerClass.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)
    }
}

/// Arguments for specifying chat templates when applying chat formatting.
public enum ChatTemplateArgument {
    /// A Jinja template to use for the conversation.
    ///
    /// Normally it is not necessary to provide a template, since it will be read from the tokenizer config.
    case literal(String)

    /// For models whose tokenizer config includes multiple chat templates, the template can be specified by name.
    ///
    /// Normally this is not necessary.
    case name(String)
}

/// A complete tokenizer interface supporting encoding, decoding, and chat template functionality.
///
/// This is the main protocol that defines all tokenizer operations, including text processing,
/// chat template application, and special token handling.
public protocol Tokenizer: Sendable {
    /// Tokenizes the input text into a sequence of tokens.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of tokens as strings
    func tokenize(text: String) -> [String]

    /// Encodes text into token IDs with special tokens included by default.
    ///
    /// This is the main entry point for most tokenization tasks.
    ///
    /// - Parameter text: The input text to encode
    /// - Returns: An array of token IDs
    func encode(text: String) -> [Int]

    /// Encodes text into token IDs with optional special token handling.
    ///
    /// - Parameters:
    ///   - text: The input text to encode
    ///   - addSpecialTokens: Whether to add special tokens (e.g., BOS, EOS)
    /// - Returns: An array of token IDs
    func encode(text: String, addSpecialTokens: Bool) -> [Int]

    /// Function call syntax for encoding text.
    ///
    /// - Parameters:
    ///   - text: The input text to encode
    ///   - addSpecialTokens: Whether to add special tokens
    /// - Returns: An array of token IDs
    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int]

    /// Decodes token IDs back into text with special tokens included.
    ///
    /// - Parameter tokens: The token IDs to decode
    /// - Returns: The decoded text string
    func decode(tokens: [Int]) -> String

    /// Decodes token IDs back into text with optional special token handling.
    ///
    /// - Parameters:
    ///   - tokens: The token IDs to decode
    ///   - skipSpecialTokens: Whether to skip special tokens in the output
    /// - Returns: The decoded text string
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID of the token, or nil if not found
    func convertTokenToId(_ token: String) -> Int?

    /// Converts multiple token strings to their corresponding numeric IDs.
    ///
    /// - Parameter tokens: An array of token strings to convert
    /// - Returns: An array of numeric IDs, with nil values for unknown tokens
    func convertTokensToIds(_ tokens: [String]) -> [Int?]

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is invalid
    func convertIdToToken(_ id: Int) -> String?

    /// Converts multiple numeric token IDs back to their string representations.
    ///
    /// - Parameter ids: An array of numeric token IDs to convert
    /// - Returns: An array of token strings, with nil values for invalid IDs
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    /// The beginning-of-sequence token string, if defined.
    var bosToken: String? { get }

    /// The numeric ID of the beginning-of-sequence token, if defined.
    var bosTokenId: Int? { get }

    /// The end-of-sequence token string, if defined.
    var eosToken: String? { get }

    /// The numeric ID of the end-of-sequence token, if defined.
    var eosTokenId: Int? { get }

    /// The unknown token string used for out-of-vocabulary words.
    var unknownToken: String? { get }

    /// The numeric ID of the unknown token.
    var unknownTokenId: Int? { get }

    /// Whether this tokenizer has a chat template configured.
    var hasChatTemplate: Bool { get }

    /// Applies the configured chat template to format messages for model input.
    ///
    /// - Parameter messages: Array of message dictionaries representing the conversation
    /// - Returns: Token IDs for the formatted conversation
    /// - Throws: `TokenizerError` if template application fails or no template is available
    func applyChatTemplate(messages: [Message]) throws -> [Int]

    /// Applies the configured chat template with optional tool specifications.
    ///
    /// - Parameters:
    ///   - messages: Array of message dictionaries representing the conversation
    ///   - tools: Optional array of tool specifications for function calling
    /// - Returns: Token IDs for the formatted conversation
    /// - Throws: `TokenizerError` if template application fails or no template is available
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int]

    /// Applies the configured chat template with tools and additional context.
    ///
    /// - Parameters:
    ///   - messages: Array of message dictionaries representing the conversation
    ///   - tools: Optional array of tool specifications for function calling
    ///   - additionalContext: Additional context variables for template rendering
    /// - Returns: Token IDs for the formatted conversation
    /// - Throws: `TokenizerError` if template application fails or no template is available
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?, additionalContext: [String: Any]?) throws -> [Int]

    /// Applies a specific chat template to format messages.
    ///
    /// - Parameters:
    ///   - messages: Array of message dictionaries representing the conversation
    ///   - chatTemplate: The chat template to use (literal template or template name)
    /// - Returns: Token IDs for the formatted conversation
    /// - Throws: `TokenizerError` if template application fails
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int]

    /// Applies a chat template provided as a literal string.
    ///
    /// - Parameters:
    ///   - messages: Array of message dictionaries representing the conversation
    ///   - chatTemplate: The Jinja template string to use for formatting
    /// - Returns: Token IDs for the formatted conversation
    /// - Throws: `TokenizerError` if template application fails
    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int]

    /// Applies a chat template with full control over all parameters.
    ///
    /// - Parameters:
    ///   - messages: Array of message dictionaries representing the conversation
    ///   - chatTemplate: Optional chat template specification
    ///   - addGenerationPrompt: Whether to add a generation prompt for the assistant
    ///   - truncation: Whether to truncate if the result exceeds maximum length
    ///   - maxLength: Maximum allowed token length
    ///   - tools: Optional array of tool specifications for function calling
    /// - Returns: Token IDs for the formatted conversation
    /// - Throws: `TokenizerError` if template application fails
    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?
    ) throws -> [Int]

    /// Applies a chat template with full control over all parameters including additional context.
    ///
    /// - Parameters:
    ///   - messages: Array of message dictionaries representing the conversation
    ///   - chatTemplate: Optional chat template specification
    ///   - addGenerationPrompt: Whether to add a generation prompt for the assistant
    ///   - truncation: Whether to truncate if the result exceeds maximum length
    ///   - maxLength: Maximum allowed token length
    ///   - tools: Optional array of tool specifications for function calling
    ///   - additionalContext: Additional context variables for template rendering
    /// - Returns: Token IDs for the formatted conversation
    /// - Throws: `TokenizerError` if template application fails
    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: Any]?
    ) throws -> [Int]
}

extension Tokenizer {
    public var hasChatTemplate: Bool { false }

    /// Call previous signature for backwards compatibility
    func applyChatTemplate(
        messages: [Message],
        // A chat template can optionally be provided or specified by name when several templates are included in the tokenizer config. Normally this is not necessary.
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: Any]?
    ) throws -> [Int] {
        if additionalContext == nil {
            try applyChatTemplate(
                messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: addGenerationPrompt, truncation: truncation, maxLength: maxLength,
                tools: tools
            )
        } else {
            throw TokenizerError.chatTemplate("Not implemented")
        }
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
        tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map { convertIdToToken($0) }
    }
}

let specialTokenAttributes: [String] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
]

/// A comprehensive tokenizer implementation supporting pre-trained models from Hugging Face.
///
/// This class provides a complete tokenizer implementation that can be initialized from
/// Hugging Face Hub configuration files and supports all standard tokenization operations
/// including chat template application, normalization, pre-tokenization, and post-processing.
public class PreTrainedTokenizer: @unchecked Sendable, Tokenizer {
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

    private let preTokenizer: PreTokenizer?
    private let normalizer: Normalizer?
    private let postProcessor: PostProcessor?
    private let decoder: Decoder?
    private let tokenizerConfig: Config

    private let cleanUpTokenizationSpaces: Bool

    /// Cache for compiled Jinja templates keyed by their literal template string
    private var compiledChatTemplateCache: [String: Template] = [:]

    /// Lock to protect the compiled chat template cache from concurrent access
    private let cacheLock = NSLock()

    /// Initializes a tokenizer from Hugging Face configuration files.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: Configuration from tokenizer_config.json
    ///   - tokenizerData: Configuration from tokenizer.json
    ///   - strict: Whether to enforce strict validation of tokenizer types
    /// - Throws: `TokenizerError` if configuration is invalid or tokenizer type is unsupported
    public required init(tokenizerConfig: Config, tokenizerData: Config, strict: Bool = true) throws {
        var addedTokens: [String: Int] = [:]
        var specialTokens: [String: Int] = [:]
        for addedToken in tokenizerData["addedTokens"].array(or: []) {
            guard let id = addedToken["id"].integer() else { continue } // malformed: token with no id
            guard let content = addedToken.content.string() else { continue } // malformed: token with no content
            addedTokens[content] = id

            if addedToken["special"].boolean(or: false) {
                specialTokens[content] = id
            }
        }

        // Convert to tuples for easier access, then sort by length (descending) to avoid early partial matches
        // (https://github.com/xenova/transformers.js/commit/c305c3824f628f1f02806a6310bd3b18b0f7f8f5)
        let unwrappedAddedTokens: [(content: String, prefix: Bool, suffix: Bool)] = (tokenizerData["addedTokens"].array(or: [])).compactMap { addedToken -> (String, Bool, Bool)? in
            guard let content = addedToken.content.string() else { return nil }
            let prefix = addedToken["lstrip"].boolean(or: false)
            let suffix = addedToken["rstrip"].boolean(or: false)
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

        self.specialTokens = specialTokens
        self.addedTokens = Set(addedTokens.keys)

        preTokenizer = PreTokenizerFactory.fromConfig(config: tokenizerData["preTokenizer"])
        normalizer = NormalizerFactory.fromConfig(config: tokenizerData["normalizer"])
        postProcessor = PostProcessorFactory.fromConfig(config: tokenizerData["postProcessor"])
        decoder = DecoderFactory.fromConfig(config: tokenizerData["decoder"], addedTokens: self.addedTokens)
        cleanUpTokenizationSpaces = tokenizerConfig.cleanUpTokenizationSpaces.boolean(or: true)
        self.tokenizerConfig = tokenizerConfig

        model = try TokenizerModel.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens, strict: strict)
    }

    private func compiledTemplate(for templateString: String) throws -> Template {
        // Fast path: check cache under lock
        cacheLock.lock()
        if let cached = compiledChatTemplateCache[templateString] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()

        // Compile template outside of lock to avoid holding lock during expensive operation
        let compiled = try Template(templateString)

        // Insert into cache under lock (using double-checked locking pattern)
        cacheLock.lock()
        defer { cacheLock.unlock() }

        // Check again in case another thread compiled the same template
        if let cached = compiledChatTemplateCache[templateString] {
            return cached
        }

        compiledChatTemplateCache[templateString] = compiled
        return compiled
    }

    func preTokenize(_ text: String, options: PreTokenizerOptions) -> [String] {
        guard let preTokenizer else { return [text] }
        return preTokenizer(text: text, options: options)
    }

    func normalize(_ text: String) -> String {
        guard let normalizer else { return text }
        return normalizer(text: text)
    }

    func postProcess(_ tokens: [String], addSpecialTokens: Bool = true) -> [String] {
        guard let postProcessor else { return tokens }
        return postProcessor(tokens: tokens, addSpecialTokens: addSpecialTokens)
    }

    func decodeTokens(_ tokens: [String]) -> [String] {
        guard let tokenDecoder = decoder else { return tokens }
        return tokenDecoder(tokens: tokens)
    }

    /// Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms
    func cleanUp(text: String) -> String {
        guard cleanUpTokenizationSpaces else { return text }

        return
            text
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

    /// Tokenizes input text using the configured normalization and pre-tokenization steps.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of token strings
    public func tokenize(text: String) -> [String] {
        // Take care of special tokens first
        let sections: [String] =
            if let regex = addedTokensRegex {
                text.split(by: regex)
            } else {
                [text]
            }
        return sections.enumerated().map { section, x in
            if addedTokens.contains(x) { return [x] }
            return preTokenize(normalize(x), options: section == 0 ? [.firstSection] : []).flatMap { model($0) }
        }.flatMap { fuseUnknown($0) }
    }

    /// Encodes input text into token IDs with optional special token handling.
    ///
    /// This is the main entry point for text encoding operations.
    ///
    /// - Parameters:
    ///   - text: The input text to encode
    ///   - addSpecialTokens: Whether to add special tokens during post-processing
    /// - Returns: An array of token IDs
    public func encode(text: String, addSpecialTokens: Bool = true) -> [Int] {
        postProcess(tokenize(text: text), addSpecialTokens: addSpecialTokens).map { model.convertTokenToId($0)! }
    }

    /// Encodes input text into token IDs with special tokens included by default.
    ///
    /// - Parameter text: The input text to encode
    /// - Returns: An array of token IDs
    public func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    /// Decodes token IDs back into human-readable text.
    ///
    /// - Parameters:
    ///   - tokens: The token IDs to decode
    ///   - skipSpecialTokens: Whether to exclude special tokens from the output text
    /// - Returns: The decoded text string
    public func decode(tokens: [Int], skipSpecialTokens: Bool = false) -> String {
        // IDs to tokens
        let tokenStrings: [String]
        if skipSpecialTokens {
            let specialTokenIDs = Set(specialTokens.values)
            tokenStrings =
                tokens
                .filter { !specialTokenIDs.contains($0) }
                .compactMap { model.convertIdToToken($0) }
        } else {
            tokenStrings = tokens.compactMap { model.convertIdToToken($0) }
        }
        let decoded = decodeTokens(tokenStrings)
        // At this point we should have a single String
        return cleanUp(text: decoded.joined(separator: ""))
    }

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID of the token, or nil if not found in the vocabulary
    public func convertTokenToId(_ token: String) -> Int? {
        model.convertTokenToId(token)
    }

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is invalid
    public func convertIdToToken(_ id: Int) -> String? {
        model.convertIdToToken(id)
    }

    /// Whether this tokenizer has a chat template configured.
    public var hasChatTemplate: Bool {
        !tokenizerConfig.chatTemplate.isNull()
    }

    public func applyChatTemplate(messages: [Message]) throws -> [Int] {
        try applyChatTemplate(messages: messages, addGenerationPrompt: true)
    }

    public func applyChatTemplate(messages: [Message], tools: [ToolSpec]? = nil) throws -> [Int] {
        try applyChatTemplate(messages: messages, addGenerationPrompt: true, tools: tools)
    }

    public func applyChatTemplate(messages: [Message], tools: [ToolSpec]? = nil, additionalContext: [String: Any]? = nil) throws
        -> [Int]
    {
        try applyChatTemplate(
            messages: messages,
            addGenerationPrompt: true,
            tools: tools,
            additionalContext: additionalContext
        )
    }

    public func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: true)
    }

    public func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: .literal(chatTemplate), addGenerationPrompt: true)
    }

    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument? = nil,
        addGenerationPrompt: Bool = false,
        truncation: Bool = false,
        maxLength: Int? = nil,
        tools: [ToolSpec]? = nil
    ) throws -> [Int] {
        try applyChatTemplate(
            messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: addGenerationPrompt, truncation: truncation, maxLength: maxLength,
            tools: tools, additionalContext: nil
        )
    }

    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument? = nil,
        addGenerationPrompt: Bool = false,
        truncation: Bool = false,
        maxLength: Int? = nil,
        // A list of tools (callable functions) that will be accessible to the model. If the template does not
        // support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
        // giving the name, description and argument types for the tool. See the
        // [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
        // for more information.
        tools: [ToolSpec]? = nil,
        additionalContext: [String: Any]? = nil
    ) throws -> [Int] {
        var selectedChatTemplate: String?
        if let chatTemplate, case let .literal(template) = chatTemplate {
            // Use chat template from argument
            selectedChatTemplate = template
        } else if !tokenizerConfig.chatTemplate.isNull() {
            let valueFromConfig: Config = tokenizerConfig.chatTemplate
            if let arrayValue = valueFromConfig.array() {
                // If the config specifies a list of chat templates, convert them to a dictionary
                let templateDict = [String: String](
                    uniqueKeysWithValues: arrayValue.compactMap { item in
                        guard let name = item["name"].string(), let template = item["template"].string() else {
                            return nil
                        }
                        return (name, template)
                    })
                if let chatTemplate, case let .name(name) = chatTemplate {
                    // Select chat template from config by name
                    if let matchingDictEntry = templateDict[name] {
                        selectedChatTemplate = matchingDictEntry
                    } else {
                        throw TokenizerError.chatTemplate("No chat template named \"\(name)\" was found in the tokenizer config")
                    }
                } else if let tools, !tools.isEmpty, let toolUseTemplate = templateDict["tool_use"] {
                    // Use tool use chat template from config
                    selectedChatTemplate = toolUseTemplate
                } else if let defaultChatTemplate = templateDict["default"] {
                    // Use default chat template from config
                    selectedChatTemplate = defaultChatTemplate
                }
            } else if let stringValue = valueFromConfig.string() {
                // Use chat template from config
                selectedChatTemplate = stringValue
            }
        }

        guard let selectedChatTemplate else {
            throw TokenizerError.missingChatTemplate
        }

        let template = try compiledTemplate(for: selectedChatTemplate)
        var context: [String: Jinja.Value] = try [
            "messages": .array(messages.map { try Value(any: $0) }),
            "add_generation_prompt": .boolean(addGenerationPrompt),
        ]
        if let tools {
            context["tools"] = try .array(tools.map { try Value(any: $0) })
        }
        if let additionalContext {
            // Additional keys and values to be added to the context provided to the prompt templating engine.
            // For example, the app could set "tools_in_user_message" to false for Llama 3.1 and 3.2 if a system message is provided.
            // The default value is true in the Llama 3.1 and 3.2 chat templates, but these models will perform better if the tools are included in a system message.
            for (key, value) in additionalContext {
                context[key] = try Value(any: value)
            }
        }

        for (key, value) in tokenizerConfig.dictionary(or: [:]) {
            if specialTokenAttributes.contains(key.string), !value.isNull() {
                if let stringValue = value.string() {
                    context[key.string] = .string(stringValue)
                } else if let dictionary = value.dictionary() {
                    if let addedTokenString = addedTokenAsString(Config(dictionary)) {
                        context[key.string] = .string(addedTokenString)
                    }
                } else if let array: [String] = value.get() {
                    context[key.string] = .array(array.map { .string($0) })
                } else {
                    context[key.string] = try Value(any: value)
                }
            }
        }

        let rendered = try template.render(context)
        var encodedTokens = encode(text: rendered, addSpecialTokens: false)
        var maxLength = maxLength ?? encodedTokens.count
        maxLength = min(maxLength, tokenizerConfig.modelMaxLength.integer() ?? maxLength)
        if encodedTokens.count > maxLength {
            if truncation {
                encodedTokens = Array(encodedTokens.prefix(maxLength))
            }
        }

        return encodedTokens
    }
}

// MARK: - Building

/// A namespace for automatically creating appropriate tokenizer instances.
///
/// `AutoTokenizer` provides static methods for loading pre-trained tokenizers
/// from the Hugging Face Hub or local directories. It automatically selects
/// the appropriate tokenizer class based on the configuration.
public enum AutoTokenizer {}

enum PreTrainedTokenizerClasses {
    /// Class overrides for custom behaviour
    /// Not to be confused with the TokenizerModel classes defined in TokenizerModel
    static let tokenizerClasses: [String: PreTrainedTokenizer.Type] = [
        "LlamaTokenizer": LlamaPreTrainedTokenizer.self
    ]
}

public extension AutoTokenizer {
    /// Determines the appropriate tokenizer class for the given configuration.
    ///
    /// - Parameter tokenizerConfig: The tokenizer configuration
    /// - Returns: The appropriate `PreTrainedTokenizer` subclass
    internal static func tokenizerClass(for tokenizerConfig: Config) -> PreTrainedTokenizer.Type {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass.string() else {
            return PreTrainedTokenizer.self
        }

        // Some tokenizer_class entries use a Fast suffix
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        if let tokenizerClass = PreTrainedTokenizerClasses.tokenizerClasses[tokenizerName] {
            return tokenizerClass
        }

        return PreTrainedTokenizer.self
    }

    /// Creates a tokenizer from configuration objects.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration (from tokenizer_config.json)
    ///   - tokenizerData: The tokenizer data (from tokenizer.json)
    ///   - strict: Whether to enforce strict validation
    /// - Returns: A configured `Tokenizer` instance
    /// - Throws: `TokenizerError` if configuration is invalid
    static func from(tokenizerConfig: Config, tokenizerData: Config, strict: Bool = true) throws -> Tokenizer {
        let tokenizerClass = tokenizerClass(for: tokenizerConfig)
        return try tokenizerClass.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, strict: strict)
    }

    /// Loads a tokenizer from a pre-trained model on the Hugging Face Hub.
    ///
    /// - Parameters:
    ///   - model: The model identifier (e.g., "bert-base-uncased")
    ///   - hubApi: The Hub API instance to use for downloading
    ///   - strict: Whether to enforce strict validation
    /// - Returns: A configured `Tokenizer` instance
    /// - Throws: `TokenizerError` if the model cannot be loaded or configured
    static func from(
        pretrained model: String,
        hubApi: HubApi = .shared,
        strict: Bool = true
    ) async throws -> Tokenizer {
        let config = LanguageModelConfigurationFromHub(modelName: model, hubApi: hubApi)
        guard let tokenizerConfig = try await config.tokenizerConfig else { throw TokenizerError.missingConfig }
        let tokenizerData = try await config.tokenizerData

        return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, strict: strict)
    }

    /// Loads a tokenizer from a local model folder.
    ///
    /// - Parameters:
    ///   - modelFolder: The URL path to the local model folder
    ///   - hubApi: The Hub API instance to use (unused for local loading)
    ///   - strict: Whether to enforce strict validation
    /// - Returns: A configured `Tokenizer` instance
    /// - Throws: `TokenizerError` if the model folder is invalid or missing files
    static func from(
        modelFolder: URL,
        hubApi: HubApi = .shared,
        strict: Bool = true
    ) async throws -> Tokenizer {
        let config = LanguageModelConfigurationFromHub(modelFolder: modelFolder, hubApi: hubApi)
        guard let tokenizerConfig = try await config.tokenizerConfig else { throw TokenizerError.missingConfig }
        let tokenizerData = try await config.tokenizerData

        return try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, strict: strict)
    }
}

// MARK: - Tokenizer model classes

class T5Tokenizer: UnigramTokenizer, @unchecked Sendable {}

// MARK: - PreTrainedTokenizer classes

let sentencePieceUnderline = "▁"

/// Hack for Llama tokenizers, see https://github.com/huggingface/transformers/blob/bcb841f0073fcd7a4fb88ea8064313c17dcab04a/src/transformers/models/llama/tokenization_llama_fast.py#L181
/// Return updated config, or nil
func maybeUpdatePostProcessor(tokenizerConfig: Config, processorConfig: Config?) throws -> Config? {
    // If it's already a Template processor (instead of a ByteLevel one), assume it's correct
    let postProcessor = PostProcessorFactory.fromConfig(config: processorConfig)
    guard !(postProcessor is TemplateProcessing) else { return nil }

    let addBosToken = tokenizerConfig.addBosToken.boolean(or: false)
    let bosToken = addedTokenAsString(tokenizerConfig.bosToken)
    if addBosToken, bosToken == nil {
        throw TokenizerError.mismatchedConfig("add_bos_token is True but bos_token is nil")
    }

    let addEosToken = tokenizerConfig.addEosToken.boolean(or: false)
    let eosToken = addedTokenAsString(tokenizerConfig.eosToken)
    if addEosToken, eosToken == nil {
        throw TokenizerError.mismatchedConfig("add_eos_token is True but eos_token is nil")
    }

    // alt implementation
    var single: [[String: Any]] = []
    if addBosToken {
        single = single + [["SpecialToken": ["id": bosToken!, "type_id": 0]]]
    }
    single = single + [["Sequence": ["id": "A", "type_id": 0]]]
    if addEosToken {
        single = single + [["SpecialToken": ["id": eosToken!, "type_id": 0]]]
    }

    var pair: [[String: Any]] = single
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

/// See https://github.com/xenova/transformers.js/blob/1a9964fb09b8f54fcbeac46dc6aae8d76795809d/src/tokenizers.js#L3203 for these exceptions
class LlamaPreTrainedTokenizer: PreTrainedTokenizer, @unchecked Sendable {
    let isLegacy: Bool

    required init(tokenizerConfig: Config, tokenizerData: Config, strict: Bool = true) throws {
        isLegacy = tokenizerConfig.legacy.boolean(or: true)
        var configDictionary = tokenizerData.dictionary(or: [:])
        if !isLegacy {
            _ = configDictionary.removeValue(forKey: "normalizer")
            configDictionary["pre_tokenizer"] = [
                "type": "Metaspace", "replacement": .init(sentencePieceUnderline), "add_prefix_space": true, "prepend_scheme": "first",
            ]
        }

        if let postProcessorConfig = try maybeUpdatePostProcessor(tokenizerConfig: tokenizerConfig, processorConfig: tokenizerData["postProcessor"]) {
            configDictionary["post_processor"] = .init(postProcessorConfig.dictionary(or: [:]))
        }

        let updatedData = Config(configDictionary)
        try super.init(tokenizerConfig: tokenizerConfig, tokenizerData: updatedData, strict: strict)
    }

    /// If `isLegacy` is `False`, a prefix token is added unless the first token is special.
    /// https://github.com/huggingface/transformers/blob/e6dcf8abd6f65bb4b6dfc1831b20d9ba49ce00e2/src/transformers/models/t5/tokenization_t5.py#L374-L387
    override func tokenize(text: String) -> [String] {
        if isLegacy || text.isEmpty {
            return super.tokenize(text: text)
        }

        let tokens = super.tokenize(text: sentencePieceUnderline + text.replacingOccurrences(of: sentencePieceUnderline, with: " "))
        if tokens.first == sentencePieceUnderline, let second = tokens.dropFirst().first, specialTokens[second] != nil {
            return Array(tokens[1...])
        }
        return tokens
    }
}
