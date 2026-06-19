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
public typealias Message = [String: any Sendable]

/// A type alias for tool specifications used in chat templating.
public typealias ToolSpec = [String: any Sendable]

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
public protocol TokenizingModel {
    func tokenize(text: String) -> [String]
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

/// Helper - possibly to be moved somewhere else
func addedTokenAsString(_ addedToken: Config?) -> String? {
    guard let addedToken else { return nil }
    if let stringValue = addedToken.string() {
        return stringValue
    }
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
public protocol PreTrainedTokenizerModel: TokenizingModel {
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
        "TokenizersBackend": BPETokenizer.self,
        "PreTrainedTokenizer": BPETokenizer.self,
        "Qwen2Tokenizer": BPETokenizer.self,
        "WhisperTokenizer": BPETokenizer.self,
        "XLMRobertaTokenizer": UnigramTokenizer.self,
        "Xlm-RobertaTokenizer": UnigramTokenizer.self,
    ]

    static func unknownToken(from tokenizerConfig: Config) -> String? {
        tokenizerConfig.unkToken.content.string() ?? tokenizerConfig.unkToken.string()
    }

    static func from(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int], strict: Bool = true) throws -> TokenizingModel {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass.string() else {
            throw TokenizerError.missingTokenizerClassInConfig
        }

        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
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
    case literal(String)
    case name(String)
}

// MARK: - Offset Mapping

/// A single encoded token paired with its character span in the original string.
///
/// Offsets are UTF-8 byte offsets (same unit as Python's `return_offsets_mapping=True`).
/// Special tokens (BOS, EOS, CLS, SEP, PAD) receive `(start: 0, end: 0)`.
@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)
public struct TokenWithOffsets: Equatable {
    /// Token id as produced by `encode(text:addSpecialTokens:)`.
    public let id: Int
    /// UTF-8 byte offset of the first character of this token's span (inclusive).
    public let start: Int
    /// UTF-8 byte offset past the last character of this token's span (exclusive).
    public let end: Int

    public init(id: Int, start: Int, end: Int) {
        self.id = id
        self.start = start
        self.end = end
    }
}

/// A complete tokenizer interface supporting encoding, decoding, and chat template functionality.
public protocol Tokenizer: Sendable {
    func tokenize(text: String) -> [String]

    func encode(text: String) -> [Int]

    func encode(text: String, addSpecialTokens: Bool) -> [Int]

    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int]

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

    var hasChatTemplate: Bool { get }

    /// Encode `text` and return token IDs together with UTF-8 byte offsets.
    ///
    /// Offsets match the semantics of Python's `return_offsets_mapping=True`.
    /// Special tokens receive `(start: 0, end: 0)`.
    ///
    /// - Parameters:
    ///   - text: The input string to tokenize.
    ///   - addSpecialTokens: Whether to prepend/append special tokens. Defaults to `true`.
    /// - Returns: Array of `TokenWithOffsets`, one per output token.
    @available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)
    func encodeWithOffsets(text: String, addSpecialTokens: Bool) -> [TokenWithOffsets]

    func applyChatTemplate(messages: [Message]) throws -> [Int]

    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int]

    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?) throws -> [Int]

    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int]

    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int]

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?
    ) throws -> [Int]

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int]
}

extension Tokenizer {
    public var hasChatTemplate: Bool { false }

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
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

// MARK: - Default encodeWithOffsets implementation

@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)
extension Tokenizer {

    /// Default implementation of `encodeWithOffsets` using a forward-scan strategy.
    ///
    /// Works for any tokenizer type (BPE, WordPiece, Unigram) without accessing
    /// tokenizer internals. O(N) in the length of `text`.
    public func encodeWithOffsets(text: String, addSpecialTokens: Bool = true) -> [TokenWithOffsets] {
        // 1. Encode WITHOUT special tokens so we can map content tokens to spans.
        let contentIds = encode(text: text, addSpecialTokens: false)

        // 2. Decode each content token back to its surface string.
        let contentStrings: [String] = contentIds.map { id in
            decode(tokens: [id], skipSpecialTokens: true)
        }

        // 3. Forward-scan: walk `text` as UTF-8 bytes, consuming each decoded string
        //    in sequence to record (start, end) byte offsets.
        var result: [TokenWithOffsets] = []
        result.reserveCapacity(contentIds.count + 4)

        let textUTF8 = Array(text.utf8)
        var bytePos: Int = 0

        for (id, surface) in zip(contentIds, contentStrings) {
            let surfaceUTF8 = Array(surface.utf8)
            guard !surfaceUTF8.isEmpty else {
                result.append(TokenWithOffsets(id: id, start: bytePos, end: bytePos))
                continue
            }

            // Skip leading bytes that don't match; handles GPT-2 Ġ → space prepend,
            // and SentencePiece leading-space on first token.
            var adjustedSurface = surfaceUTF8
            if bytePos == 0,
               adjustedSurface.first == UInt8(ascii: " "),
               textUTF8.first != UInt8(ascii: " ") {
                adjustedSurface = Array(adjustedSurface.dropFirst())
            }

            while bytePos < textUTF8.count && textUTF8[bytePos] != adjustedSurface[0] {
                bytePos += 1
            }

            let start = bytePos
            for byte in adjustedSurface {
                if bytePos < textUTF8.count && textUTF8[bytePos] == byte {
                    bytePos += 1
                }
            }
            let end = bytePos

            result.append(TokenWithOffsets(id: id, start: start, end: end))
        }

        // 4. Wrap with special tokens if requested.
        guard addSpecialTokens else { return result }

        let fullIds = encode(text: text, addSpecialTokens: true)
        guard fullIds.count != contentIds.count else {
            return result
        }

        var prefix = 0
        for (full, content) in zip(fullIds, contentIds) {
            if full == content { break }
            prefix += 1
        }
        let suffix = fullIds.count - contentIds.count - prefix

        let prefixTokens = fullIds.prefix(prefix).map { id in
            TokenWithOffsets(id: id, start: 0, end: 0)
        }
        let suffixTokens = fullIds.suffix(max(suffix, 0)).map { id in
            TokenWithOffsets(id: id, start: 0, end: 0)
        }

        return prefixTokens + result + suffixTokens
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

    private var compiledChatTemplateCache: [String: Template] = [:]
    private let cacheLock = NSLock()

    public required init(tokenizerConfig: Config, tokenizerData: Config, strict: Bool = true) throws {
        var addedTokens: [String: Int] = [:]
        var specialTokens: [String: Int] = [:]
        for addedToken in tokenizerData["addedTokens"].array(or: []) {
            guard let id = addedToken["id"].integer() else { continue }
            guard let content = addedToken.content.string() else { continue }
            addedTokens[content] = id

            if addedToken["special"].boolean(or: false) {
                specialTokens[content] = id
            }
        }

        let unwrappedAddedTokens: [(content: String, prefix: Bool, suffix: Bool)] = (tokenizerData["addedTokens"].array(or: [])).compactMap { addedToken -> (String, Bool, Bool)? in
            guard let content = addedToken.content.string() else { return nil }
            let prefix = addedToken["lstrip"].boolean(or: false)
            let suffix = addedToken["rstrip"].boolean(or: false)
            return (content: content, prefix: prefix, suffix: suffix)
        }.sorted {
            $0.content.count > $1.content.count
        }

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
        cacheLock.lock()
        if let cached = compiledChatTemplateCache[templateString] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()

        let compiled = try Template(templateString, with: .init(lstripBlocks: true, trimBlocks: true))

        cacheLock.lock()
        defer { cacheLock.unlock() }

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

    public func tokenize(text: String) -> [String] {
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

    public func encode(text: String, addSpecialTokens: Bool = true) -> [Int] {
        postProcess(tokenize(text: text), addSpecialTokens: addSpecialTokens).map { model.convertTokenToId($0)! }
    }

    public func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    public func decode(tokens: [Int], skipSpecialTokens: Bool = false) -> String {
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
        return cleanUp(text: decoded.joined(separator: ""))
    }

    public func convertTokenToId(_ token: String) -> Int? {
        model.convertTokenToId(token)
    }

    public func convertIdToToken(_ id: Int) -> String? {
        model.convertIdToToken(id)
    }

    public var hasChatTemplate: Bool {
        !tokenizerConfig.chatTemplate.isNull()
    }

    public func applyChatTemplate(messages: [Message]) throws -> [Int] {
        try applyChatTemplate(messages: messages, addGenerationPrompt: true)
    }

    public func applyChatTemplate(messages: [Message], tools: [ToolSpec]? = nil) throws -> [Int] {
        try applyChatTemplate(messages: messages, addGenerationPrompt: true, tools: tools)
    }

    public func applyChatTemplate(messages: [Message], tools: [ToolSpec]? = nil, additionalContext: [String: any Sendable]? = nil) throws
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
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [Int] {
        var selectedChatTemplate: String?
        if let chatTemplate, case let .literal(template) = chatTemplate {
            selectedChatTemplate = template
        } else if !tokenizerConfig.chatTemplate.isNull() {
            let valueFromConfig: Config = tokenizerConfig.chatTemplate
            if let arrayValue = valueFromConfig.array() {
                let templateDict = [String: String](
                    uniqueKeysWithValues: arrayValue.compactMap { item in
                        guard let name = item["name"].string(), let template = item["template"].string() else {
                            return nil
                        }
                        return (name, template)
                    })
                if let chatTemplate, case let .name(name) = chatTemplate {
                    if let matchingDictEntry = templateDict[name] {
                        selectedChatTemplate = matchingDictEntry
                    } else {
                        throw TokenizerError.chatTemplate("No chat template named \"\(name)\" was found in the tokenizer config")
                    }
                } else if let tools, !tools.isEmpty, let toolUseTemplate = templateDict["tool_use"] {
                    selectedChatTemplate = toolUseTemplate
                } else if let defaultChatTemplate = templateDict["default"] {
                    selectedChatTemplate = defaultChatTemplate
                }
            } else if let stringValue = valueFromConfig.string() {
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

public enum AutoTokenizer {}

enum PreTrainedTokenizerClasses {
    static let tokenizerClasses: [String: PreTrainedTokenizer.Type] = [
        "LlamaTokenizer": LlamaPreTrainedTokenizer.self
    ]
}

public extension AutoTokenizer {
    internal static func tokenizerClass(for tokenizerConfig: Config) -> PreTrainedTokenizer.Type {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass.string() else {
            return PreTrainedTokenizer.self
        }

        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        if let tokenizerClass = PreTrainedTokenizerClasses.tokenizerClasses[tokenizerName] {
            return tokenizerClass
        }

        return PreTrainedTokenizer.self
    }

    static func from(tokenizerConfig: Config, tokenizerData: Config, strict: Bool = true) throws -> Tokenizer {
        let tokenizerClass = tokenizerClass(for: tokenizerConfig)
        return try tokenizerClass.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, strict: strict)
    }

    static func from(
        pretrained model: String,
        hubApi: HubApi = .shared,
        strict: Bool = true
    ) async throws -> Tokenizer {
        try await from(pretrained: model, revision: "main", hubApi: hubApi, strict: strict)
    }

    static func from(
        pretrained model: String,
        revision: String,
        hubApi: HubApi = .shared,
        strict: Bool = true
    ) async throws -> Tokenizer {
        let config = LanguageModelConfigurationFromHub(modelName: model, revision: revision, hubApi: hubApi)
        guard let tokenizerConfig = try await config.tokenizerConfig else { throw TokenizerError.missingConfig }
        let tokenizerData = try await config.tokenizerData

        return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, strict: strict)
    }

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

func maybeUpdatePostProcessor(tokenizerConfig: Config, processorConfig: Config?) throws -> Config? {
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

#if !canImport(Darwin)
private extension String {
    init(localized key: String, comment: String? = nil) {
        self = key
    }
}
#endif
