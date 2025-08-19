import Hub
@_exported import TokenizersCore
import Jinja
import Foundation

let specialTokenAttributes: [String] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens"
]

open class PreTrainedTokenizerWithTemplates : PreTrainedTokenizer {
    // I don't know why these need to be here. They are implemented in the protocol, **and** in the superclass.
    public override func applyChatTemplate(messages: [Message]) throws -> [Int] {
        try applyChatTemplate(messages: messages, addGenerationPrompt: true)
    }

    public override func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: true)
    }

    public override func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: .literal(chatTemplate), addGenerationPrompt: true)
    }

    public override func applyChatTemplate(
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

    public override func applyChatTemplate(
        messages: [Message],
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
        tools: [[String: Any]]? = nil,
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

        let template = try Template(selectedChatTemplate)
        var context: [String: Any] = [
            "messages": messages,
            "add_generation_prompt": addGenerationPrompt,
        ]
        if let tools {
            context["tools"] = tools
        }
        if let additionalContext {
            /*
             Additional keys and values to be added to the context provided to the prompt templating engine.
             For example, the app could set "tools_in_user_message" to false for Llama 3.1 and 3.2 if a system message is provided.
             The default value is true in the Llama 3.1 and 3.2 chat templates, but these models will perform better if the tools are included in a system message.
             */
            for (key, value) in additionalContext {
                context[key] = value
            }
        }

        for (key, value) in tokenizerConfig.dictionary(or: [:]) {
            if specialTokenAttributes.contains(key.string), !value.isNull() {
                if let stringValue = value.string() {
                    context[key.string] = stringValue
                } else if let dictionary = value.dictionary() {
                    context[key.string] = addedTokenAsString(Config(dictionary))
                } else if let array: [String] = value.get() {
                    context[key.string] = array
                } else {
                    context[key.string] = value
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

// Template-enabled tokenizer classes
/// See https://github.com/xenova/transformers.js/blob/1a9964fb09b8f54fcbeac46dc6aae8d76795809d/src/tokenizers.js#L3203 for these exceptions
class LlamaPreTrainedTokenizerWithTemplates: PreTrainedTokenizerWithTemplates {
    let isLegacy: Bool

    required init(tokenizerConfig: Config, tokenizerData: Config) throws {
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
        try super.init(tokenizerConfig: tokenizerConfig, tokenizerData: updatedData)
    }
}

struct PreTrainedTokenizerTemplateClasses {
    /// Template-enabled class overrides
    static let tokenizerClasses: [String: PreTrainedTokenizerWithTemplates.Type] = [
        "LlamaTokenizer": LlamaPreTrainedTokenizerWithTemplates.self,
    ]
}

// Override AutoTokenizer to use template-enabled tokenizers
// See Sources/Tokenizers/Tokenizer.swift
public extension AutoTokenizer {
    private static func tokenizerClassWithTemplates(for tokenizerConfig: Config) -> PreTrainedTokenizerWithTemplates.Type {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass?.string() else {
            return PreTrainedTokenizerWithTemplates.self
        }

        // Some tokenizer_class entries use a Fast suffix
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        if let tokenizerClass = PreTrainedTokenizerTemplateClasses.tokenizerClasses[tokenizerName] {
            return tokenizerClass
        }

        return PreTrainedTokenizerWithTemplates.self
    }

    static func from(tokenizerConfig: Config, tokenizerData: Config) throws -> any Tokenizer {
        let tokenizerClass = tokenizerClassWithTemplates(for: tokenizerConfig)
        return try tokenizerClass.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    static func from(
        pretrained model: String,
        hubApi: HubApi = .shared
    ) async throws -> any Tokenizer {
        let config = LanguageModelConfigurationFromHub(modelName: model, hubApi: hubApi)
        guard let tokenizerConfig = try await config.tokenizerConfig else { throw TokenizerError.missingConfig }
        let tokenizerData = try await config.tokenizerData

        return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    static func from(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) async throws -> any Tokenizer {
        let config = LanguageModelConfigurationFromHub(modelFolder: modelFolder, hubApi: hubApi)
        guard let tokenizerConfig = try await config.tokenizerConfig else { throw TokenizerError.missingConfig }
        let tokenizerData = try await config.tokenizerData

        return try PreTrainedTokenizerWithTemplates(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }
}
