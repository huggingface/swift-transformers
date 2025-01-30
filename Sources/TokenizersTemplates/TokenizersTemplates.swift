import Hub
import TokenizersCore
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
    public override func applyChatTemplate(messages: [[String: String]]) throws -> [Int] {
        try applyChatTemplate(messages: messages, addGenerationPrompt: true)
    }

    public override func applyChatTemplate(messages: [[String: String]], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: true)
    }

    public override func applyChatTemplate(messages: [[String: String]], chatTemplate: String) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: .literal(chatTemplate), addGenerationPrompt: true)
    }

    public override func applyChatTemplate(
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
        var selectedChatTemplate: String?
        if let chatTemplate, case .literal(let template) = chatTemplate {
            // Use chat template from argument
            selectedChatTemplate = template
        } else if let valueFromConfig = tokenizerConfig.chatTemplate {
            if let arrayValue = valueFromConfig.arrayValue {
                // If the config specifies a list of chat templates, convert them to a dictionary
                let templateDict = Dictionary<String, String>(uniqueKeysWithValues: arrayValue.compactMap { item in
                    guard let name = item.name?.stringValue, let template = item.template?.stringValue else {
                        return nil
                    }
                    return (name, template)
                })
                if let chatTemplate, case .name(let name) = chatTemplate {
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
            } else if let stringValue = valueFromConfig.stringValue {
                // Use chat template from config
                selectedChatTemplate = stringValue
            }
        }

        guard let selectedChatTemplate else {
            throw TokenizerError.chatTemplate("No chat template was specified")
        }

        let template = try Template(selectedChatTemplate)
        var context: [String: Any] = [
            "messages": messages,
            "add_generation_prompt": addGenerationPrompt
            // TODO: Add `tools` entry when support is added in Jinja
            // "tools": tools
        ]

        // TODO: maybe keep NSString here
        for (key, value) in tokenizerConfig.dictionary as [String : Any] {
            if specialTokenAttributes.contains(key), !(value is NSNull) {
                if let stringValue = value as? String {
                    context[key] = stringValue
                } else if let dictionary = value as? [NSString:Any] {
                    context[key] = addedTokenAsString(Config(dictionary))
                } else {
                    context[key] = value
                }
            }
        }

        let rendered = try template.render(context)
        var encodedTokens = encode(text: rendered, addSpecialTokens: false)
        var maxLength = maxLength ?? encodedTokens.count
        maxLength = min(maxLength, tokenizerConfig.modelMaxLength?.intValue ?? maxLength)
        if encodedTokens.count > maxLength {
            if truncation {
                encodedTokens = Array(encodedTokens.prefix(maxLength))
            }
        }

        return encodedTokens
    }
}
