import Foundation
import Hub

@_exported import TokenizersCore

#if canImport(TokenizersTemplates)
import TokenizersTemplates
public typealias PreTrainedTokenizer = PreTrainedTokenizerWithTemplates
#endif

public struct AutoTokenizer {}

struct PreTrainedTokenizerClasses {
    /// Class overrides for custom behaviour
    /// Not to be confused with the TokenizerModel classes defined in TokenizerModel
    static let tokenizerClasses: [String : PreTrainedTokenizer.Type] = [
        "LlamaTokenizer": LlamaPreTrainedTokenizer.self
    ]
}

extension AutoTokenizer {
    static func tokenizerClass(for tokenizerConfig: Config) -> PreTrainedTokenizer.Type {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass?.string() else {
            return PreTrainedTokenizer.self
        }

        // Some tokenizer_class entries use a Fast suffix
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        if let tokenizerClass = PreTrainedTokenizerClasses.tokenizerClasses[tokenizerName] {
            return tokenizerClass
        }

        return PreTrainedTokenizer.self
    }

    public static func from(tokenizerConfig: Config, tokenizerData: Config) throws -> Tokenizer {
        let tokenizerClass = tokenizerClass(for: tokenizerConfig)
        return try tokenizerClass.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    public static func from(
        pretrained model: String,
        hubApi: HubApi = .shared
    ) async throws -> Tokenizer {
        let config = LanguageModelConfigurationFromHub(modelName: model, hubApi: hubApi)
        guard let tokenizerConfig = try await config.tokenizerConfig else { throw TokenizerError.missingConfig }
        let tokenizerData = try await config.tokenizerData

        return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
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

// See https://github.com/xenova/transformers.js/blob/1a9964fb09b8f54fcbeac46dc6aae8d76795809d/src/tokenizers.js#L3203 for these exceptions
class LlamaPreTrainedTokenizer: PreTrainedTokenizer {
    let isLegacy: Bool

    required init(tokenizerConfig: Config, tokenizerData: Config) throws {
        isLegacy = tokenizerConfig.legacy?.boolean() ?? true
        var configDictionary = tokenizerData.dictionary(or: [:])
        if !isLegacy {
            configDictionary.removeValue(forKey: "normalizer")
            configDictionary["pre_tokenizer"] = Config(["type": "Metaspace", "replacement": sentencePieceUnderline, "add_prefix_space": true, "prepend_scheme": "first"])
        }

        if let postProcessorConfig = try maybeUpdatePostProcessor(tokenizerConfig: tokenizerConfig, processorConfig: tokenizerData.postProcessor) {
            configDictionary["post_processor"] = postProcessorConfig
        }

        let updatedData = Config(configDictionary)
        try super.init(tokenizerConfig: tokenizerConfig, tokenizerData: updatedData)
    }
}

