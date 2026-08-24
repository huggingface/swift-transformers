import Foundation
import Hub
import TokenizersCore

public extension AutoTokenizer {
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
        try await from(pretrained: model, revision: "main", hubApi: hubApi, strict: strict)
    }

    /// Loads a tokenizer from a pre-trained model on the Hugging Face Hub at a specific revision.
    ///
    /// - Parameters:
    ///   - model: The model identifier (e.g., "bert-base-uncased")
    ///   - revision: Git revision to load — a branch, tag, commit SHA, or PR ref like `"refs/pr/1"`.
    ///   - hubApi: The Hub API instance to use for downloading
    ///   - strict: Whether to enforce strict validation
    /// - Returns: A configured `Tokenizer` instance
    /// - Throws: `TokenizerError` if the model cannot be loaded or configured
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
