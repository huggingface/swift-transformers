//
//  LanguageModel.swift
//
//
//  Created by Pedro Cuenca on 7/5/23.
//

#if canImport(CoreML)
import CoreML

import Generation
import Hub
import Tokenizers

/// A high-level interface for language model inference using CoreML.
///
/// `LanguageModel` provides a convenient way to load and interact with pre-trained
/// language models that have been converted to CoreML format. It handles model
/// initialization, input/output processing, and context length management.
public class LanguageModel {
    /// The underlying CoreML model used for inference.
    public let model: MLModel

    /// The minimum context length supported by the model.
    public let minContextLength: Int

    /// The maximum context length supported by the model.
    public let maxContextLength: Int

    let input_ids = "input_ids"
    let attention_mask = "attention_mask"

    struct Configurations {
        var modelConfig: Config
        var tokenizerConfig: Config?
        var tokenizerData: Config
    }

    private var configuration: LanguageModelConfigurationFromHub?
    private var _tokenizer: Tokenizer?

    /// Creates a new language model instance from a CoreML model.
    ///
    /// - Parameter model: The CoreML model to wrap
    /// - Important: Triggers a fatal error if the model doesn't have the expected input shape information
    public required init(model: MLModel) {
        self.model = model

        // We assume inputs named "input_ids" with shape (1, seq_length)
        // Perhaps we should convert to vectors of shape (seq_length) and use sequenceConstraint instead of shapeConstraint
        let inputDescription = model.modelDescription.inputDescriptionsByName["input_ids"]

        guard let shapeConstraint = inputDescription?.multiArrayConstraint?.shapeConstraint else {
            fatalError("Cannot obtain shape information")
        }

        switch shapeConstraint.type {
        case .enumerated:
            // TODO: support a set of fixed shapes (keeping the first one here)
            minContextLength = shapeConstraint.enumeratedShapes[0][1].intValue
            maxContextLength = minContextLength
        case .range:
            let range = inputDescription?.multiArrayConstraint?.shapeConstraint.sizeRangeForDimension[1] as? NSRange
            minContextLength = range?.location ?? 1
            maxContextLength = range?.length ?? 128
        case .unspecified:
            minContextLength = 128
            maxContextLength = 128
        @unknown default:
            minContextLength = 128
            maxContextLength = 128
        }

        configuration = LanguageModelConfigurationFromHub(modelName: modelName)
    }
}

public extension LanguageModel {
    /// Loads a compiled CoreML model from disk.
    ///
    /// - Parameters:
    ///   - url: The URL of the compiled CoreML model file (.mlmodelc)
    ///   - computeUnits: The compute units to use for model inference
    /// - Returns: A configured `LanguageModel` instance
    /// - Throws: An error if the model cannot be loaded from the specified URL
    static func loadCompiled(url: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws -> LanguageModel {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let model = try MLModel(contentsOf: url, configuration: config)
        return LanguageModel(model: model)
    }
}

public extension LanguageModel {
    /// A human-readable description of the model.
    ///
    /// Returns the model's description from its metadata, or the display name if no description is available.
    var description: String {
        if let description = model.modelDescription.metadata[MLModelMetadataKey.description] as? String,
            !description.isEmpty
        {
            return description
        }
        return model.configuration.modelDisplayName ?? ""
    }

    /// The name or path of the model.
    ///
    /// Returns the model identifier from Hugging Face Hub metadata if available,
    /// otherwise falls back to the model's display name.
    var modelName: String {
        if let userFields = model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String],
            let name = userFields["co.huggingface.exporters.name"]
        {
            return name
        }
        // This is usually the basename of the file, that's our best bet if no metadata exists
        guard let modelName = model.configuration.modelDisplayName else { fatalError("Models must have a name that identifies them") }
        return modelName
    }

    /// The feature description for the input_ids input.
    var inputIdsDescription: MLFeatureDescription {
        model.modelDescription.inputDescriptionsByName[input_ids]!
    }

    /// The name of the input_ids feature in the model.
    var inputIdsName: String {
        inputIdsDescription.name
    }

    /// The expected shape of the input_ids tensor.
    var inputIdsShape: [Int] {
        inputIdsDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }

    /// Whether the model requires attention mask inputs.
    var requiresAttention: Bool {
        model.modelDescription.inputDescriptionsByName[attention_mask] != nil
    }

    /// Predicts the next token scores for the given input tokens.
    ///
    /// - Parameters:
    ///   - tokens: The input token sequence
    ///   - config: The generation configuration containing model parameters
    /// - Returns: A shaped array containing the logits for the next token prediction
    func predictNextTokenScores(_ tokens: InputTokens, config: GenerationConfig) -> any MLShapedArrayProtocol {
        // TODO: exceptions

        // Maybe pad or truncate
        let maxTokens = min(tokens.count, maxContextLength)
        let padLength = maxTokens >= minContextLength ? 0 : minContextLength - maxTokens
        let inputTokens = Array(tokens[0..<maxTokens]) + Array(repeating: config.padTokenId ?? 0, count: padLength)

        let inputIds = MLShapedArray<Int32>(scalars: inputTokens.map { Int32($0) }, shape: inputIdsShape)
        var inputDictionary = [inputIdsName: MLFeatureValue(shapedArray: inputIds)]
        if requiresAttention {
            let mask = Array(repeating: 1, count: maxTokens) + Array(repeating: 0, count: padLength)
            let attentionMask = MLShapedArray<Int32>(scalars: mask.map { Int32($0) }, shape: inputIdsShape)
            inputDictionary[attention_mask] = MLFeatureValue(shapedArray: attentionMask)
        }
        let input = try! MLDictionaryFeatureProvider(dictionary: inputDictionary)

        let output = try! model.prediction(from: input)

        // TODO: maybe try to support models with "token_scores" too (after the softmax)
        assert(output.featureNames.first! == "logits")

        let scores = output.featureValue(for: output.featureNames.first!)!.shapedArrayValue(of: Float.self)!
        let nextTokenScores = scores[0, maxTokens - 1]
        return nextTokenScores
    }
}

// MARK: - Configuration Properties

/// Asynchronous properties that are downloaded from the Hugging Face Hub configuration.
public extension LanguageModel {
    /// The model configuration dictionary.
    ///
    /// - Returns: The model's configuration as parsed from config.json
    /// - Throws: An error if the configuration cannot be loaded
    var modelConfig: Config? {
        get async throws {
            try await configuration!.modelConfig
        }
    }

    /// The tokenizer configuration dictionary.
    ///
    /// - Returns: The tokenizer configuration if available, nil otherwise
    /// - Throws: An error if the configuration cannot be loaded
    var tokenizerConfig: Config? {
        get async throws {
            try await configuration!.tokenizerConfig
        }
    }

    /// The tokenizer data dictionary containing vocabulary and merges.
    ///
    /// - Returns: The tokenizer data configuration
    /// - Throws: An error if the tokenizer data cannot be loaded
    var tokenizerData: Config {
        get async throws {
            try await configuration!.tokenizerData
        }
    }

    /// The model architecture type.
    ///
    /// - Returns: A string identifying the model type (e.g., "llama", "gpt2")
    /// - Throws: An error if the model configuration cannot be accessed
    var modelType: String? {
        get async throws {
            try await modelConfig?.modelType.string()
        }
    }

    /// Text generation specific parameters from the model configuration.
    ///
    /// - Returns: Configuration parameters for text generation, if specified
    /// - Throws: An error if the model configuration cannot be accessed
    var textGenerationParameters: Config? {
        get async throws {
            try await modelConfig?.taskSpecificParams.textGeneration
        }
    }

    /// The default sampling behavior for this model.
    ///
    /// - Returns: Whether sampling should be used by default for text generation
    /// - Throws: An error if the configuration cannot be accessed
    var defaultDoSample: Bool {
        get async throws {
            try await textGenerationParameters?.doSample.boolean() ?? true
        }
    }

    /// The beginning-of-sequence token ID.
    ///
    /// - Returns: The BOS token ID if specified in the configuration
    /// - Throws: An error if the model configuration cannot be accessed
    var bosTokenId: Int? {
        get async throws {
            let modelConfig = try await modelConfig
            return modelConfig?.bosTokenId.integer()
        }
    }

    /// The end-of-sequence token ID.
    ///
    /// - Returns: The EOS token ID if specified in the configuration
    /// - Throws: An error if the model configuration cannot be accessed
    var eosTokenId: Int? {
        get async throws {
            let modelConfig = try await modelConfig
            return modelConfig?.eosTokenId.integer()
        }
    }

    /// The tokenizer associated with this model.
    ///
    /// Lazily loads and caches the tokenizer on first access.
    ///
    /// - Returns: A configured tokenizer instance
    /// - Throws: `TokenizerError.tokenizerConfigNotFound` if tokenizer configuration is missing,
    ///           or other errors during tokenizer creation
    var tokenizer: Tokenizer {
        get async throws {
            guard _tokenizer == nil else { return _tokenizer! }
            guard let tokenizerConfig = try await tokenizerConfig else {
                throw TokenizerError.tokenizerConfigNotFound
            }
            let tokenizerData = try await tokenizerData
            _tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
            return _tokenizer!
        }
    }
}

// MARK: - TextGenerationModel Conformance

extension LanguageModel: TextGenerationModel {
    /// The default generation configuration for this model.
    ///
    /// Provides sensible defaults based on the model type, with model-specific
    /// optimizations for known architectures like GPT models.
    public var defaultGenerationConfig: GenerationConfig {
        var config = GenerationConfig(maxNewTokens: 30)
        switch modelName.lowercased() {
        case let x where x.contains("gpt"):
            config.doSample = true
            config.topK = 50
        default: break
        }
        return config
    }
}

/// Errors that can occur during tokenizer operations in language models.
public enum TokenizerError: LocalizedError {
    /// The tokenizer configuration file could not be found.
    case tokenizerConfigNotFound

    public var errorDescription: String? {
        switch self {
        case .tokenizerConfigNotFound:
            String(localized: "Tokenizer configuration could not be found. The model may be missing required tokenizer files.", comment: "Error when tokenizer configuration is missing")
        }
    }
}

#endif // canImport(CoreML)
