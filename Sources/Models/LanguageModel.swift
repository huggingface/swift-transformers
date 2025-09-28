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

@available(macOS 15.0, iOS 18.0, *)
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

    private var configuration: LanguageModelConfigurationFromHub?
    private var _tokenizer: Tokenizer?

    /// Creates a new language model instance from a CoreML model.
    ///
    /// - Parameter model: The CoreML model to wrap
    /// - Important: Triggers a fatal error if the model doesn't have the expected input shape information
    public required init(model: MLModel) {
        self.model = model
        (minContextLength, maxContextLength) = Self.contextRange(from: model)
        configuration = LanguageModelConfigurationFromHub(modelName: modelName)
    }

    public func resetState() async {}

    public func predictNextTokenScores(
        _ tokens: MLTensor,
        config: GenerationConfig
    ) async -> MLTensor {
        assert(tokens.rank == 2) // [batch, current sequence length]
        let tokenCount = tokens.shape[1]
        let padLength = maxContextLength - tokenCount
        let padding = MLTensor(repeating: Int32(config.padTokenId ?? 0), shape: [1, padLength])
        let inputIDs = MLTensor(concatenating: [tokens, padding], alongAxis: -1)
        var inputDictionary = [inputIdsName: inputIDs]
        if isRequiringAttentionMask {
            let mask = [Int32](repeating: 1, count: tokenCount) + [Int32](repeating: 0, count: padLength)
            let attentionMask = MLTensor(shape: inputIDs.shape, scalars: mask)
            inputDictionary[Keys.attentionMask] = attentionMask
        }
        let outputs = try! await model.prediction(from: inputDictionary)

        assert(outputs.keys.contains(Keys.logits))

        let scores = outputs[Keys.logits]!
        assert(scores.rank == 3)
        let tokenIndex = tokenCount - 1
        let nextTokenScores = scores[nil, tokenIndex, nil].expandingShape(at: 0)
        assert(nextTokenScores.rank == 3)
        assert(nextTokenScores.shape[0] == 1 && nextTokenScores.shape[1] == 1)
        return nextTokenScores
    }
}

@available(macOS 15.0, iOS 18.0, *)
private extension LanguageModel {
    static func contextRange(from model: MLModel) -> (min: Int, max: Int) {
        contextRange(from: model, inputKey: Keys.inputIds)
    }

    static func contextRange(from model: MLModel, inputKey: String) -> (min: Int, max: Int) {
        let inputDescription = model.modelDescription.inputDescriptionsByName[inputKey]

        guard let shapeConstraint = inputDescription?.multiArrayConstraint?.shapeConstraint else {
            fatalError("Cannot obtain shape information")
        }

        var minContextLength = 128
        var maxContextLength = 128

        switch shapeConstraint.type {
        case .enumerated:
            minContextLength = shapeConstraint.enumeratedShapes[0][1].intValue
            maxContextLength = minContextLength
        case .range:
            if let sizeRangeForDimension = inputDescription?.multiArrayConstraint?.shapeConstraint.sizeRangeForDimension {
                let lastAxis = sizeRangeForDimension.count - 1
                let range = sizeRangeForDimension[lastAxis] as? NSRange
                minContextLength = range?.location ?? 1
                maxContextLength = range?.length ?? 128
            }
        case .unspecified:
            break
        @unknown default:
            break
        }

        return (minContextLength, maxContextLength)
    }
}

@available(macOS 15.0, iOS 18.0, *)
extension LanguageModel {
    struct Configurations {
        var modelConfig: Config
        var tokenizerConfig: Config?
        var tokenizerData: Config
    }
}

@available(macOS 15.0, iOS 18.0, *)
extension LanguageModel {
    enum Keys {
        // Input keys
        static let inputIds = "inputIds"
        static let attentionMask = "attentionMask"
        static let causalMask = "causalMask"
        static let keyCache = "keyCache"
        static let valueCache = "valueCache"
        // Output keys
        static let logits = "logits"
        // swift-format-ignore: DontRepeatTypeInStaticProperties
        static let presentKeys = "presentKeys"
        static let presentValues = "presentValues"
    }
}

@available(macOS 15.0, iOS 18.0, *)
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
        return switch kvCacheAvailability(for: model) {
        case .statefulKVCache: LanguageModelWithStatefulKVCache(model: model)
        default: LanguageModel(model: model)
        }
    }
}

@available(macOS 15.0, iOS 18.0, *)
extension LanguageModel {
    enum KVCacheAvailability {
        /// Language models that support KV cache via state. Implementation details for handling state
        /// encapsulated within the Core ML framework.
        ///
        /// Input: State
        /// Output: N/A
        case statefulKVCache
    }
}

@available(macOS 15.0, iOS 18.0, *)
public extension LanguageModel {
    /// Metadata fields associated to the Core ML model.
    var metadata: [MLModelMetadataKey: Any] {
        model.modelDescription.metadata
    }

    /// A description of a model containing input, output, and state feature descriptions.
    ///
    /// Returns a MLModelDescription instance.
    var modelDescription: MLModelDescription {
        model.modelDescription
    }

    /// A human-readable description of the model.
    ///
    /// Returns the model's description from its metadata, or the display name if no description is available.
    var description: String {
        if let description = metadata[MLModelMetadataKey.description] as? String,
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
        if let userFields = metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String],
            let name = userFields["co.huggingface.exporters.name"]
        {
            return name
        }
        // This is usually the basename of the file, that's our best bet if no metadata exists
        guard let modelName = model.configuration.modelDisplayName else {
            fatalError("Models must have a name that identifies them")
        }
        return modelName
    }

    /// The feature description for the input_ids input.
    var inputIdsDescription: MLFeatureDescription {
        modelDescription.inputDescriptionsByName[Keys.inputIds]!
    }

    /// The name of the input_ids feature in the model.
    var inputIdsName: String {
        inputIdsDescription.name
    }

    /// The expected shape of the input_ids tensor.
    var inputIdsShape: [Int] {
        inputIdsDescription.multiArrayConstraint!.shape.map(\.intValue)
    }

    /// Whether the model requires attention mask inputs.
    var isRequiringAttentionMask: Bool {
        modelDescription.inputDescriptionsByName[Keys.attentionMask] != nil
    }

    /// Whether the model requires a causal attention mask.
    var isRequiringCausalMask: Bool {
        modelDescription.inputDescriptionsByName[Keys.causalMask] != nil
    }

    /// Determines the type of KV Cache available for the model, if any.
    ///
    /// - Parameter model: The Core ML model
    /// - Returns: The type of KV Cache available.
    fileprivate static func kvCacheAvailability(for model: MLModel) -> KVCacheAvailability? {
        func isStatefulKVCacheAvailable(for model: MLModel) -> Bool {
            let kCacheState = model.modelDescription.stateDescriptionsByName[Keys.keyCache] != nil
            let vCacheState = model.modelDescription.stateDescriptionsByName[Keys.valueCache] != nil
            guard Set([kCacheState, vCacheState]).count == 1 else {
                fatalError("Invalid model configuration, expecting KV cache for states")
            }
            return kCacheState && kCacheState
        }

        func isDynamicallyShaped(_ description: MLFeatureDescription) -> Bool {
            guard let multiArrayConstraint = description.multiArrayConstraint else {
                return false
            }
            return switch multiArrayConstraint.shapeConstraint.type {
            case .unspecified: true
            case .enumerated: multiArrayConstraint.shapeConstraint.enumeratedShapes.count > 1
            case .range: true
            default: false
            }
        }

        if isStatefulKVCacheAvailable(for: model) {
            return .statefulKVCache
        }
        let kCacheInput = model.modelDescription.inputDescriptionsByName[Keys.keyCache] != nil
        let vCacheInput = model.modelDescription.inputDescriptionsByName[Keys.valueCache] != nil
        let kCacheOutput = model.modelDescription.outputDescriptionsByName[Keys.presentKeys] != nil
        let vCacheOutput = model.modelDescription.outputDescriptionsByName[Keys.presentValues] != nil

        guard Set([kCacheInput, vCacheInput, kCacheOutput, vCacheOutput]).count == 1 else {
            fatalError("Invalid model configuration, expecting KV cache for inputs and outputs")
        }
        guard kCacheInput else {
            return nil
        }
        // Check if cache is dynamic or not.
        let kCacheConstraint = model.modelDescription.inputDescriptionsByName[Keys.keyCache]!
        if isDynamicallyShaped(kCacheConstraint) {
            fatalError(
                """
                KV Cache using IO is currently not supported, please file a feature request on \
                https://github.com/huggingface/swift-transformers
                """)
        } else {
            fatalError(
                """
                KV Cache using IO is currently not supported, please file a feature request on \
                https://github.com/huggingface/swift-transformers
                """)
        }
    }
}

// MARK: - Configuration Properties

/// Asynchronous properties that are downloaded from the Hugging Face Hub configuration.
@available(macOS 15.0, iOS 18.0, *)
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

@available(macOS 15.0, iOS 18.0, *)
extension LanguageModel: TextGenerationModel {
    /// The default generation configuration for this model.
    ///
    /// Provides sensible defaults based on the model type, with model-specific
    /// optimizations for known architectures like GPT models.
    public var defaultGenerationConfig: GenerationConfig {
        var config = GenerationConfig(maxNewTokens: 2048)
        switch modelName.lowercased() {
        case let x where x.contains("gpt"):
            config.doSample = true
            config.topK = 50
        default: break
        }
        return config
    }
}

/// Language model implementation with stateful KV Cache.
///
/// Maintains a KV Cache as sequence generation progresses,
/// using stateful Core ML buffers to minimize latency.
@available(macOS 15.0, iOS 18.0, *)
public class LanguageModelWithStatefulKVCache: LanguageModel {
    private enum Mode {
        case prefilling
        case extending
    }
    private var mode: Mode = .prefilling

    var state: MLState?

    public required init(model: MLModel) {
        super.init(model: model)
        // To support pre-filling and extend, the input must support
        // flexible shapes.
        guard maxContextLength - minContextLength > 1 else {
            fatalError("Expecting ranged query sequence length.")
        }
    }

    public override func resetState() async {
        state = model.makeState()
        mode = .prefilling
    }

    public override func predictNextTokenScores(
        _ tokens: MLTensor,
        config _: GenerationConfig
    ) async -> MLTensor {
        assert(tokens.rank == 2) // [batch, current sequence length]
        let tokenCount = tokens.shape[1]
        guard let state else {
            fatalError(
                """
                Encountered uninitialized `state`. Ensure `resetState` is called prior to calling \
                `predictNextTokenScores`. 
                """)
        }
        let inputIds =
            switch mode {
            case .prefilling: tokens // Pass in all takens if pre-filling prompt
            case .extending: tokens[nil, -1].expandingShape(at: 0) // otherwise just the last token
            }
        mode = .extending

        var inputDictionary = [
            Keys.inputIds: inputIds
        ]
        if isRequiringAttentionMask {
            #if !((os(macOS) || (macCatalyst)) && arch(x86_64))
            // TODO: Infer scalar type from cache or model I/O descriptors
            let attentionMask = MLTensor(zeros: [1, 1, 1, tokenCount + 1], scalarType: Float16.self)
            inputDictionary[Keys.attentionMask] = attentionMask
            #else
            fatalError()
            #endif
        }
        if isRequiringCausalMask {
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            // TODO: Infer scalar type from cache or model I/O descriptors
            let causalMask = MLTensor(zeros: [1, 1, 1, tokenCount + 1], scalarType: Float16.self)
            inputDictionary[Keys.causalMask] = causalMask
            #else
            fatalError()
            #endif
        }
        let outputs = try! await model.prediction(from: inputDictionary, using: state)

        assert(outputs.keys.contains(Keys.logits))
        let scores = outputs[Keys.logits]!
        assert(scores.rank == 3)
        let tokenIndex = inputIds.shape[1] - 1
        let nextTokenScores = scores[nil, tokenIndex, nil].expandingShape(at: 0)
        assert(nextTokenScores.rank == 3)
        assert(nextTokenScores.shape[0] == 1 && nextTokenScores.shape[1] == 1)
        return nextTokenScores
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
