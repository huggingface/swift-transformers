//
//  LanguageModel.swift
//
//
//  Created by Pedro Cuenca on 7/5/23.
//

import CoreML
import Generation
import Hub
import Tokenizers

@available(macOS 15.0, iOS 18.0, *)
public class LanguageModel {
    public let model: MLModel

    public let minContextLength: Int
    public let maxContextLength: Int

    private var configuration: LanguageModelConfigurationFromHub?
    private var _tokenizer: Tokenizer?

    public required init(model: MLModel) {
        self.model = model
        (minContextLength, maxContextLength) = Self.contextRange(from: model)
        configuration = LanguageModelConfigurationFromHub(modelName: modelName)
    }

    public func resetState() async { }

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
        static let presentKeys = "presentKeys"
        static let presentValues = "presentValues"
    }
}

@available(macOS 15.0, iOS 18.0, *)
public extension LanguageModel {
    static func loadCompiled(
        url: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) throws -> LanguageModel {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let model = try MLModel(contentsOf: url, configuration: config)
        return switch kvCacheAvailability(for: model) {
        case .statefulKVCache: LanguageModelWithStatefulKVCache(model: model)
        default: LanguageModel(model: model)
        }
    }
}

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
    var metadata: [MLModelMetadataKey: Any] {
        model.modelDescription.metadata
    }

    var modelDescription: MLModelDescription {
        model.modelDescription
    }

    var description: String {
        if let description = metadata[MLModelMetadataKey.description] as? String,
           !description.isEmpty
        {
            return description
        }
        return model.configuration.modelDisplayName ?? ""
    }

    /// `name_or_path` in the Python world
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

    var inputIdsDescription: MLFeatureDescription {
        modelDescription.inputDescriptionsByName[Keys.inputIds]!
    }

    var inputIdsName: String {
        inputIdsDescription.name
    }

    /// The expected shape of the models latent sample input
    var inputIdsShape: [Int] {
        inputIdsDescription.multiArrayConstraint!.shape.map(\.intValue)
    }

    var isRequiringAttentionMask: Bool {
        modelDescription.inputDescriptionsByName[Keys.attentionMask] != nil
    }

    var isRequiringCausalMask: Bool {
        modelDescription.inputDescriptionsByName[Keys.causalMask] != nil
    }

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
            fatalError("""
                KV Cache using IO is currently not supported, please file a feature request on \
                https://github.com/huggingface/swift-transformers
                """)
        } else {
            fatalError("""
                KV Cache using IO is currently not supported, please file a feature request on \
                https://github.com/huggingface/swift-transformers
                """)
        }
    }
}

/// async properties downloaded from the configuration
@available(macOS 15.0, iOS 18.0, *)
public extension LanguageModel {
    var modelConfig: Config {
        get async throws {
            try await configuration!.modelConfig
        }
    }

    var tokenizerConfig: Config? {
        get async throws {
            try await configuration!.tokenizerConfig
        }
    }

    var tokenizerData: Config {
        get async throws {
            try await configuration!.tokenizerData
        }
    }

    var modelType: String? {
        get async throws {
            try await modelConfig.modelType?.stringValue
        }
    }

    var textGenerationParameters: Config? {
        get async throws {
            try await modelConfig.taskSpecificParams?.textGeneration
        }
    }

    var defaultDoSample: Bool {
        get async throws {
            try await textGenerationParameters?.doSample?.boolValue ?? true
        }
    }

    var bosTokenId: Int? {
        get async throws {
            let modelConfig = try await modelConfig
            return modelConfig.bosTokenId?.intValue
        }
    }

    var eosTokenId: Int? {
        get async throws {
            let modelConfig = try await modelConfig
            return modelConfig.eosTokenId?.intValue
        }
    }

    var tokenizer: Tokenizer {
        get async throws {
            if let _tokenizer {
                return _tokenizer
            }
            guard let tokenizerConfig = try await tokenizerConfig else {
                throw "Cannot retrieve Tokenizer configuration"
            }
            let tokenizerData = try await tokenizerData
            _tokenizer = try AutoTokenizer.from(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData
            )
            return _tokenizer!
        }
    }
}

@available(macOS 15.0, iOS 18.0, *)
extension LanguageModel: TextGenerationModel {
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
            fatalError("""
                Encountered uninitialized `state`. Ensure `resetState` is called prior to calling \
                `predictNextTokenScores`. 
                """)
        }
        let inputIds = switch mode {
        case .prefilling: tokens // Pass in all takens if pre-filling prompt
        case .extending: tokens[nil, -1].expandingShape(at: 0) // otherwise just the last token
        }
        mode = .extending

        var inputDictionary = [
            Keys.inputIds: inputIds,
        ]
        if isRequiringAttentionMask {
            // TODO: Infer scalar type from cache or model I/O descriptors
            let attentionMask = MLTensor(zeros: [1, 1, 1, tokenCount + 1], scalarType: Float16.self)
            inputDictionary[Keys.attentionMask] = attentionMask
        }
        if isRequiringCausalMask {
            // TODO: Infer scalar type from cache or model I/O descriptors
            let causalMask = MLTensor(zeros: [1, 1, 1, tokenCount + 1], scalarType: Float16.self)
            inputDictionary[Keys.causalMask] = causalMask
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

extension String: @retroactive Error {}
