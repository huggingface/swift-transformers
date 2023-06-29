//
//  LanguageModel.swift
//  
//
//  Created by Pedro Cuenca on 7/5/23.
//

import CoreML
import Tokenizers
import Generation
import Hub

public class LanguageModel {
    public let model: MLModel
    
    public let minContextLength: Int
    public let maxContextLength: Int
    
    let input_ids = "input_ids"
    let attention_mask = "attention_mask"
    
    struct Configurations {
        var modelConfig: Config
        var tokenizerConfig: Config?
        var tokenizerData: Config
    }
    
    private var configPromise: Task<Configurations, Error>? = nil
    private var _tokenizer: Tokenizer? = nil

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
        
        self.configPromise = Task.init {
            return try await self.loadConfig()
        }
    }
}

public extension LanguageModel {
    static func loadCompiled(url: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws -> LanguageModel {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let model = try MLModel(contentsOf: url, configuration: config)
        return LanguageModel(model: model)
    }
}

public extension LanguageModel {
    var description: String {
        if let description = model.modelDescription.metadata[MLModelMetadataKey.description] as? String,
           !description.isEmpty {
            return description
        }
        return model.configuration.modelDisplayName ?? ""
    }
    
    /// `name_or_path` in the Python world
    var modelName: String {
        if let userFields = model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String : String],
           let name = userFields["co.huggingface.exporters.name"] {
            return name
        }
        // This is usually the basename of the file, that's our best bet if no metadata exists
        guard let modelName = model.configuration.modelDisplayName else { fatalError("Models must have a name that identifies them") }
        return modelName
    }
        
    var inputIdsDescription: MLFeatureDescription {
        model.modelDescription.inputDescriptionsByName[input_ids]!
    }
    
    var inputIdsName: String {
        inputIdsDescription.name
    }
    
    /// The expected shape of the models latent sample input
    var inputIdsShape: [Int] {
        inputIdsDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
    
    var requiresAttention: Bool {
        model.modelDescription.inputDescriptionsByName[attention_mask] != nil
    }
    
    // MLShapedArrayProtocol is either a MLShapedArray or a MLShapedArraySlice
    func predictNextTokenScores(_ tokens: InputTokens, config: GenerationConfig) -> any MLShapedArrayProtocol {
        // TODO: exceptions
        
        // Maybe pad or truncate
        let maxTokens = min(tokens.count, maxContextLength)
        let padLength = maxTokens >= minContextLength ? 0 : minContextLength-maxTokens
        let inputTokens = Array(tokens[0..<maxTokens]) + Array(repeating: config.padTokenId ?? 0, count: padLength)
        
        let inputIds = MLMultiArray.from(inputTokens, dims: inputIdsShape.count)
        var inputDictionary = [inputIdsName: inputIds]
        if requiresAttention {
            let mask = Array(repeating: 1, count: maxTokens) + Array(repeating: 0, count: padLength)
            inputDictionary[attention_mask] = MLMultiArray.from(mask, dims: inputIdsShape.count)
        }
        let input = try! MLDictionaryFeatureProvider(dictionary: inputDictionary)
        
        let output = try! model.prediction(from: input)
        
        // TODO: maybe try to support models with "token_scores" too (default in exporters)
        assert(output.featureNames.first! == "logits")

        let scores = output.featureValue(for: output.featureNames.first!)!.shapedArrayValue(of: Float.self)!
        let nextTokenScores = scores[0, maxTokens - 1]
        return nextTokenScores
    }
}

extension LanguageModel {
    func loadConfig() async throws -> Configurations {
        // TODO: caching
        async let modelConfig = try Hub.downloadConfig(repoId: modelName, filename: "config.json")
        async let tokenizerConfig = try Hub.downloadConfig(repoId: modelName, filename: "tokenizer_config.json")
        async let tokenizerVocab = try Hub.downloadConfig(repoId: modelName, filename: "tokenizer.json")
        
        // Note tokenizerConfig may be nil (does not exist in all models)
        let configs = await Configurations(modelConfig: try modelConfig, tokenizerConfig: try? tokenizerConfig, tokenizerData: try tokenizerVocab)
        return configs
    }
}

/// async properties downloaded from the configuration
public extension LanguageModel {
    var modelConfig: Config {
        get async throws {
            try await configPromise!.value.modelConfig
        }
    }
    
    var tokenizerConfig: Config? {
        get async throws {
            try await configPromise!.value.tokenizerConfig
        }
    }
    
    var tokenizerData: Config {
        get async throws {
            try await configPromise!.value.tokenizerData
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
    
    var architecture: Architecture? {
        get async throws {
            guard let modelType = try await modelType else { return nil }
            return Architecture.from(modelType: modelType)
        }
    }
    
    var padTokenId: Int? {
        get async throws {
            guard let architecture = try await architecture else { return nil }
            return architecture.padTokenId ?? architecture.eosTokenId
        }
    }
    
    var bosTokenId: Int? {
        get async throws {
            let modelConfig = try await modelConfig
            if let bosTokenId = modelConfig.bosTokenId?.intValue { return bosTokenId }
            return try await architecture?.bosTokenId
        }
    }
    
    var eosTokenId: Int? {
        get async throws {
            let modelConfig = try await modelConfig
            if let eosTokenId = modelConfig.eosTokenId?.intValue { return eosTokenId }
            return try await architecture?.eosTokenId
        }
    }
    
    var tokenizer: Tokenizer {
        get async throws {
            guard _tokenizer == nil else { return _tokenizer! }
            guard let architecture = try await architecture else { throw "Cannot retrieve Tokenizer" }
            let tokenizerData = try await tokenizerData
            guard let vocab = tokenizerData.model?.vocab?.dictionary as? [String: Int] else { throw "Cannot find vocab in tokenizer JSON" }
            let merges = tokenizerData.model?.merges?.value as? [String]
            _tokenizer = architecture.tokenizerClass.init(vocab: vocab, merges: merges)
            return _tokenizer!
        }
    }
}

extension LanguageModel: TextGenerationModel {
    //TODO: retrieve from the json: https://huggingface.co/nlpcloud/instruct-gpt-j-fp16/blob/main/config.json#L26
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

extension String: Error {}
