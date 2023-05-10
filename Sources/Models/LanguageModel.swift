//
//  LanguageModel.swift
//  
//
//  Created by Pedro Cuenca on 7/5/23.
//

import CoreML
import Tokenizers
import Generation

public class LanguageModel {
    public let model: MLModel
    
    public let minContextLength: Int
    public let maxContextLength: Int
    
    let input_ids = "input_ids"
    let attention_mask = "attention_mask"

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
    
    var architecture: Architecture {
        guard let architecture = Architecture.from(modelName: modelName) else { fatalError("Cannot obtain model architecture") }
        return architecture
    }
    
    var tokenizer: Tokenizer {
        return architecture.tokenizerClass.init()
    }
    
    var padTokenId: Int? { architecture.padTokenId ?? architecture.eosTokenId }
    var bosTokenId: Int? { architecture.bosTokenId }
    var eosTokenId: Int? { architecture.eosTokenId }
    
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
    
    func predictNextTokenScores(_ tokens: InputTokens) -> MLMultiArray {
        // TODO: exceptions
        
        // Maybe pad or truncate
        let maxTokens = min(tokens.count, maxContextLength)
        let padLength = maxTokens >= minContextLength ? 0 : minContextLength-maxTokens
        let inputTokens = Array(tokens[0..<maxTokens]) + Array(repeating: padTokenId ?? 0, count: padLength)
        
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

        let scores = output.featureValue(for: output.featureNames.first!)!.multiArrayValue!
        let nextTokenScores = MLMultiArray.slice(
            scores,
            indexing: [.select(0), .select(maxTokens - 1), .slice]
        )
        
        return nextTokenScores
    }
}

extension LanguageModel: TextGenerationModel {
    //TODO: retrieve from the json: https://huggingface.co/nlpcloud/instruct-gpt-j-fp16/blob/main/config.json#L26
    public var defaultGenerationConfig: GenerationConfig {
        var config = GenerationConfig(maxNewTokens: 30)
        switch modelName.lowercased() {
        case let x where x.contains("gpt"):
            config.doSample = true
            config.topK = 10
        default: break
        }
        return config
    }
}
