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
    public let contextLength: Int
    
    let input_ids = "input_ids"
    let attention_mask = "attention_mask"

    public required init(model: MLModel) {
        self.model = model
        
        // We assume inputs named "input_ids" with shape (1, seq_length)
        // Perhaps we should convert to vectors of shape (seq_length) and use sequenceConstraint instead of shapeConstraint
        let inputDescription = model.modelDescription.inputDescriptionsByName["input_ids"]
        let range = inputDescription?.multiArrayConstraint?.shapeConstraint.sizeRangeForDimension[1] as? NSRange
        self.contextLength = range?.length ?? 128
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
    
    var tokenizer: Tokenizer {
        guard let architecture = Architecture.from(modelName: modelName) else { fatalError("Cannot obtain model architecture") }
        return architecture.tokenizerClass.init()
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
    
    func predictNextToken(_ tokens: InputTokens) -> Int {
        // TODO: exceptions
        let maxTokens = tokens.count
        let inputIds = MLMultiArray.from(tokens, dims: inputIdsShape.count)
        var inputDictionary = [inputIdsName: inputIds]
        if requiresAttention {
            inputDictionary[attention_mask] = MLMultiArray.from(Array(repeating: 1, count: maxTokens), dims: inputIdsShape.count)
        }
        let input = try! MLDictionaryFeatureProvider(dictionary: inputDictionary)
        
        let output = try! model.prediction(from: input)
        
        // TODO: maybe do something different if the output is "logits"
        assert(output.featureNames.first! == "token_scores")
        
        let scores = output.featureValue(for: output.featureNames.first!)!.multiArrayValue!
        let nextTokenScores = MLMultiArray.slice(
            scores,
            indexing: [.select(0), .select(maxTokens - 1), .slice]
        )
        
        // Argmax
        let (nextToken, _) = Math.argmax(nextTokenScores)
        return nextToken
    }
}

extension LanguageModel: TextGenerationModel {}
