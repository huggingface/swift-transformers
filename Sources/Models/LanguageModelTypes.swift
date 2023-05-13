//
//  LanguageModelTypes.swift
//  
//
//  Created by Pedro Cuenca on 8/5/23.
//

import CoreML
import Tokenizers
import Generation

public protocol LanguageModelProtocol {
    /// `name_or_path` in the Python world
    var modelName: String { get }

    var tokenizer: Tokenizer { get }
    var model: MLModel { get }
    
    init(model: MLModel)
    
    /// Make prediction callable (this works like __call__ in Python)
    func predictNextTokenScores(_ tokens: InputTokens) -> any MLShapedArrayProtocol //MLShapedArray<Float>
    func callAsFunction(_ tokens: InputTokens) -> any MLShapedArrayProtocol //MLShapedArray<Float>
}

public extension LanguageModelProtocol {
    func callAsFunction(_ tokens: InputTokens) -> any MLShapedArrayProtocol {
        predictNextTokenScores(tokens)
    }
}

public protocol TextGenerationModel: Generation, LanguageModelProtocol {
    var defaultGenerationConfig: GenerationConfig { get }
    func generate(config: GenerationConfig, prompt: String, callback: PredictionStringCallback?) async -> String
}

public extension TextGenerationModel {
    func generate(config: GenerationConfig, prompt: String, callback: PredictionStringCallback? = nil) async -> String {
        await self.generate(config: config, prompt: prompt, model: self.callAsFunction(_:), tokenizer: self.tokenizer, callback: callback)
    }
}
