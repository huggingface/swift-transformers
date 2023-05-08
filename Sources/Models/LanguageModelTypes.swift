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
    var tokenizer: Tokenizer { get }
    var model: MLModel { get }
    
    init(model: MLModel)
    
    /// Make prediction callable (this works like __call__ in Python)
    func predictNextToken(_ tokens: InputTokens) -> Int
    func callAsFunction(_ tokens: InputTokens) -> Int
}

public extension LanguageModelProtocol {
    func callAsFunction(_ tokens: InputTokens) -> Int {
        predictNextToken(tokens)
    }
}

public protocol TextGenerationModel: Generation, LanguageModelProtocol {
    func generate(config: GenerationConfig, prompt: String, callback: PredictionStringCallback?) async -> String
}

public extension TextGenerationModel {
    func generate(config: GenerationConfig, prompt: String, callback: PredictionStringCallback? = nil) async -> String {
        await self.generate(config: config, prompt: prompt, model: self.callAsFunction(_:), tokenizer: self.tokenizer, callback: callback)
    }
}
