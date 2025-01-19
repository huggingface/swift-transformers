//
//  LanguageModelTypes.swift
//
//
//  Created by Pedro Cuenca on 8/5/23.
//

import CoreML
import Generation
import Tokenizers

public protocol LanguageModelProtocol {
    /// `name_or_path` in the Python world
    var modelName: String { get }

    var tokenizer: Tokenizer { get async throws }
    var model: MLModel { get }

    init(model: MLModel)

    /// Make prediction callable (this works like __call__ in Python)
    func predictNextTokenScores(_ tokens: InputTokens, config: GenerationConfig) -> any MLShapedArrayProtocol
    func callAsFunction(_ tokens: InputTokens, config: GenerationConfig) -> any MLShapedArrayProtocol
}

extension LanguageModelProtocol {
    public func callAsFunction(_ tokens: InputTokens, config: GenerationConfig) -> any MLShapedArrayProtocol {
        predictNextTokenScores(tokens, config: config)
    }
}

public protocol TextGenerationModel: Generation, LanguageModelProtocol {
    var defaultGenerationConfig: GenerationConfig { get }
    func generate(config: GenerationConfig, prompt: String, callback: PredictionStringCallback?) async throws -> String
}

extension TextGenerationModel {
    @discardableResult
    public func generate(config: GenerationConfig, prompt: String, callback: PredictionStringCallback? = nil)
        async throws -> String
    {
        try await self.generate(
            config: config, prompt: prompt, model: self.callAsFunction, tokenizer: self.tokenizer, callback: callback)
    }
}
