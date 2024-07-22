//
//  LanguageModelTypes.swift
//
//
//  Created by Pedro Cuenca on 8/5/23.
//

import CoreML
import Generation
import Tokenizers


/// A causal language model.
@available(macOS 15.0, iOS 18.0, *)
public protocol LanguageModelProtocol {
    /// `name_or_path` in the Python world
    var modelName: String { get }

    var tokenizer: Tokenizer { get async throws }
    var model: MLModel { get }

    /// Resets the state of the language model.
    ///
    /// Call `resetState()` for each new sequence generated.
    func resetState() async

    init(model: MLModel)

    /// Returns the next token conditioned on the given input.
    /// - Parameters:
    ///   - input: The input sequence to condition the language model.
    ///   - config: The generation configuration.
    /// - Returns: The raw scores of the next token.
    func predictNextTokenScores(_ input: MLTensor, config: GenerationConfig) async -> MLTensor
}

@available(macOS 15.0, iOS 18.0, *)
public extension LanguageModelProtocol {
    func callAsFunction(_ input: MLTensor, config: GenerationConfig) async -> MLTensor {
        await predictNextTokenScores(input, config: config)
    }
}

@available(macOS 15.0, iOS 18.0, *)
public protocol TextGenerationModel: Generation, LanguageModelProtocol {
    var defaultGenerationConfig: GenerationConfig { get }

    func generate(
        config: GenerationConfig,
        prompt: String,
        callback: PredictionStringCallback?
    ) async throws -> String
}

@available(macOS 15.0, iOS 18.0, *)
public extension TextGenerationModel {
    @discardableResult
    func generate(
        config: GenerationConfig,
        prompt: String,
        callback: PredictionStringCallback? = nil
    ) async throws -> String {
        // Prepare the language model for a new sequence.
        await resetState()

        // Run inference.
        return try await generate(
            config: config,
            prompt: prompt,
            model: callAsFunction,
            tokenizer: tokenizer,
            callback: callback
        )
    }
}
