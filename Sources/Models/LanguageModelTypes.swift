//
//  LanguageModelTypes.swift
//
//
//  Created by Pedro Cuenca on 8/5/23.
//

#if canImport(CoreML)
import CoreML

import Generation
import Tokenizers

/// Protocol defining the core interface for language model implementations.
///
/// This protocol establishes the fundamental requirements for any language model
/// that can perform next-token prediction and text generation tasks.
@available(macOS 15.0, iOS 18.0, *)
public protocol LanguageModelProtocol {
    /// The name or path of the model.
    ///
    /// This corresponds to the `name_or_path` field in Hugging Face transformers.
    var modelName: String { get }

    /// The tokenizer associated with this model.
    ///
    /// - Returns: A configured tokenizer instance
    /// - Throws: An error if the tokenizer cannot be loaded or configured
    var tokenizer: Tokenizer { get async throws }

    /// The underlying CoreML model used for inference.
    var model: MLModel { get }

    /// Resets the state of the language model.
    ///
    /// Call `resetState()` for each new sequence generated.
    func resetState() async

    /// Creates a new language model instance from a CoreML model.
    ///
    /// - Parameter model: The CoreML model to wrap
    init(model: MLModel)

    /// Predicts the next token scores for the given input tokens.
    ///
    /// - Parameters:
    ///   - input: The input sequence tensor.
    ///   - config: The generation configuration containing model parameters.
    /// - Returns: MLTensor with the raw scores of the next token.
    func predictNextTokenScores(_ input: MLTensor, config: GenerationConfig) async -> MLTensor
}

@available(macOS 15.0, iOS 18.0, *)
public extension LanguageModelProtocol {
    /// Function call syntax for next token prediction.
    ///
    /// This provides a more convenient syntax for calling `predictNextTokenScores`.
    ///
    /// - Parameters:
    ///   - tokens: The input token sequence
    ///   - config: The generation configuration containing model parameters
    /// - Returns: A shaped array containing the logits for the next token prediction
    func callAsFunction(_ input: MLTensor, config: GenerationConfig) async -> MLTensor {
        await predictNextTokenScores(input, config: config)
    }
}

/// Protocol for language models that support text generation capabilities.
///
/// This protocol extends `LanguageModelProtocol` and `Generation` to provide
/// high-level text generation functionality with configurable parameters.
@available(macOS 15.0, iOS 18.0, *)
public protocol TextGenerationModel: Generation, LanguageModelProtocol {
    /// The default generation configuration for this model.
    ///
    /// Provides model-specific defaults for generation parameters such as
    /// sampling behavior, temperature, and token limits.
    var defaultGenerationConfig: GenerationConfig { get }

    /// Generates text based on the given prompt and configuration.
    ///
    /// - Parameters:
    ///   - config: The generation configuration specifying parameters like max tokens and sampling
    ///   - prompt: The input text to use as the generation starting point
    ///   - callback: Optional callback to receive intermediate generation results
    /// - Returns: The generated text as a string
    /// - Throws: An error if text generation fails
    func generate(
        config: GenerationConfig,
        prompt: String,
        callback: PredictionStringCallback?
    ) async throws -> String
}

@available(macOS 15.0, iOS 18.0, *)
public extension TextGenerationModel {
    /// Default implementation of text generation that uses the underlying generation framework.
    ///
    /// - Parameters:
    ///   - config: The generation configuration specifying parameters like max tokens and sampling
    ///   - prompt: The input text to use as the generation starting point
    ///   - callback: Optional callback to receive intermediate generation results
    /// - Returns: The generated text as a string
    /// - Throws: An error if text generation fails
    @discardableResult
    func generate(config: GenerationConfig, prompt: String, callback: PredictionStringCallback? = nil) async throws -> String {
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
#endif // canImport(CoreML)
