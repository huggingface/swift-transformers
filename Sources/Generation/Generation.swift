//
//  Generation.swift
//
//
//  Created by Pedro Cuenca on 7/5/23.
//

#if canImport(CoreML)
import CoreML

import Tokenizers

/// Supported text generation modes.
public enum GenerationMode {
    /// Contrastive search generation mode
    case contrastiveSearch
    /// Greedy decoding generation mode
    case greedy
    /// Sampling-based generation mode
    case sample
    /// Beam search generation mode
    case beam
    /// Group beam search generation mode
    case groupBeam
    /// Unsupported generation mode
    case unsupported
}

/// Array of token IDs representing input tokens.
public typealias InputTokens = [Int]

/// Array of token IDs representing generated output tokens.
public typealias GenerationOutput = [Int]

/// A callable model that predicts the next token after a given sequence.
///
/// - Parameter tokens: Input token sequence
/// - Parameter config: Generation configuration
/// - Returns: Logits array for next token prediction
public typealias NextTokenModel = (InputTokens, GenerationConfig) -> any MLShapedArrayProtocol

/// Callback for receiving generated tokens during streaming.
public typealias PredictionTokensCallback = (GenerationOutput) -> Void

/// Callback for receiving generated text during streaming.
public typealias PredictionStringCallback = (String) -> Void

/// Protocol for text generation implementations.
public protocol Generation {
    /// Performs greedy search generation.
    ///
    /// - Parameters:
    ///   - config: Generation configuration
    ///   - tokens: Input token sequence
    ///   - model: Model for next token prediction
    ///   - callback: Optional callback for streaming tokens
    /// - Returns: Generated token sequence
    func greedySearch(config: GenerationConfig, tokens: InputTokens, model: NextTokenModel, callback: PredictionTokensCallback?) async -> GenerationOutput

    /// Generates text from a prompt string.
    ///
    /// - Parameters:
    ///   - config: Generation configuration
    ///   - prompt: Input prompt text
    ///   - model: Model for next token prediction
    ///   - tokenizer: Tokenizer for encoding/decoding
    ///   - callback: Optional callback for streaming text
    /// - Returns: Generated text string
    func generate(config: GenerationConfig, prompt: String, model: NextTokenModel, tokenizer: Tokenizer, callback: PredictionStringCallback?) async -> String
}

public extension Generation {
    func greedySearch(config: GenerationConfig, tokens: InputTokens, model: NextTokenModel, callback: PredictionTokensCallback? = nil) async -> GenerationOutput {
        // Iterate until we find the eos token or reach the max length
        // TODO: additional stopping criteria
        var outputTokens = tokens
        while outputTokens.count < config.maxLength {
            let logits = model(outputTokens, config)
            let (nextToken, _) = Math.argmax(logits)
            if nextToken == config.eosTokenId { break }
            outputTokens.append(nextToken)
            callback?(outputTokens)
        }
        return outputTokens
    }

    /// Performs sampling-based text generation with configurable logits warping.
    ///
    /// Uses various logits warpers (temperature, top-k, top-p, repetition penalty) to modify
    /// token probabilities before sampling, enabling diverse and controllable text generation.
    ///
    /// - Parameters:
    ///   - config: Generation configuration with sampling parameters
    ///   - tokens: Input token sequence
    ///   - model: Model for next token prediction
    ///   - callback: Optional callback for streaming tokens
    /// - Returns: Generated token sequence
    ///
    /// - Note: Based on https://github.com/huggingface/transformers/blob/42017d82baa083da2bee3055fdac80c81ee97b8a/src/transformers/generation/utils.py#L1552
    func sample(config: GenerationConfig, tokens: InputTokens, model: NextTokenModel, callback: PredictionTokensCallback? = nil) async -> GenerationOutput {
        // Iterate until we find the eos token or reach the max length
        // TODO: additional stopping criteria
        var outputTokens = tokens
        let logitsProcessor = LogitsProcessor(logitsWarpers: logitsWarpers(config: config))
        while outputTokens.count < config.maxLength {
            let outputs = model(outputTokens, config)
            // `floats` can be much faster than `scalars` for a vector with stride 1, as it uses `memcpy` in that case
            let logits = (outputs as? MLShapedArraySlice<Float>)?.floats ?? outputs.scalars as! [Float]
            let (indexes, processedLogits) = logitsProcessor(logits)
            let nextToken = Math.sample(indexes: indexes, probs: Math.softmax(processedLogits))
            if nextToken == config.eosTokenId { break }
            outputTokens.append(nextToken)
            callback?(outputTokens)
        }
        return outputTokens
    }

    func generate(config: GenerationConfig, prompt: String, model: NextTokenModel, tokenizer: Tokenizer, callback: PredictionStringCallback? = nil) async -> String {
        let tokens = tokenizer.encode(text: prompt)
        var generationConfig = config
        generationConfig.maxLength = config.maxNewTokens + tokens.count

        let output: GenerationOutput
        switch generationConfig.generationMode {
        case .greedy:
            output = await greedySearch(config: generationConfig, tokens: tokens, model: model) { tokens in
                callback?(tokenizer.decode(tokens: tokens))
            }
        case .sample:
            output = await sample(config: generationConfig, tokens: tokens, model: model) { tokens in
                callback?(tokenizer.decode(tokens: tokens))
            }
        default:
            fatalError("Generation mode \(generationConfig.generationMode) not implemented yet")
        }

        return tokenizer.decode(tokens: output)
    }

    private func logitsWarpers(config: GenerationConfig) -> [any LogitsWarper] {
        var logitsWarpers = [any LogitsWarper]()
        if config.temperature > 0, config.temperature != 1 {
            logitsWarpers.append(TemperatureLogitsWarper(temperature: Float(config.temperature)))
        }
        if config.topK > 0 {
            logitsWarpers.append(TopKLogitsWarper(k: config.topK))
        }
        if config.topP < 1.0 {
            logitsWarpers.append(TopPLogitsWarper(p: Float(config.topP)))
        }
        if config.repetitionPenalty != 1.0 {
            logitsWarpers.append(RepetitionPenaltyWarper(penalty: config.repetitionPenalty))
        }
        return logitsWarpers
    }
}
#endif // canImport(CoreML)
