//
//  Generation.swift
//
//
//  Created by Pedro Cuenca on 7/5/23.
//

#if canImport(CoreML)
import CoreML

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
@available(macOS 15.0, iOS 18.0, *)
public typealias NextTokenModel = (MLTensor, GenerationConfig) async -> MLTensor

/// Callback for receiving generated tokens during streaming.
public typealias PredictionTokensCallback = (GenerationOutput) -> Void

/// Callback for receiving generated text during streaming.
public typealias PredictionStringCallback = (String) -> Void

/// Protocol for text generation implementations.
@available(macOS 15.0, iOS 18.0, *)
public protocol Generation {
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

@available(macOS 15.0, iOS 18.0, *)
extension Generation {
    public func generate(
        config: GenerationConfig,
        tokens: InputTokens,
        model: NextTokenModel,
        callback: PredictionTokensCallback? = nil
    ) async -> GenerationOutput {
        let tokens = tokens.map { Int32($0) }
        var outputTokens = MLTensor(tokens).expandingShape(at: 0)
        while outputTokens.shape[1] < config.maxLength {
            let nextTokenScores = await model(outputTokens, config)
            let nextToken =
                switch config.generationMode {
                case .greedy:
                    selectNextTokenUsingGreedyDecoding(from: nextTokenScores)
                case .sample:
                    selectNextTokenUsingTopKSampling(
                        from: nextTokenScores,
                        temperature: config.temperature,
                        topK: config.topK
                    )
                default:
                    fatalError("Generation mode \(config.generationMode) not implemented yet")
                }

            if let nextTokenId = await tensorToGenerationOutput(nextToken).first, nextTokenId == config.eosTokenId {
                break
            }

            outputTokens = MLTensor(concatenating: [outputTokens, nextToken], alongAxis: -1)
            if let callback {
                let outputTokenIDs = await tensorToGenerationOutput(outputTokens)
                callback(outputTokenIDs)
            }
        }
        return await tensorToGenerationOutput(outputTokens)
    }

    private func tensorToGenerationOutput(_ tensor: MLTensor) async -> GenerationOutput {
        await tensor.shapedArray(of: Int32.self).scalars.map { Int($0) }
    }
}

@available(macOS 15.0, iOS 18.0, *)
public extension Generation {
    /// Performs greedy or sampling-based text generation based on generation configuration.
    ///
    /// - Parameters:
    ///   - config: Generation configuration with sampling parameters
    ///   - prompt: Input string
    ///   - model: Model for next token prediction
    ///   - tokenizer: Tokenizer to convert prompt to input tokens
    ///   - callback: Optional callback for streaming tokens
    /// - Returns: Generated token sequence
    ///
    /// - Note: Based on https://github.com/huggingface/transformers/blob/42017d82baa083da2bee3055fdac80c81ee97b8a/src/transformers/generation/utils.py#L1552
    func generate(
        config: GenerationConfig,
        prompt: String,
        model: NextTokenModel,
        tokenizer: Tokenizer,
        callback: PredictionStringCallback? = nil
    ) async -> String {
        let tokens = tokenizer.encode(text: prompt)
        var generationConfig = config
        generationConfig.maxLength = config.maxNewTokens + tokens.count
        generationConfig.eosTokenId = tokenizer.eosTokenId
        generationConfig.bosTokenId = tokenizer.bosTokenId
        let output = await generate(
            config: generationConfig,
            tokens: tokens,
            model: model
        ) { tokens in
            callback?(tokenizer.decode(tokens: tokens))
        }

        return tokenizer.decode(tokens: output)
    }
}
#endif // canImport(CoreML)
