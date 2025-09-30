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

        // Create logits processor list based on config
        let logitsProcessorList = createLogitsProcessorList(config: config)

        let inputLength = outputTokens.shape[1]
        let maxTotalLength = min(config.maxLength, inputLength + config.maxNewTokens)

        while outputTokens.shape[1] < maxTotalLength {
            // Get raw logits from model
            let nextTokenScores = await model(outputTokens, config)

            // Apply logits processors
            let processedScores = await logitsProcessorList(outputTokens, nextTokenScores)

            // Select next token based on generation mode
            let nextToken =
                switch config.generationMode {
                case .greedy:
                    selectNextTokenUsingGreedyDecoding(from: processedScores)
                case .sample:
                    selectNextTokenUsingSampling(from: processedScores)
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

    /// Creates a list of logits processors based on generation configuration.
    ///
    /// - Parameter config: Generation configuration specifying which processors to apply
    /// - Returns: List of logits processors to apply during generation
    private func createLogitsProcessorList(config: GenerationConfig) -> LogitsProcessorList {
        var processors: [any LogitsProcessor] = []

        // Repetition penalty (applied before sampling warpers)
        if config.repetitionPenalty != 1.0 {
            if let processor = try? RepetitionPenaltyLogitsProcessor(penalty: Float(config.repetitionPenalty)) {
                processors.append(processor)
            }
        }

        // Temperature scaling (if not default)
        if config.temperature > 0 && config.temperature != 1.0 {
            if let processor = try? TemperatureLogitsWarper(temperature: config.temperature) {
                processors.append(processor)
            }
        }

        // Top-K filtering (only apply if topK is meaningful)
        // Note: We can't determine vocab size here, so TopKLogitsWarper handles the case
        // where topK >= vocabSize internally
        if config.topK > 0 && config.topK < Int.max {
            if let processor = try? TopKLogitsWarper(topK: config.topK) {
                processors.append(processor)
            }
        }

        // Top-P (nucleus) sampling
        if config.topP < 1.0 {
            if let processor = try? TopPLogitsWarper(topP: Float(config.topP)) {
                processors.append(processor)
            }
        }

        return LogitsProcessorList(processors: processors)
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
