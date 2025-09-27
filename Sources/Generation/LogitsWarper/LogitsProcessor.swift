#if canImport(CoreML)
import CoreML

/// Abstract base class for all logits processors that can be applied during generation.
///
/// Logits processors modify the probability distribution over vocabulary tokens by transforming
/// the raw logit scores produced by language models. This enables various sampling strategies
/// such as temperature scaling, top-k/top-p filtering, and repetition penalties.
///
/// Based on: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
@available(macOS 15.0, iOS 18.0, *)
public protocol LogitsProcessor {
    /// Processes logits for next token prediction.
    ///
    /// - Parameters:
    ///   - inputIds: Tensor of input token IDs with shape `[batch_size, sequence_length]`
    ///   - scores: Tensor of raw logit scores with shape `[batch_size, vocab_size]`
    /// - Returns: Processed logits tensor with shape `[batch_size, vocab_size]`
    ///
    /// - Note: The `inputIds` parameter provides context for processors that need to examine
    ///   the generated sequence (e.g., repetition penalty). Processors that don't need this
    ///   context (e.g., temperature) can ignore it.
    func callAsFunction(_ inputIds: MLTensor, _ scores: MLTensor) async -> MLTensor
}

/// A list of logits processors that applies each processor sequentially.
///
/// This class provides a convenient way to chain multiple logits processors together.
/// Each processor is applied in order to the logits tensor, with the output of one
/// processor becoming the input to the next.
@available(macOS 15.0, iOS 18.0, *)
public struct LogitsProcessorList {
    public var processors: [any LogitsProcessor]

    public init(processors: [any LogitsProcessor]) {
        self.processors = processors
    }

    /// Applies all logits processors sequentially to the input scores.
    ///
    /// - Parameters:
    ///   - inputIds: Tensor of input token IDs with shape `[batch_size, sequence_length]`
    ///   - scores: Tensor of raw logit scores with shape `[batch_size, vocab_size]`
    /// - Returns: Processed logits tensor with shape `[batch_size, vocab_size]`
    public func callAsFunction(_ inputIds: MLTensor, _ scores: MLTensor) async -> MLTensor {
        var processedScores = scores
        for processor in processors {
            processedScores = await processor(inputIds, processedScores)
        }
        return processedScores
    }
}
#endif