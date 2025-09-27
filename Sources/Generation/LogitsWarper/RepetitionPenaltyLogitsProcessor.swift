#if canImport(CoreML)
import CoreML

/// LogitsProcessor that prevents repetition of previous tokens through a penalty.
///
/// For each token that has already appeared in the sequence:
/// - If the token's logit is negative: multiply by penalty (further suppresses it)
/// - If the token's logit is positive: divide by penalty (suppresses it)
///
/// This penalty is applied at most once per token, regardless of how many times it appears.
///
/// Typical penalty values:
/// - 1.0: No penalty
/// - > 1.0: Penalize repetition (e.g., 1.2 for balanced generation)
/// - 0.0 - 1.0: Encourage repetition
///
/// Based on:
/// - https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L297
/// - Paper: https://arxiv.org/abs/1909.05858
@available(macOS 15.0, iOS 18.0, *)
public struct RepetitionPenaltyLogitsProcessor: LogitsProcessor {
    public let penalty: Float

    /// Creates a repetition penalty logits processor.
    ///
    /// - Parameter penalty: Penalty factor. Values > 1.0 penalize repetition, values < 1.0 encourage it.
    public init(penalty: Float) {
        precondition(penalty > 0, "penalty must be strictly positive, got \(penalty)")
        self.penalty = penalty
    }

    public func callAsFunction(_ inputIds: MLTensor, _ scores: MLTensor) async -> MLTensor {
        guard penalty != 1.0 else { return scores }

        // Implementation approach (following transformers):
        // 1. Get unique token IDs from inputIds
        // 2. For each unique token, gather its logit value
        // 3. Apply conditional penalty: if logit < 0: *= penalty, else: /= penalty
        // 4. Scatter penalized values back to original positions

        // Convert to CPU for gather/scatter operations
        let scoresArray = await scores.shapedArray(of: Float.self)
        let inputIdsArray = await inputIds.shapedArray(of: Int32.self)

        // Process each batch item
        var scoresData = scoresArray.scalars
        let batchSize = scores.shape[0]
        let vocabSize = scores.shape[1]

        for batchIdx in 0..<batchSize {
            let seqStart = batchIdx * vocabSize

            // Get unique token IDs from this sequence
            let seqStartIds = batchIdx * inputIds.shape[1]
            let seqEndIds = seqStartIds + inputIds.shape[1]
            let tokenIds = Set(inputIdsArray.scalars[seqStartIds..<seqEndIds].map { Int($0) })

            // Apply penalty to each token that appeared in the sequence
            for tokenId in tokenIds {
                guard tokenId >= 0 && tokenId < vocabSize else { continue }

                let scoreIdx = seqStart + tokenId
                let score = scoresData[scoreIdx]

                // Apply penalty based on sign (following transformers implementation)
                scoresData[scoreIdx] = score < 0 ? score * penalty : score / penalty
            }
        }

        // Create new tensor with penalized scores
        return MLTensor(shape: scores.shape, scalars: scoresData, scalarType: Float.self)
    }
}
#endif