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

        // Optimized implementation following transformers:
        // 1. Gather scores for tokens that appear in input_ids
        // 2. Apply conditional penalty: if score < 0: *= penalty, else: /= penalty
        // 3. Scatter penalized values back to original positions

        // Gather scores for tokens that appear in input_ids
        let gatheredScores = scores.gathering(atIndices: inputIds, alongAxis: -1)

        // Apply conditional penalty based on sign (vectorized)
        let negativeScores = gatheredScores .< 0.0
        let penalizedScores = negativeScores.cast(to: Float.self) * (gatheredScores * penalty) +
                              (1.0 - negativeScores.cast(to: Float.self)) * (gatheredScores / penalty)

        // Scatter penalized values back to original positions
        // Note: MLTensor doesn't have direct scatter, so we use CPU operations for this step
        let vocabSize = scores.shape[scores.rank - 1]
        let batchSize = scores.shape[0]

        let inputIdsArray = await inputIds.shapedArray(of: Int32.self)
        let penalizedArray = await penalizedScores.shapedArray(of: Float.self)
        var scoresArray = await scores.shapedArray(of: Float.self)

        for batchIdx in 0..<batchSize {
            let seqStart = batchIdx * inputIds.shape[1]
            let seqEnd = seqStart + inputIds.shape[1]
            let batchOffset = batchIdx * scoresArray.shape.dropFirst().reduce(1, *)

            for (tokenIdx, inputIdxInSeq) in (seqStart..<seqEnd).enumerated() {
                let tokenId = Int(inputIdsArray.scalars[inputIdxInSeq])
                guard tokenId >= 0 && tokenId < vocabSize else { continue }

                // For rank-2: [batch_size, vocab_size]
                if scores.rank == 2 {
                    let scoreIdx = batchOffset + tokenId
                    let penalizedIdx = seqStart + tokenIdx
                    scoresArray.scalars[scoreIdx] = penalizedArray.scalars[penalizedIdx]
                }
                // For rank-3: [batch_size, seq_len, vocab_size] - update last position
                else if scores.rank == 3 {
                    let lastSeqPos = scores.shape[1] - 1
                    let scoreIdx = batchOffset + lastSeqPos * vocabSize + tokenId
                    let penalizedIdx = seqStart + tokenIdx
                    scoresArray.scalars[scoreIdx] = penalizedArray.scalars[penalizedIdx]
                }
            }
        }

        return MLTensor(shape: scores.shape, scalars: scoresArray.scalars, scalarType: Float.self)
    }
}
#endif
