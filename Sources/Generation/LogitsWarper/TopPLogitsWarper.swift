#if canImport(CoreML)
import CoreML

/// LogitsProcessor that performs top-p (nucleus) sampling by restricting to the smallest
/// set of tokens whose cumulative probability exceeds a threshold.
///
/// Filters out low-probability tokens by computing cumulative probabilities after sorting
/// and masking tokens beyond the top_p threshold. This is more dynamic than top-k as the
/// number of kept tokens varies based on the probability distribution.
///
/// Often used together with `TemperatureLogitsWarper` and `TopKLogitsWarper`.
/// Pro tip: In practice, LLMs use top_p in the 0.9-0.95 range.
///
/// Based on:
/// - https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L465
/// - Paper: https://arxiv.org/abs/1904.09751 (Nucleus Sampling)
@available(macOS 15.0, iOS 18.0, *)
public struct TopPLogitsWarper: LogitsProcessor {
    public let topP: Float
    public let filterValue: Float
    public let minTokensToKeep: Int

    /// Creates a top-p logits warper.
    ///
    /// - Parameters:
    ///   - topP: Cumulative probability threshold. Must be between 0 and 1.
    ///   - filterValue: Value to set filtered tokens to (default: -infinity)
    ///   - minTokensToKeep: Minimum tokens that cannot be filtered (default: 1)
    public init(topP: Float, filterValue: Float = -.infinity, minTokensToKeep: Int = 1) {
        precondition(topP >= 0 && topP <= 1.0, "topP must be in [0, 1], got \(topP)")
        precondition(minTokensToKeep >= 1, "minTokensToKeep must be at least 1, got \(minTokensToKeep)")

        self.topP = topP
        self.filterValue = filterValue
        self.minTokensToKeep = minTokensToKeep
    }

    public func callAsFunction(_ inputIds: MLTensor, _ scores: MLTensor) async -> MLTensor {
        // Algorithm (following transformers implementation):
        // 1. Sort logits in descending order
        // 2. Compute softmax probabilities on sorted logits
        // 3. Compute cumulative sum of probabilities
        // 4. Remove tokens with cumulative probability > top_p
        // 5. Keep at least min_tokens_to_keep
        // 6. Scatter mask back to original indexing using argsort indices

        let vocabSize = scores.shape[scores.rank - 1]

        // Get sorted indices (descending order - highest scores first)
        let sortedIndices = scores.argsort(alongAxis: -1, descendingOrder: true)

        // Gather scores in sorted order
        let sortedScores = scores.gathering(atIndices: sortedIndices, alongAxis: -1)

        // Compute softmax on sorted scores
        let sortedProbs = sortedScores.softmax(alongAxis: -1)

        // Compute cumulative sum
        let cumulativeProbs = sortedProbs.cumulativeSum(alongAxis: -1)

        // Create mask: remove tokens where cumsum > topP
        // The HuggingFace implementation removes tokens where cumsum - current_prob > topP
        // This ensures we include the first token that pushes us over the threshold

        // Shift cumsum to get cumsum of previous tokens
        // For first token, this will be 0
        let cumulativeProbsShifted = cumulativeProbs - sortedProbs

        // Need CPU fallback for scatter and minTokensToKeep logic
        let cumulativeProbsArray = await cumulativeProbsShifted.shapedArray(of: Float.self)
        let indicesArray = await sortedIndices.shapedArray(of: Int32.self)
        let scoresArray = await scores.shapedArray(of: Float.self)

        var maskedScores = scoresArray.scalars
        let batchSize = scores.shape[0]

        for batchIdx in 0..<batchSize {
            let batchStart = batchIdx * vocabSize
            for i in 0..<vocabSize {
                let sortedIdx = batchStart + i
                let originalIdx = Int(indicesArray.scalars[sortedIdx])

                // Keep first minTokensToKeep tokens, otherwise check cumsum threshold
                let shouldRemove = (i >= minTokensToKeep) && (cumulativeProbsArray.scalars[sortedIdx] > topP)

                if shouldRemove {
                    maskedScores[batchStart + originalIdx] = filterValue
                }
            }
        }

        return MLTensor(shape: scores.shape, scalars: maskedScores, scalarType: Float.self)
    }
}
#endif
