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
        // 1. Sort logits in ascending order (transformers uses descending=False)
        // 2. Compute softmax probabilities on sorted logits
        // 3. Compute cumulative sum of probabilities
        // 4. Remove tokens with cumulative probability <= (1 - top_p)
        // 5. Keep at least min_tokens_to_keep
        // 6. Scatter mask back to original indexing

        let batchSize = scores.shape[0]
        let vocabSize = scores.shape[scores.rank - 1]

        // Convert to CPU for processing
        let scoresArray = await scores.shapedArray(of: Float.self)
        var maskedScores = scoresArray.scalars

        // Process each batch item
        for batchIdx in 0..<batchSize {
            let batchStart = batchIdx * vocabSize

            // Get this batch's scores
            let batchScores = Array(maskedScores[batchStart..<(batchStart + vocabSize)])

            // Create indices and sort by score (ascending)
            let indices = Array(0..<vocabSize)
            let sortedPairs = zip(batchScores, indices).sorted { $0.0 < $1.0 }
            let sortedScores = sortedPairs.map { $0.0 }
            let sortedIndices = sortedPairs.map { $0.1 }

            // Compute softmax probabilities on sorted scores
            let maxScore = sortedScores.max() ?? 0
            let expScores = sortedScores.map { exp($0 - maxScore) }
            let sumExp = expScores.reduce(0, +)
            let sortedProbs = expScores.map { $0 / sumExp }

            // Compute cumulative sum
            var cumulativeProbs = [Float]()
            var cumSum: Float = 0
            for prob in sortedProbs {
                cumSum += prob
                cumulativeProbs.append(cumSum)
            }

            // Find tokens to remove: cumsum <= (1 - top_p)
            // Keep tokens from the end (highest probability) where cumsum > (1 - top_p)
            let threshold = 1.0 - topP
            var indicesToRemove = Set<Int>()

            for i in 0..<vocabSize {
                // Keep at least minTokensToKeep tokens (from the end)
                if i < vocabSize - minTokensToKeep {
                    if cumulativeProbs[i] <= threshold {
                        indicesToRemove.insert(sortedIndices[i])
                    }
                }
            }

            // Apply mask
            for idx in indicesToRemove {
                maskedScores[batchStart + idx] = filterValue
            }
        }

        return MLTensor(shape: scores.shape, scalars: maskedScores, scalarType: Float.self)
    }
}
#endif