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
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
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
    /// - Throws: If topP is not in [0, 1] or if minTokensToKeep is less than 1
    public init(topP: Float, filterValue: Float = -.infinity, minTokensToKeep: Int = 1) throws {
        guard topP >= 0 && topP <= 1.0 else {
            throw LogitsProcessorError.invalidParameter("topP must be in [0, 1], got \(topP)")
        }
        guard minTokensToKeep >= 1 else {
            throw LogitsProcessorError.invalidParameter("minTokensToKeep must be at least 1, got \(minTokensToKeep)")
        }

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
        // 6. Scatter mask back to original indexing using inverse permutation

        let vocabSize = scores.shape[scores.rank - 1]

        // Scores are already in Float32 (handled by LogitsProcessorList)
        // Sort in descending order (highest scores first)
        let sortedIndices = scores.argsort(alongAxis: -1, descendingOrder: true)

        // Build inverse permutation for scattering back
        let inversePermutation = sortedIndices.argsort(alongAxis: -1)

        // Gather scores in sorted order
        let sortedScores = scores.gathering(atIndices: sortedIndices, alongAxis: -1)

        // Compute probabilities and cumulative sum in sorted order
        let sortedProbs = sortedScores.softmax(alongAxis: -1)
        let cumulativeProbs = sortedProbs.cumulativeSum(alongAxis: -1)

        // Shift cumsum to exclude current token (HuggingFace convention)
        // This ensures we include the first token that pushes us over the threshold
        let cumulativeProbsShifted = cumulativeProbs - sortedProbs

        // Create position tensor [0, 1, 2, ..., vocabSize-1] for minTokensToKeep check
        let baseShape = Array(repeating: 1, count: sortedScores.rank - 1) + [vocabSize]
        var multiples = sortedScores.shape
        multiples[multiples.count - 1] = 1

        let positions = MLTensor(
            rangeFrom: Int32(0),
            to: Int32(vocabSize),
            by: 1,
            scalarType: Int32.self
        )
        .reshaped(to: baseShape)
        .tiled(multiples: multiples)
        .cast(to: Float.self)

        // Create mask in sorted order:
        // Remove if: position >= minTokensToKeep AND cumsum_shifted > topP
        let beyondMinimum = positions .>= Float(minTokensToKeep)
        let exceedsThreshold = cumulativeProbsShifted .> topP
        let removeMaskSorted = beyondMinimum .& exceedsThreshold

        // Apply filter value in sorted space
        let filterTensor = MLTensor(
            repeating: filterValue,
            shape: sortedScores.shape,
            scalarType: Float.self
        )
        let filteredSorted = sortedScores.replacing(with: filterTensor, where: removeMaskSorted)

        // Scatter back to original indexing using inverse permutation
        return filteredSorted.gathering(atIndices: inversePermutation, alongAxis: -1)
    }
}
#endif
