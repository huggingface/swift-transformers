#if canImport(CoreML)
import CoreML

/// LogitsProcessor that performs min-p filtering on the logits.
///
/// Min-p keeps all tokens that are above a minimum probability, scaled by the probability
/// of the most likely token. As a result, the filter becomes more aggressive in the presence
/// of high-probability tokens, which is a sign of a confident output that we shouldn't deviate from.
///
/// Often used together with `TemperatureLogitsWarper`. Used as an alternative to `TopPLogitsWarper`
/// and `TopKLogitsWarper`.
///
/// Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in
/// the 0.99-0.8 range (use the opposite of normal `top_p` values).
///
/// Based on:
/// - https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L460
@available(macOS 15.0, iOS 18.0, *)
public struct MinPLogitsWarper: LogitsProcessor {
    public let minP: Float
    public let minTokensToKeep: Int
    public let filterValue: Float

    /// Creates a min-p logits warper.
    ///
    /// - Parameters:
    ///   - minP: Minimum token probability, which will be scaled by the probability of the most likely token.
    ///           Must be between 0 and 1. Typical values are 0.01-0.2.
    ///   - minTokensToKeep: Minimum number of tokens that cannot be filtered.
    ///   - filterValue: Value to set for filtered tokens (default: -infinity)
    /// - Throws: If parameters are invalid
    public init(minP: Float, minTokensToKeep: Int = 1, filterValue: Float = -Float.infinity) throws {
        guard minP >= 0 && minP <= 1.0 else {
            throw LogitsProcessorError.invalidParameter("minP must be in [0, 1], got \(minP)")
        }
        guard minTokensToKeep >= 1 else {
            throw LogitsProcessorError.invalidParameter("minTokensToKeep must be >= 1, got \(minTokensToKeep)")
        }
        self.minP = minP
        self.minTokensToKeep = minTokensToKeep
        self.filterValue = filterValue
    }

    public func callAsFunction(_ inputIds: MLTensor, _ scores: MLTensor) async -> MLTensor {
        // Algorithm (following transformers implementation):
        // 1. Compute probabilities from logits
        // 2. Find max probability per batch
        // 3. Create threshold = minP * maxProb
        // 4. Sort logits and mask tokens where prob < threshold
        // 5. Keep at least minTokensToKeep
        // 6. Scatter back to original order

        let vocabSize = scores.shape[scores.rank - 1]

        // Compute probabilities
        let probs = scores.softmax(alongAxis: -1)

        // Sort probabilities descending to get max (first element)
        let sortedProbIndices = probs.argsort(alongAxis: -1, descendingOrder: true)
        let sortedProbs = probs.gathering(atIndices: sortedProbIndices, alongAxis: -1)

        // Extract max prob per batch: first element of each sorted sequence
        // Do this on CPU to avoid complex broadcasting issues
        let sortedProbsArray = await sortedProbs.shapedArray(of: Float.self)
        let batchSize = scores.shape[0]
        var thresholdScalars = [Float]()
        thresholdScalars.reserveCapacity(batchSize * vocabSize)
        for batchIdx in 0..<batchSize {
            let maxProb = sortedProbsArray.scalars[batchIdx * vocabSize] // First element
            let thresholdVal = minP * maxProb
            for _ in 0..<vocabSize {
                thresholdScalars.append(thresholdVal)
            }
        }
        let threshold = MLTensor(shape: probs.shape, scalars: thresholdScalars, scalarType: Float.self)

        // Create mask: tokensToRemove where prob < threshold
        let tokensToRemove = probs .< threshold

        // Sort scores descending
        let sortedScoreIndices = scores.argsort(alongAxis: -1, descendingOrder: true)
        let inversePermutation = sortedScoreIndices.argsort(alongAxis: -1)

        // Gather mask in sorted order
        let sortedTokensToRemove = tokensToRemove.gathering(atIndices: sortedScoreIndices, alongAxis: -1)

        // Create position tensor for minTokensToKeep check
        let posBaseShape = Array(repeating: 1, count: scores.rank - 1) + [vocabSize]
        var posMultiples = scores.shape
        posMultiples[posMultiples.count - 1] = 1

        let positions = MLTensor(
            rangeFrom: Int32(0),
            to: Int32(vocabSize),
            by: 1,
            scalarType: Int32.self
        )
        .reshaped(to: posBaseShape)
        .tiled(multiples: posMultiples)

        // Mask: remove if (position >= minTokensToKeep AND shouldRemove)
        let beyondMinimum = positions .>= Int32(minTokensToKeep)
        let finalRemoveMask = sortedTokensToRemove .& beyondMinimum

        // Apply filter in sorted space
        let sortedScores = scores.gathering(atIndices: sortedScoreIndices, alongAxis: -1)
        let filterTensor = MLTensor(repeating: filterValue, shape: sortedScores.shape, scalarType: Float.self)
        let filteredSorted = sortedScores.replacing(with: filterTensor, where: finalRemoveMask)

        // Scatter back to original order
        return filteredSorted.gathering(atIndices: inversePermutation, alongAxis: -1)
    }
}
#endif
