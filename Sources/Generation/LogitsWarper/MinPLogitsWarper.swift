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
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
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
        // 2. Find max probability per batch (with keepdim)
        // 3. Calculate threshold = minP * maxProb
        // 4. Create mask for tokens where prob < threshold
        // 5. Use topK to get min_tokens_to_keep and unmask them
        // 6. Apply mask to scores

        let vocabSize = scores.shape[scores.rank - 1]

        // Convert logits to probabilities
        let probs = scores.softmax(alongAxis: -1)

        // Get the probability of the top token for each sequence in the batch
        // Using max with keepRank=true to maintain dimensions for broadcasting
        let topProbs = probs.max(alongAxes: [-1], keepRank: true)

        // Calculate the actual min_p threshold by scaling min_p with the top token's probability
        let scaledMinP = topProbs * minP

        // Create a mask for tokens that have a probability less than the scaled min_p
        let tokensToRemove = probs .< scaledMinP

        // Keep at least min_tokens_to_keep tokens (clip k to vocab size if needed)
        let k = min(minTokensToKeep, vocabSize)

        // Get indices of top-k probabilities
        let topKResult = probs.topK(k)
        let topKIndices = topKResult.indices

        // Create a mask to keep the top-k tokens
        // Since MLTensor doesn't have a scatter operation that works like PyTorch's scatter_,
        // we use replacing(atIndices:with:alongAxis:) which replaces values at specified indices.
        // For our case, we want to unmask (set to False/0) the top-k token positions.

        // Convert boolean mask to Int32 (1 = remove, 0 = keep)
        let zerosInt = MLTensor(repeating: Int32(0), shape: tokensToRemove.shape, scalarType: Int32.self)
        let onesInt = MLTensor(repeating: Int32(1), shape: tokensToRemove.shape, scalarType: Int32.self)
        let tokensToRemoveAsInt = zerosInt.replacing(with: onesInt, where: tokensToRemove)

        // Try using replacing(atIndices:with:alongAxis:) which takes a scalar value
        // This replaces slices at the specified indices with the scalar value
        let finalTokensToRemoveInt = tokensToRemoveAsInt.replacing(atIndices: topKIndices, with: Int32(0), alongAxis: -1)

        // Convert back to boolean mask
        let zerosComparison = MLTensor(repeating: Int32(0), shape: tokensToRemove.shape, scalarType: Int32.self)
        let finalTokensToRemove = finalTokensToRemoveInt .!= zerosComparison

        // Apply mask to scores
        let filterTensor = MLTensor(repeating: filterValue, shape: scores.shape, scalarType: Float.self)
        return scores.replacing(with: filterTensor, where: finalTokensToRemove)
    }
}
#endif
