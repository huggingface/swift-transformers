#if canImport(CoreML)
import CoreML

/// LogitsProcessor that performs top-k filtering, restricting to the k highest probability elements.
///
/// Filters out all tokens except the k most likely ones by setting their logits to -inf.
/// This reduces the risk of low-probability tokens being sampled.
///
/// Often used together with `TemperatureLogitsWarper` and `TopPLogitsWarper`.
/// Pro tip: In practice, LLMs use top_k in the 5-50 range.
///
/// Based on: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L532
@available(macOS 15.0, iOS 18.0, *)
public struct TopKLogitsWarper: LogitsProcessor {
    public let topK: Int
    public let filterValue: Float
    public let minTokensToKeep: Int

    /// Creates a top-k logits warper.
    ///
    /// - Parameters:
    ///   - topK: Number of highest probability tokens to keep
    ///   - filterValue: Value to set filtered tokens to (default: -infinity)
    ///   - minTokensToKeep: Minimum tokens that cannot be filtered (default: 1)
    public init(topK: Int, filterValue: Float = -.infinity, minTokensToKeep: Int = 1) {
        precondition(topK > 0, "topK must be strictly positive, got \(topK)")
        precondition(minTokensToKeep >= 1, "minTokensToKeep must be at least 1, got \(minTokensToKeep)")

        self.topK = max(topK, minTokensToKeep)
        self.filterValue = filterValue
        self.minTokensToKeep = minTokensToKeep
    }

    public func callAsFunction(_ inputIds: MLTensor, _ scores: MLTensor) async -> MLTensor {
        let vocabSize = scores.shape[scores.rank - 1]
        let k = min(topK, vocabSize)  // Safety check

        // Get the k-th highest score (the threshold)
        let (topKValues, _) = scores.topK(k)

        // The threshold is the smallest value in the top-k (last element)
        // We need to get the value at index [k-1] along the last dimension
        let thresholdScores = await topKValues.shapedArray(of: Float.self)

        // For each batch item, get the k-th largest score
        let batchSize = scores.shape[0]
        var thresholds = [Float]()

        for batchIdx in 0..<batchSize {
            let batchStart = batchIdx * k
            let threshold = thresholdScores.scalars[batchStart + k - 1]
            thresholds.append(threshold)
        }

        // Create mask: filter out all scores below the threshold
        let scoresArray = await scores.shapedArray(of: Float.self)
        var maskedScores = scoresArray.scalars

        for batchIdx in 0..<batchSize {
            let threshold = thresholds[batchIdx]
            let vocabStart = batchIdx * vocabSize

            for tokenIdx in 0..<vocabSize {
                let scoreIdx = vocabStart + tokenIdx
                if maskedScores[scoreIdx] < threshold {
                    maskedScores[scoreIdx] = filterValue
                }
            }
        }

        return MLTensor(shape: scores.shape, scalars: maskedScores, scalarType: Float.self)
    }
}
#endif