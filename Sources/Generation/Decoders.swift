#if canImport(CoreML)
import CoreML

// MARK: Greedy Decoding

@available(macOS 15.0, iOS 18.0, *)
func selectNextTokenUsingGreedyDecoding(from scores: MLTensor) -> MLTensor {
    scores.argmax(alongAxis: -1).reshaped(to: [1, 1])
}

// MARK: Sampling

/// Performs multinomial sampling from processed logits.
///
/// Assumes logits have already been processed by LogitsProcessorList
/// (temperature, top-k, top-p, etc. already applied).
///
/// - Parameter scores: Processed logits tensor [batch_size, vocab_size]
/// - Returns: Sampled token ID tensor [batch_size, 1]
@available(macOS 15.0, iOS 18.0, *)
func selectNextTokenUsingSampling(from scores: MLTensor) -> MLTensor {
    // Convert logits to probabilities
    let probs = scores.softmax(alongAxis: -1)

    // Multinomial sampling using cumulative sum method
    // 1. Generate random number in [0, 1)
    // 2. Compute cumulative sum of probabilities
    // 3. Find first index where cumsum >= random_number

    let rnd = Float.random(in: 0..<1)
    var cumulativeProbs = probs.cumulativeSum(alongAxis: -1)

    // Mark all positions where cumsum >= rnd with a large value
    // Then argsort will give us the first such position
    cumulativeProbs = cumulativeProbs + (cumulativeProbs .< rnd) * 100.0

    let sampledIndex = cumulativeProbs.argsort(alongAxis: -1)[..., 0]
    return sampledIndex.reshaped(to: [1, 1])
}

// MARK: Legacy Top-K Sampling (deprecated, use LogitsProcessorList instead)

/// Legacy top-k sampling function that combines temperature, top-k, and sampling.
///
/// - Note: This function is deprecated. Use `selectNextTokenUsingSampling` with
///   `TemperatureLogitsWarper` and `TopKLogitsWarper` in a `LogitsProcessorList` instead.
@available(macOS 15.0, iOS 18.0, *)
func selectNextTokenUsingTopKSampling(from scores: MLTensor, temperature: Float, topK: Int) -> MLTensor {
    let temperatureAdjustedScores = scores / temperature
    let (topKScores, topKIndices) = temperatureAdjustedScores.topK(topK)
    let topKProbs = topKScores.softmax(alongAxis: -1)
    let rnd = topKProbs.sum() * Float.random(in: 0..<1)
    var accumTopKProbs = topKProbs.cumulativeSum(alongAxis: -1)
    accumTopKProbs += (accumTopKProbs .< rnd) * 100.0
    let topKIndex = accumTopKProbs.argsort()[..., 0]
    let nextTokenTensor = topKIndices.gathering(
        atIndices: topKIndex,
        alongAxis: topKIndices.rank - 1
    )
    return nextTokenTensor.reshaped(to: [1, 1])
}
#endif // canImport(CoreML)
