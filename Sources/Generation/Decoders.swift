#if canImport(CoreML)
import CoreML

// MARK: Greedy Decoding

@available(macOS 15.0, iOS 18.0, *)
func selectNextTokenUsingGreedyDecoding(from scores: MLTensor) -> MLTensor {
    let indices = scores.argmax(alongAxis: -1).reshaped(to: [1, 1])
    // Ensure indices are Int32 for concatenation with input tokens
    return indices.scalarType == Int32.self ? indices : indices.cast(to: Int32.self)
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

    // Multinomial sampling using cumulative sum method:
    // 1. Generate random number in [0, 1)
    // 2. Compute cumulative sum of probabilities
    // 3. Find first index where cumsum >= random_number
    //
    // This is equivalent to torch.multinomial() but using available MLTensor ops

    let batchSize = scores.shape[0]
    let rndTensor = MLTensor(randomUniform: [batchSize, 1], in: 0..<1, scalarType: Float.self)
    let cumulativeProbs = probs.cumulativeSum(alongAxis: -1)

    // Ensure random tensor matches the type of cumulativeProbs
    let rnd = cumulativeProbs.scalarType == Float.self ? rndTensor : rndTensor.cast(to: cumulativeProbs.scalarType)

    // Create mask where cumsum >= rnd (these are candidates)
    // We want the FIRST position where this is true
    // Strategy: Set all positions where cumsum < rnd to a large value (1000.0)
    // Set all positions where cumsum >= rnd to their index value
    // Then argmin will give us the first qualifying index

    let mask = cumulativeProbs .< rnd
    let penalized = mask * 1000.0 // Large value for positions to skip
    let indexed = penalized + cumulativeProbs // Positions >= rnd will have small values

    let sampledIndex = indexed.argmin(alongAxis: -1).reshaped(to: [1, 1])
    // Ensure indices are Int32 for concatenation with input tokens
    return sampledIndex.scalarType == Int32.self ? sampledIndex : sampledIndex.cast(to: Int32.self)
}
#endif // canImport(CoreML)
