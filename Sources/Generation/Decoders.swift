#if canImport(CoreML)
import CoreML

// MARK: Greedy Decoding

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
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
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func selectNextTokenUsingSampling(from scores: MLTensor) async -> MLTensor {
    // Convert logits to probabilities
    let probs = scores.softmax(alongAxis: -1)

    // Multinomial sampling via inverse CDF, searched on the CPU.
    //
    // CPU inverse-CDF search, replacing optimized tensor argmin path that breaks for vocabs > 2^16 (#365).
    let cumulativeProbs = probs.cumulativeSum(alongAxis: -1)
    let floatCumulativeProbs = cumulativeProbs.scalarType == Float.self ? cumulativeProbs : cumulativeProbs.cast(to: Float.self)
    let cdf = await floatCumulativeProbs.shapedArray(of: Float.self).scalars

    let vocabSize = scores.shape.last ?? 1
    let batchSize = max(cdf.count / vocabSize, 1)

    var sampledTokens = [Int32]()
    sampledTokens.reserveCapacity(batchSize)
    for batch in 0..<batchSize {
        let row = cdf[(batch * vocabSize)..<((batch + 1) * vocabSize)]
        // Scale the draw by the total mass to be robust to floating-point
        // rounding in the last CDF entry.
        let rnd = Float.random(in: 0..<1) * (row.last ?? 1)

        // Binary search for the first index whose cumulative probability
        // reaches the drawn value.
        var low = row.startIndex
        var high = row.endIndex - 1
        while low < high {
            let mid = (low + high) / 2
            if row[mid] < rnd {
                low = mid + 1
            } else {
                high = mid
            }
        }
        sampledTokens.append(Int32(low - row.startIndex))
    }

    // Int32 indices, shaped [batch_size, 1] for concatenation with input tokens
    return MLTensor(shape: [batchSize, 1], scalars: sampledTokens)
}
#endif // canImport(CoreML)
