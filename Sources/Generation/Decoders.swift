#if canImport(CoreML)
import CoreML

// MARK: Greedy Decoding

@available(macOS 15.0, iOS 18.0, *)
func selectNextTokenUsingGreedyDecoding(from scores: MLTensor) -> MLTensor {
    scores.argmax(alongAxis: -1).reshaped(to: [1, 1])
}

// MARK: Top-K Sampling

@available(macOS 15.0, iOS 18.0, *)
func selectNextTokenUsingTopKSampling(from scores: MLTensor, temperature: Float, topK: Int) -> MLTensor {
    let temperatureAdjustedScores = temperature == 1.0 ? scores : scores / temperature
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

// MARK: Top-P (Nucleus) Sampling

/// Selects the next token using top-p (nucleus) sampling.
///
/// Top-p sampling dynamically selects from the smallest possible set of words
/// whose cumulative probability exceeds the probability p. This provides more
/// diversity than top-k by adapting the vocabulary size based on the probability
/// distribution.
@available(macOS 15.0, iOS 18.0, *)
func selectNextTokenUsingTopPSampling(from scores: MLTensor, temperature: Float, topP: Double) -> MLTensor {
    let temperatureAdjustedScores = temperature == 1.0 ? scores : scores / temperature
    let probs = temperatureAdjustedScores.softmax(alongAxis: -1)

    // Sort probabilities in descending order by negating values first
    let negatedProbs = -probs
    let sortedIndices = negatedProbs.argsort(alongAxis: -1)
    let sortedProbs = probs.gathering(atIndices: sortedIndices, alongAxis: -1)

    // Calculate cumulative sum
    let cumProbs = sortedProbs.cumulativeSum(alongAxis: -1)

    // Find cutoff point - keep tokens where cumulative probability <= topP
    let cutoffMask = cumProbs .<= Float(topP)

    // Always keep at least the first (highest probability) token
    let firstToken = MLTensor(repeating: 1.0, shape: Array(cutoffMask.shape.dropLast()) + [1])
    if cutoffMask.shape.last! > 1 {
        let restMask = cutoffMask[..., 1...]
        let finalMask = MLTensor(concatenating: [firstToken, restMask], alongAxis: -1)

        // Apply mask to sorted probabilities
        let maskedSortedProbs = finalMask * sortedProbs

        // Sample from the masked distribution
        let totalMaskedProb = maskedSortedProbs.sum(alongAxes: [-1]).expandingShape(at: -1)
        let normalizedProbs = maskedSortedProbs / totalMaskedProb

        let rnd = Float.random(in: 0..<1)
        let cumMaskedProbs = normalizedProbs.cumulativeSum(alongAxis: -1)
        var accumProbs = cumMaskedProbs
        accumProbs += (accumProbs .< rnd) * 100.0
        let selectedIdx = accumProbs.argsort()[..., 0]

        let nextTokenTensor = sortedIndices.gathering(
            atIndices: selectedIdx,
            alongAxis: sortedIndices.rank - 1
        )

        return nextTokenTensor.reshaped(to: [1, 1])
    } else {
        // Only one token, just return it
        let selectedIdx = MLTensor([Int32(0)])
        let nextTokenTensor = sortedIndices.gathering(
            atIndices: selectedIdx,
            alongAxis: sortedIndices.rank - 1
        )
        return nextTokenTensor.reshaped(to: [1, 1])
    }
}

#endif // canImport(CoreML)
