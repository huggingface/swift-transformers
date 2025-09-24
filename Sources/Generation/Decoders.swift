import CoreML

// MARK: Greedy Decoding

func selectNextTokenUsingGreedyDecoding(from scores: MLTensor) -> MLTensor {
    scores.argmax(alongAxis: -1).reshaped(to: [1, 1])
}

// MARK: Top-K Sampling

func selectNextTokenUsingTopKSampling(from scores: MLTensor, temperature: Float, topK: Int) -> MLTensor {
    let temperatureAdjustedScores = scores / temperature
    let (topKScores, topKIndices) = temperatureAdjustedScores.topK(topK)
    let topKProbs = topKScores.softmax(alongAxis: -1)
    let rnd = topKProbs.sum() * Float.random(in: 0 ..< 1)
    var accumTopKProbs = topKProbs.cumulativeSum(alongAxis: -1)
    accumTopKProbs += (accumTopKProbs .< rnd) * 100.0
    let topKIndex = accumTopKProbs.argsort()[..., 0]
    let nextTokenTensor = topKIndices.gathering(
        atIndices: topKIndex,
        alongAxis: topKIndices.rank - 1
    )
    return nextTokenTensor.reshaped(to: [1, 1])
}
