import Foundation

/// Logits warper that implements nucleus (top-p) sampling.
///
/// Selects the smallest set of tokens whose cumulative probability exceeds
/// the threshold p, providing dynamic vocabulary selection based on the
/// probability distribution rather than a fixed number of tokens.
///
/// - Note: Based on https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
public struct TopPLogitsWarper: LogitsWarper {
    /// Cumulative probability threshold.
    public var p: Float

    /// Creates a top-p (nucleus) logits warper.
    ///
    /// - Parameter p: Cumulative probability threshold (0.0 to 1.0)
    public init(p: Float) {
        self.p = p
    }

    /// Applies top-p (nucleus) filtering to the logits.
    ///
    /// Computes softmax probabilities, sorts by probability, and selects tokens
    /// until their cumulative probability exceeds the threshold p.
    ///
    /// - Parameters:
    ///   - indices: Current token indices
    ///   - logits: Current logits values
    /// - Returns: Tuple of (filtered indices, filtered logits)
    public func warp(indices: [Int], logits: [Float]) -> (indices: [Int], logits: [Float]) {
        guard !logits.isEmpty else {
            return (indices: [], logits: [])
        }

        let arrSoftmax = Math.softmax(logits)
        var indexLogitProb = [(index: Int, logit: Float, prob: Float)]()
        indexLogitProb.reserveCapacity(logits.count)
        for (index, data) in zip(logits, arrSoftmax).enumerated() {
            indexLogitProb.append((index: index, logit: data.0, prob: data.1))
        }
        indexLogitProb.sort { $0.prob > $1.prob }

        let cumsum = Math.cumsum(indexLogitProb.map(\.prob))
        var sliceIndex = cumsum.count - 1
        for (index, element) in cumsum.enumerated() where element > p {
            sliceIndex = index
            break
        }

        let toppIndices = indexLogitProb[0...sliceIndex].map { indices[$0.index] }
        let toppLogits = indexLogitProb[0...sliceIndex].map(\.logit)
        return (indices: toppIndices, logits: toppLogits)
    }
}
