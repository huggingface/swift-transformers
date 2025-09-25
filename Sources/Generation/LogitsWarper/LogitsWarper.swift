import Foundation

/// Protocol for logits warpers that transform token probabilities during generation.
///
/// Logits warpers modify the probability distribution over tokens before sampling,
/// enabling techniques like temperature scaling, top-k/top-p filtering, and repetition penalties.
public protocol LogitsWarper {
    /// Warps the logits and corresponding indices.
    ///
    /// - Parameters:
    ///   - indices: Array of token indices corresponding to the logits
    ///   - logits: Array of logit values to transform
    /// - Returns: Tuple of transformed (indices, logits)
    func warp(indices: [Int], logits: [Float]) -> (indices: [Int], logits: [Float])

    /// Convenience method that calls the warp function.
    ///
    /// - Parameters:
    ///   - indices: Array of token indices
    ///   - logits: Array of logit values
    /// - Returns: Tuple of transformed (indices, logits)
    func callAsFunction(_ indices: [Int], _ logits: [Float]) -> (indices: [Int], logits: [Float])
}

public extension LogitsWarper {
    /// Default implementation of callAsFunction that delegates to warp.
    ///
    /// - Parameters:
    ///   - indices: Array of token indices
    ///   - logits: Array of logit values
    /// - Returns: Tuple of transformed (indices, logits)
    func callAsFunction(_ indices: [Int], _ logits: [Float]) -> (indices: [Int], logits: [Float]) {
        warp(indices: indices, logits: logits)
    }
}
