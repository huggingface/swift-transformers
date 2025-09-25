import Foundation

/// Logits warper that prevents repetition of previous tokens through a penalty.
///
/// Applies a penalty to tokens that have already been generated, reducing their
/// probability of being selected again. The penalty is applied differently based
/// on the sign of the logit value to maintain numerical stability.
///
/// - Note: Based on https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L294
public struct RepetitionPenaltyWarper: LogitsWarper {
    /// Penalty factor applied to repeated tokens.
    public var penalty: Float

    /// Creates a repetition penalty warper.
    ///
    /// - Parameter penalty: Penalty factor (>1.0 discourages repetition, <1.0 encourages it)
    public init(penalty: Double) {
        self.penalty = Float(penalty)
    }

    /// Applies repetition penalty to the logits.
    ///
    /// For positive logits, divides by penalty. For negative logits, multiplies by penalty.
    /// This asymmetric approach maintains numerical stability while effectively penalizing repetition.
    ///
    /// - Parameters:
    ///   - indices: Token indices to apply penalty to
    ///   - logits: Current logits values
    /// - Returns: Tuple of (indices, penalized logits)
    public func warp(indices: [Int], logits: [Float]) -> (indices: [Int], logits: [Float]) {
        var logits = logits
        for index in indices {
            if logits[index] < 0 {
                logits[index] *= penalty
            } else {
                logits[index] /= penalty
            }
        }

        return (indices, logits)
    }
}
