#if canImport(CoreML)
import CoreML

/// Logits warper that applies repetition penalty.
///
/// Repetition penalty reduces the likelihood of generating tokens that have
/// already appeared in the input sequence. This helps reduce repetitive text
/// generation.
///
/// - Note: Penalty > 1.0 penalizes repetition, penalty < 1.0 encourages it
@available(macOS 15.0, iOS 18.0, *)
public struct RepetitionPenaltyWarper: LogitsWarper {
    /// The repetition penalty factor.
    public let penalty: Float

    /// Creates a new repetition penalty warper.
    ///
    /// - Parameter penalty: Penalty factor (must be > 0). Values > 1.0 penalize repetition.
    public init(penalty: Double) {
        precondition(penalty > 0, "Penalty must be strictly positive")
        self.penalty = Float(penalty)
    }

    /// Applies repetition penalty to tokens that appear in the input sequence.
    ///
    /// - Parameters:
    ///   - inputIds: The input token sequence used to identify repeated tokens
    ///   - logits: The logits tensor to modify
    /// - Returns: Logits with repetition penalty applied
    public func warp(inputIds: MLTensor, logits: MLTensor) -> MLTensor {
        if penalty == 1.0 {
            return logits
        }

        // TODO: Implement repetition penalty when MLTensor API allows for easier tensor updates
        // For now, we'll return the original logits to avoid compilation errors
        // This functionality will need to be implemented when tensor item access and update operations are available

        print("Warning: Repetition penalty is not yet implemented due to MLTensor API limitations")
        return logits
    }
}

#endif // canImport(CoreML)
