import Foundation

/// Logits warper that applies temperature scaling to control generation randomness.
///
/// Temperature scaling modifies the "sharpness" of the probability distribution:
/// - Temperature > 1.0: Makes distribution more uniform (more random)
/// - Temperature < 1.0: Makes distribution more peaked (less random)
/// - Temperature = 1.0: No change
public struct TemperatureLogitsWarper: LogitsWarper {
    /// Temperature scaling factor.
    public var temperature: Float

    /// Creates a temperature logits warper.
    ///
    /// - Parameter temperature: Scaling factor (higher values increase randomness)
    public init(temperature: Float) {
        self.temperature = temperature
    }

    /// Applies temperature scaling to the logits.
    ///
    /// Divides each logit by the temperature value, which affects the final
    /// probability distribution after softmax is applied.
    ///
    /// - Parameters:
    ///   - indices: Token indices (unchanged)
    ///   - logits: Current logits values
    /// - Returns: Tuple of (indices, temperature-scaled logits)
    public func warp(indices: [Int], logits: [Float]) -> (indices: [Int], logits: [Float]) {
        (indices: indices, logits: logits.map { $0 / temperature })
    }
}
