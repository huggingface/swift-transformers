#if canImport(CoreML)
import CoreML

/// Logits warper that applies temperature scaling.
///
/// Temperature scaling modifies the sharpness of the probability distribution:
/// - Temperature < 1.0: Makes the distribution more concentrated (less random)
/// - Temperature = 1.0: No change to the distribution
/// - Temperature > 1.0: Makes the distribution more uniform (more random)
@available(macOS 15.0, iOS 18.0, *)
public struct TemperatureLogitsWarper: LogitsWarper {
    /// The temperature value for scaling logits.
    public let temperature: Float

    /// Creates a new temperature logits warper.
    ///
    /// - Parameter temperature: Temperature value (must be > 0)
    public init(temperature: Double) {
        precondition(temperature > 0, "Temperature must be strictly positive")
        self.temperature = Float(temperature)
    }

    /// Applies temperature scaling to the logits.
    ///
    /// - Parameters:
    ///   - inputIds: The input token sequence (unused by temperature warper)
    ///   - logits: The logits tensor to scale
    /// - Returns: Temperature-scaled logits
    public func warp(inputIds: MLTensor, logits: MLTensor) -> MLTensor {
        if temperature == 1.0 {
            return logits
        }
        return logits / temperature
    }
}

#endif // canImport(CoreML)
