#if canImport(CoreML)
import CoreML

/// LogitsProcessor for temperature scaling, which effectively controls the randomness
/// of predicted tokens by modulating the logits distribution.
///
/// Temperature < 1.0 makes the model more confident (sharper distribution).
/// Temperature > 1.0 makes the model less confident (flatter distribution).
/// Temperature = 1.0 leaves the distribution unchanged.
///
/// Often used together with `TopPLogitsWarper` and `TopKLogitsWarper`.
///
/// Based on: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L231
@available(macOS 15.0, iOS 18.0, *)
public struct TemperatureLogitsWarper: LogitsProcessor {
    public let temperature: Float

    /// Creates a temperature logits warper.
    ///
    /// - Parameter temperature: Strictly positive float value used to modulate the logits distribution.
    ///   Must be > 0. Values close to 0 approximate greedy decoding.
    public init(temperature: Float) {
        precondition(temperature > 0, "temperature must be strictly positive, got \(temperature)")
        self.temperature = temperature
    }

    public func callAsFunction(_ inputIds: MLTensor, _ scores: MLTensor) async -> MLTensor {
        scores / temperature
    }
}
#endif