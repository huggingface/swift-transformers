#if canImport(CoreML)
import CoreML

/// Protocol for modifying logits before token sampling.
///
/// Logits warpers can be used to apply various transformations to the logits
/// distribution before sampling, such as temperature scaling, top-k filtering,
/// top-p (nucleus) filtering, or repetition penalties.
@available(macOS 15.0, iOS 18.0, *)
public protocol LogitsWarper {
    /// Warps (modifies) the logits before sampling.
    ///
    /// - Parameters:
    ///   - inputIds: The input token sequence used for context-dependent warping
    ///   - logits: The logits tensor to be modified
    /// - Returns: The modified logits tensor
    func warp(inputIds: MLTensor, logits: MLTensor) -> MLTensor

    /// Alternative call syntax for convenience.
    func callAsFunction(inputIds: MLTensor, logits: MLTensor) -> MLTensor
}

@available(macOS 15.0, iOS 18.0, *)
public extension LogitsWarper {
    /// Default implementation of callAsFunction that delegates to warp.
    func callAsFunction(inputIds: MLTensor, logits: MLTensor) -> MLTensor {
        warp(inputIds: inputIds, logits: logits)
    }
}

/// A collection of logits warpers that processes logits sequentially.
@available(macOS 15.0, iOS 18.0, *)
public struct LogitsProcessor {
    private let warpers: [LogitsWarper]

    /// Creates a new logits processor with the specified warpers.
    ///
    /// - Parameter warpers: Array of logits warpers to apply sequentially
    public init(warpers: [LogitsWarper] = []) {
        self.warpers = warpers
    }

    /// Applies all warpers sequentially to the logits.
    ///
    /// - Parameters:
    ///   - inputIds: The input token sequence
    ///   - logits: The logits tensor to process
    /// - Returns: The processed logits tensor
    public func process(inputIds: MLTensor, logits: MLTensor) -> MLTensor {
        var processedLogits = logits
        for warper in warpers {
            processedLogits = warper.warp(inputIds: inputIds, logits: processedLogits)
        }
        return processedLogits
    }

    /// Alternative call syntax for convenience.
    public func callAsFunction(inputIds: MLTensor, logits: MLTensor) -> MLTensor {
        process(inputIds: inputIds, logits: logits)
    }
}

#endif // canImport(CoreML)
