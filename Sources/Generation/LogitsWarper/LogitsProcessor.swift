import Foundation

/// Processes logits by applying a sequence of logits warpers.
///
/// Coordinates the application of multiple logits warpers in sequence,
/// allowing for complex probability transformations during text generation.
public struct LogitsProcessor {
    /// Array of logits warpers to apply in sequence.
    public var logitsWarpers: [any LogitsWarper]

    /// Creates a new logits processor.
    ///
    /// - Parameter logitsWarpers: Array of warpers to apply in sequence
    public init(logitsWarpers: [any LogitsWarper]) {
        self.logitsWarpers = logitsWarpers
    }

    /// Processes logits by applying all warpers in sequence.
    ///
    /// Each warper is applied to the output of the previous warper, allowing
    /// for complex chaining of probability transformations.
    ///
    /// - Parameter arr: Input logits array
    /// - Returns: Tuple of processed (indices, logits)
    public func callAsFunction(_ arr: [Float]) -> (indices: [Int], logits: [Float]) {
        var indices = Array(arr.indices)
        var logits = arr
        for warper in logitsWarpers {
            (indices, logits) = warper(indices, logits)
        }
        return (indices: indices, logits: logits)
    }
}
