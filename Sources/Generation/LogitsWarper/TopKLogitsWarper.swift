#if canImport(Accelerate)
import Accelerate
import Foundation

/// Logits warper that implements top-k filtering for sampling.
///
/// Selects the k most probable tokens and sets all other token probabilities
/// to zero, effectively limiting the sampling space to the top k candidates.
/// This helps balance diversity and quality in generated text.
public struct TopKLogitsWarper: LogitsWarper {
    /// Number of top tokens to keep.
    public var k: Int

    /// Creates a top-k logits warper.
    ///
    /// - Parameter k: Number of top tokens to retain (others are filtered out)
    public init(k: Int) {
        self.k = k
    }

    /// Applies top-k filtering to the logits.
    ///
    /// Uses Accelerate framework's optimized top-k algorithm to efficiently
    /// select the k highest-valued logits and their corresponding indices.
    ///
    /// - Parameters:
    ///   - indices: Current token indices
    ///   - logits: Current logits values
    /// - Returns: Tuple of (top-k indices, top-k logits)
    public func warp(indices: [Int], logits: [Float]) -> (indices: [Int], logits: [Float]) {
        guard !logits.isEmpty else {
            return (indices: [], logits: [])
        }
        let k = min(k, logits.count)
        let arrDescriptor = BNNSNDArrayDescriptor.allocate(
            initializingFrom: logits,
            shape: .vector(logits.count)
        )
        defer {
            arrDescriptor.deallocate()
        }
        let bestIndices = BNNSNDArrayDescriptor.allocateUninitialized(
            scalarType: Int32.self,
            shape: .vector(k)
        )
        defer {
            bestIndices.deallocate()
        }
        let bestValues = BNNSNDArrayDescriptor.allocateUninitialized(
            scalarType: Float.self,
            shape: .vector(k)
        )
        defer {
            bestValues.deallocate()
        }
        try! Accelerate.BNNS.applyTopK(
            k: k,
            input: arrDescriptor,
            bestValues: bestValues,
            bestIndices: bestIndices,
            axis: 0,
            batchSize: 1,
            filterParameters: nil
        )
        let topkLogits = bestValues.data!.withMemoryRebound(to: Float.self, capacity: k) { ptr in
            Array(UnsafeBufferPointer(start: ptr, count: k))
        }
        let topkIndices = bestIndices.data!.withMemoryRebound(to: Int32.self, capacity: k) { ptr in
            Array(UnsafeBufferPointer(start: ptr, count: k))
        }
        return (indices: topkIndices.map { indices[Int($0)] }, logits: topkLogits)
    }
}
#endif // canImport(Accelerate)
