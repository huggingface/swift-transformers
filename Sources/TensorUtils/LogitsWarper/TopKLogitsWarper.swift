import Foundation
import Accelerate

/// Top-K.
/// Select the k most-probable element indices from `arr`
/// and return both the indices (from the original array)
/// and their probabilities.
public struct TopKLogitsWarper: LogitsWarper {
    public var k: Int
    
    public init(k: Int) {
        self.k = k
    }

    public func warp(_ arr: [Float]) -> (indexes: [Int], logits: [Float]) {
        guard !arr.isEmpty else {
            return (indexes: [], logits: [])
        }
        let k = min(k, arr.count)
        let arrDescriptor = BNNSNDArrayDescriptor.allocate(
            initializingFrom: arr,
            shape: .vector(arr.count)
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
        let distances = bestValues.data!.withMemoryRebound(to: Float.self, capacity: k) { ptr in
            Array(UnsafeBufferPointer(start: ptr, count: k))
        }
        let indices = bestIndices.data!.withMemoryRebound(to: Int32.self, capacity: k) { ptr in
            Array(UnsafeBufferPointer(start: ptr, count: k))
        }
        return (indexes: indices.map { Int($0) }, logits: distances)
    }
}
