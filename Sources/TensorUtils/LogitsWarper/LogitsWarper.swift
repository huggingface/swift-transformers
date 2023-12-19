import Foundation

/// Protocol for all logit warpers that can be applied during generation
public protocol LogitsWarper {
    func warp(_ arr: [Float]) -> (indexes: [Int], logits: [Float])
    func callAsFunction(_ arr: [Float]) -> (indexes: [Int], logits: [Float])
}

extension LogitsWarper {
    public func callAsFunction(_ arr: [Float]) -> (indexes: [Int], logits: [Float]) {
        warp(arr)
    }
}
