import Foundation

/// Protocol for all logit warpers that can be applied during generation
public protocol LogitsWarper {
    func warp(indexes: [Int], logits: [Float]) -> (indexes: [Int], logits: [Float])
    func callAsFunction(_ indexes: [Int], _ logits: [Float]) -> (indexes: [Int], logits: [Float])
}

extension LogitsWarper {
    public func callAsFunction(_ indexes: [Int], _ logits: [Float]) -> (indexes: [Int], logits: [Float]) {
        warp(indexes: indexes, logits: logits)
    }
}
