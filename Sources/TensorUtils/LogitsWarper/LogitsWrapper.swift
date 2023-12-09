import Foundation

/// Protocol for all logit warpers that can be applied during generation
public protocol LogitsWarper {
    func callAsFunction(_ arr: [Float]) -> (indexes: [Int], logits: [Float])
}
