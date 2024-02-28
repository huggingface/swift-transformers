import Foundation

public struct LogitsProcessor {
    public var logitsWarpers: [any LogitsWarper]

    public init(logitsWarpers: [any LogitsWarper]) {
        self.logitsWarpers = logitsWarpers
    }

    public func callAsFunction(_ arr: [Float]) -> (indexes: [Int], logits: [Float]) {
        var indexes = Array(arr.indices)
        var logits = arr
        for warper in logitsWarpers {
            (indexes, logits) = warper(indexes, logits)
        }
        return (indexes: indexes, logits: logits)
    }
}
