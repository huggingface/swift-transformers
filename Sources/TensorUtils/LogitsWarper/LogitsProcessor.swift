import Foundation

public struct LogitsProcessor {
    public var logitsWrappers: [any LogitsWarper]

    public init(logitsWrappers: [any LogitsWarper]) {
        self.logitsWrappers = logitsWrappers
    }

    public func callAsFunction(_ arr: [Float]) -> (indexes: [Int], logits: [Float]) {
        var indexes = Array(arr.indices)
        var logits = arr
        for wrapper in logitsWrappers {
            (indexes, logits) = wrapper(logits)
        }
        return (indexes: indexes, logits: logits)
    }
}
