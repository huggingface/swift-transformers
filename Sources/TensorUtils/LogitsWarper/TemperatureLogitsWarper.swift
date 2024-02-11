import Foundation

public struct TemperatureLogitsWarper: LogitsWarper {
    public var temperature: Float
    
    public init(temperature: Float) {
        self.temperature = temperature
    }

    public func warp(indexes: [Int], logits: [Float]) -> (indexes: [Int], logits: [Float]) {
        return (indexes: indexes, logits: logits.map { $0 / temperature })
    }
}
