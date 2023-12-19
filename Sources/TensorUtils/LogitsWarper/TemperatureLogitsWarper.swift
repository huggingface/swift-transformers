import Foundation

public struct TemperatureLogitsWarper: LogitsWarper {
    public var temperature: Float
    
    public init(temperature: Float) {
        self.temperature = temperature
    }

    public func warp(_ arr: [Float]) -> (indexes: [Int], logits: [Float]) {
        let logits = arr.map { $0 / temperature }
        return (indexes: Array(logits.indices), logits: logits)
    }
}
