import CoreML
import Testing

@testable import Generation

#if canImport(CoreML)
@Suite("Logits Warper Tests")
struct LogitsWarperTests {

    @Test("Temperature warper scaling")
    @available(macOS 15.0, iOS 18.0, *)
    func temperatureWarper() {
        let logits = MLTensor([[1.0, 2.0, 3.0]])
        let inputIds = MLTensor([[1, 2]])

        let tempWarper = TemperatureLogitsWarper(temperature: 2.0)
        let warpedLogits = tempWarper.warp(inputIds: inputIds, logits: logits)

        #expect(warpedLogits.shape == logits.shape)

        let identityWarper = TemperatureLogitsWarper(temperature: 1.0)
        let unchangedLogits = identityWarper.warp(inputIds: inputIds, logits: logits)
        #expect(unchangedLogits.shape == logits.shape)
    }

    @Test("LogitsProcessor with multiple warpers")
    @available(macOS 15.0, iOS 18.0, *)
    func logitsProcessor() {
        let logits = MLTensor([[1.0, 2.0, 3.0]])
        let inputIds = MLTensor([[1, 2]])

        let warpers: [LogitsWarper] = [
            TemperatureLogitsWarper(temperature: 2.0)
        ]

        let processor = LogitsProcessor(warpers: warpers)
        let processedLogits = processor.process(inputIds: inputIds, logits: logits)

        #expect(processedLogits.shape == logits.shape)
    }

    @Test("LogitsProcessor with no warpers")
    @available(macOS 15.0, iOS 18.0, *)
    func logitsProcessorEmpty() {
        let logits = MLTensor([[1.0, 2.0, 3.0]])
        let inputIds = MLTensor([[1, 2]])

        let processor = LogitsProcessor(warpers: [])
        let processedLogits = processor.process(inputIds: inputIds, logits: logits)

        #expect(processedLogits.shape == logits.shape)
    }

    @Test("Repetition penalty warper")
    @available(macOS 15.0, iOS 18.0, *)
    func repetitionPenaltyWarper() {
        let logits = MLTensor([[1.0, 2.0, 3.0]])
        let inputIds = MLTensor([[0, 1]])

        let repWarper = RepetitionPenaltyWarper(penalty: 1.2)
        let warpedLogits = repWarper.warp(inputIds: inputIds, logits: logits)

        #expect(warpedLogits.shape == logits.shape)

        let identityWarper = RepetitionPenaltyWarper(penalty: 1.0)
        let unchangedLogits = identityWarper.warp(inputIds: inputIds, logits: logits)
        #expect(unchangedLogits.shape == logits.shape)
    }
}
#endif
