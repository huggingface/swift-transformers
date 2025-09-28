import CoreML
import Testing

@testable import Generation

#if canImport(CoreML)
@Suite("Sampling Tests")
struct SamplingTests {
    @Test
    @available(macOS 15.0, iOS 18.0, *)
    func testTopKSampling() {
        let logits = MLTensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        let result = selectNextTokenUsingTopKSampling(
            from: logits,
            temperature: 1.0,
            topK: 3
        )

        #expect(result.shape == [1, 1])
    }

    @Test
    @available(macOS 15.0, iOS 18.0, *)
    func testTopPSampling() {
        let logits = MLTensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        let result = selectNextTokenUsingTopPSampling(
            from: logits,
            temperature: 1.0,
            topP: 0.9
        )

        #expect(result.shape == [1, 1])
    }

    @Test
    @available(macOS 15.0, iOS 18.0, *)
    func testTopPSamplingWithHighP() {
        let logits = MLTensor([[1.0, 2.0, 3.0]])

        let result = selectNextTokenUsingTopPSampling(
            from: logits,
            temperature: 1.0,
            topP: 1.0
        )

        #expect(result.shape == [1, 1])
    }

    @Test
    @available(macOS 15.0, iOS 18.0, *)
    func testGreedyDecoding() {
        let logits = MLTensor([[1.0, 3.0, 2.0]])

        let result = selectNextTokenUsingGreedyDecoding(from: logits)

        #expect(result.shape == [1, 1])
    }

    @Test
    @available(macOS 15.0, iOS 18.0, *)
    func testTemperatureScaling() {
        let logits = MLTensor([[1.0, 2.0, 3.0]])

        let highTempResult = selectNextTokenUsingTopKSampling(
            from: logits,
            temperature: 2.0,
            topK: 2
        )
        #expect(highTempResult.shape == [1, 1])

        let lowTempResult = selectNextTokenUsingTopKSampling(
            from: logits,
            temperature: 0.5,
            topK: 2
        )
        #expect(lowTempResult.shape == [1, 1])
    }
}
#endif
