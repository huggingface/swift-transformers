import Foundation
import Testing

@testable import Models

#if canImport(CoreML)
import CoreML

@Suite("LanguageModel Core ML Mask Tests")
struct LanguageModelCoreMLMaskTests {
    @Test("Build full attention mask for prefill")
    func buildFullAttentionMaskForPrefill() async {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let mask = LanguageModel.additiveAttentionMask(queryLength: 3, totalLength: 3)
        let values = await mask.shapedArray(of: Float16.self).scalars
        let blocked = -Float16.greatestFiniteMagnitude

        #expect(values == [
            0, blocked, blocked,
            0, 0, blocked,
            0, 0, 0,
        ])
    }

    @Test("Build sliding attention mask for decode")
    func buildSlidingAttentionMaskForDecode() async {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let mask = LanguageModel.additiveAttentionMask(queryLength: 1, totalLength: 6, slidingWindow: 4)
        let values = await mask.shapedArray(of: Float16.self).scalars
        let blocked = -Float16.greatestFiniteMagnitude

        #expect(values == [
            blocked, blocked, 0, 0, 0, 0,
        ])
    }

    @Test("Build sliding attention mask for prefill with past offset")
    func buildSlidingAttentionMaskForPrefillWithPastOffset() async {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let mask = LanguageModel.additiveAttentionMask(queryLength: 2, totalLength: 5, slidingWindow: 2)
        let values = await mask.shapedArray(of: Float16.self).scalars
        let blocked = -Float16.greatestFiniteMagnitude

        #expect(values == [
            blocked, blocked, 0, 0, blocked,
            blocked, blocked, blocked, 0, 0,
        ])
    }

    @Test("Build stateful mixed-attention model inputs")
    func buildStatefulMixedAttentionInputs() async {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) else {
            return
        }
        let inputIds = MLTensor(shape: [1, 2], scalars: [Int32(10), Int32(20)], scalarType: Int32.self)
        let inputs = LanguageModel.statefulGenerationInputs(
            inputIds: inputIds,
            tokenCount: 5,
            includeAttentionMask: false,
            includeCausalMask: false,
            includeFullAttentionMask: true,
            includeSlidingAttentionMask: true,
            slidingWindow: 2
        )

        #expect(inputs.keys.contains(LanguageModel.Keys.inputIds))
        #expect(inputs.keys.contains(LanguageModel.Keys.fullAttentionMask))
        #expect(inputs.keys.contains(LanguageModel.Keys.slidingAttentionMask))
        #expect(!inputs.keys.contains(LanguageModel.Keys.causalMask))
        #expect(!inputs.keys.contains(LanguageModel.Keys.attentionMask))

        let fullMask = inputs[LanguageModel.Keys.fullAttentionMask]!
        let slidingMask = inputs[LanguageModel.Keys.slidingAttentionMask]!
        #expect(fullMask.shape == [1, 1, 2, 5])
        #expect(slidingMask.shape == [1, 1, 2, 5])

        let blocked = -Float16.greatestFiniteMagnitude
        let fullValues = await fullMask.shapedArray(of: Float16.self).scalars
        let slidingValues = await slidingMask.shapedArray(of: Float16.self).scalars

        #expect(fullValues == [
            0, 0, 0, 0, blocked,
            0, 0, 0, 0, 0,
        ])
        #expect(slidingValues == [
            blocked, blocked, 0, 0, blocked,
            blocked, blocked, blocked, 0, 0,
        ])
    }
}
#endif // canImport(CoreML)
