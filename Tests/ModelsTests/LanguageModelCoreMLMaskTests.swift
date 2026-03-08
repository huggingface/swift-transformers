import Foundation
import Testing

@testable import Models

#if canImport(CoreML)
import CoreML

@Suite("LanguageModel Core ML Mask Tests")
struct LanguageModelCoreMLMaskTests {
    @Test("Build full attention mask for prefill")
    func buildFullAttentionMaskForPrefill() async {
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
        let mask = LanguageModel.additiveAttentionMask(queryLength: 1, totalLength: 6, slidingWindow: 4)
        let values = await mask.shapedArray(of: Float16.self).scalars
        let blocked = -Float16.greatestFiniteMagnitude

        #expect(values == [
            blocked, blocked, 0, 0, 0, 0,
        ])
    }

    @Test("Build sliding attention mask for prefill with past offset")
    func buildSlidingAttentionMaskForPrefillWithPastOffset() async {
        let mask = LanguageModel.additiveAttentionMask(queryLength: 2, totalLength: 5, slidingWindow: 2)
        let values = await mask.shapedArray(of: Float16.self).scalars
        let blocked = -Float16.greatestFiniteMagnitude

        #expect(values == [
            blocked, blocked, 0, 0, blocked,
            blocked, blocked, blocked, 0, 0,
        ])
    }
}
#endif // canImport(CoreML)
