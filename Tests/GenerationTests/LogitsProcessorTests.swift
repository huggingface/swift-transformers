#if canImport(CoreML)
import CoreML
import Testing

@testable import Generation

@Suite
final class LogitsProcessorTests {
    private let accuracy: Float = 0.0001

    // MARK: - Temperature Tests
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testTemperatureWarper() async throws {
        let warper = try TemperatureLogitsWarper(temperature: 2.0)

        // Create input: batch_size=1, seq_len=3
        let inputIds = MLTensor(shape: [1, 3], scalars: [Int32(1), Int32(2), Int32(3)], scalarType: Int32.self)
        // Create scores: batch_size=1, vocab_size=3
        let scores = MLTensor(shape: [1, 3], scalars: [Float(2.0), Float(4.0), Float(6.0)], scalarType: Float.self)

        let result = await warper(inputIds, scores)
        let expected: [Float] = [1.0, 2.0, 3.0] // Each score divided by 2.0

        await assertMLTensorEqual(result, expected: expected, accuracy: accuracy)
    }
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testTemperatureWarperWithDifferentValues() async throws {
        // Test temperature < 1 (sharper distribution)
        let sharper = try TemperatureLogitsWarper(temperature: 0.5)
        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 2], scalars: [Float(1.0), Float(2.0)], scalarType: Float.self)

        let result = await sharper(inputIds, scores)
        let expected: [Float] = [2.0, 4.0] // Divided by 0.5 = multiplied by 2

        await assertMLTensorEqual(result, expected: expected, accuracy: accuracy)
    }

    // MARK: - Top-K Tests
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testTopKWarper() async throws {
        let warper = try TopKLogitsWarper(topK: 3)

        let inputIds = MLTensor(shape: [1, 2], scalars: [Int32(1), Int32(2)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 5], scalars: [Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)], scalarType: Float.self)

        let result = await warper(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        // Top 3 tokens (5, 4, 3) should remain, others should be -inf
        #expect(resultArray[0].isInfinite && resultArray[0] < 0, "Token 0 should be -inf")
        #expect(resultArray[1].isInfinite && resultArray[1] < 0, "Token 1 should be -inf")
        #expect(abs(resultArray[2] - 3.0) <= accuracy, "Token 2 should be kept")
        #expect(abs(resultArray[3] - 4.0) <= accuracy, "Token 3 should be kept")
        #expect(abs(resultArray[4] - 5.0) <= accuracy, "Token 4 should be kept")
    }
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testTopKWarperWithSmallK() async throws {
        let warper = try TopKLogitsWarper(topK: 1)

        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 3], scalars: [Float(1.0), Float(5.0), Float(3.0)], scalarType: Float.self)

        let result = await warper(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        // Only token with score 5.0 should remain
        #expect(resultArray[0].isInfinite && resultArray[0] < 0)
        #expect(abs(resultArray[1] - 5.0) <= accuracy)
        #expect(resultArray[2].isInfinite && resultArray[2] < 0)
    }

    // MARK: - Top-P Tests
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testTopPWarper() async throws {
        let warper = try TopPLogitsWarper(topP: 0.9)

        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        // Create a distribution where top tokens dominate: [0.0, 1.0, 2.0, 3.0, 10.0]
        // After softmax, token 4 will have ~99.7% probability
        let scores = MLTensor(shape: [1, 5], scalars: [Float(0.0), Float(1.0), Float(2.0), Float(3.0), Float(10.0)], scalarType: Float.self)

        let result = await warper(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        // Token 4 (score 10.0) should definitely be kept (highest probability)
        #expect(!(resultArray[4].isInfinite), "Highest probability token should be kept")

        // Some lower tokens should be filtered to -inf
        let filteredCount = resultArray.filter { $0.isInfinite && $0 < 0 }.count
        #expect(filteredCount > 0, "Top-P should filter some low-probability tokens")
    }
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testTopPWarperWithHighThreshold() async throws {
        // With topP=0.99, almost all tokens should be kept
        let warper = try TopPLogitsWarper(topP: 0.99)

        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 5], scalars: [Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)], scalarType: Float.self)

        let result = await warper(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        // With high topP and relatively uniform distribution, most tokens should be kept
        let keptCount = resultArray.filter { !($0.isInfinite && $0 < 0) }.count
        #expect(keptCount >= 4, "High topP should keep most tokens")
    }

    // MARK: - Repetition Penalty Tests
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testRepetitionPenaltyProcessor() async throws {
        let processor = try RepetitionPenaltyLogitsProcessor(penalty: 2.0)

        // Input sequence with tokens [1, 2, 3]
        let inputIds = MLTensor(shape: [1, 3], scalars: [Int32(1), Int32(2), Int32(3)], scalarType: Int32.self)

        // Scores for vocab of size 5: [0.5, -0.5, 1.0, -1.0, 2.0]
        let scores = MLTensor(shape: [1, 5], scalars: [Float(0.5), Float(-0.5), Float(1.0), Float(-1.0), Float(2.0)], scalarType: Float.self)

        let result = await processor(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        // Token 0: not in sequence, unchanged
        #expect(abs(resultArray[0] - 0.5) <= accuracy, "Token 0 should be unchanged")

        // Token 1 (score -0.5 < 0): multiplied by penalty = -1.0
        #expect(abs(resultArray[1] - -1.0) <= accuracy, "Token 1 should be penalized (negative)")

        // Token 2 (score 1.0 > 0): divided by penalty = 0.5
        #expect(abs(resultArray[2] - 0.5) <= accuracy, "Token 2 should be penalized (positive)")

        // Token 3 (score -1.0 < 0): multiplied by penalty = -2.0
        #expect(abs(resultArray[3] - -2.0) <= accuracy, "Token 3 should be penalized (negative)")

        // Token 4: not in sequence, unchanged
        #expect(abs(resultArray[4] - 2.0) <= accuracy, "Token 4 should be unchanged")
    }
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testRepetitionPenaltyWithNoPenalty() async throws {
        let processor = try RepetitionPenaltyLogitsProcessor(penalty: 1.0)

        let inputIds = MLTensor(shape: [1, 2], scalars: [Int32(1), Int32(2)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 5], scalars: [Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)], scalarType: Float.self)

        let result = await processor(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars
        let expectedArray = await scores.shapedArray(of: Float.self).scalars

        // With penalty=1.0, scores should be unchanged
        #expect(resultArray == expectedArray, "Penalty of 1.0 should not change scores")
    }
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testRepetitionPenaltyWithRank3Scores() async throws {
        let processor = try RepetitionPenaltyLogitsProcessor(penalty: 2.0)

        // Input sequence with tokens [1, 2, 3]
        let inputIds = MLTensor(shape: [1, 3], scalars: [Int32(1), Int32(2), Int32(3)], scalarType: Int32.self)

        // Scores shaped as [batch, sequence_length, vocab] -> [1, 1, 5]
        let scores = MLTensor(
            shape: [1, 1, 5],
            scalars: [Float(0.5), Float(-0.5), Float(1.0), Float(-1.0), Float(2.0)],
            scalarType: Float.self
        )

        let result = await processor(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        let expected: [Float] = [0.5, -1.0, 0.5, -2.0, 2.0]
        #expect(resultArray.count == expected.count, "Flattened tensor mismatch")
        for (value, exp) in zip(resultArray, expected) {
            #expect(abs(value - exp) <= accuracy)
        }
    }

    // MARK: - Processor List Tests
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testLogitsProcessorList() async throws {
        let temp = try TemperatureLogitsWarper(temperature: 2.0)
        let topK = try TopKLogitsWarper(topK: 3)
        let processorList = LogitsProcessorList(processors: [temp, topK])

        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 5], scalars: [Float(2.0), Float(4.0), Float(6.0), Float(8.0), Float(10.0)], scalarType: Float.self)

        // First temperature divides by 2: [1, 2, 3, 4, 5]
        // Then top-k keeps top 3: [-inf, -inf, 3, 4, 5]
        let result = await processorList(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        #expect(resultArray[0].isInfinite && resultArray[0] < 0)
        #expect(resultArray[1].isInfinite && resultArray[1] < 0)
        #expect(abs(resultArray[2] - 3.0) <= accuracy)
        #expect(abs(resultArray[3] - 4.0) <= accuracy)
        #expect(abs(resultArray[4] - 5.0) <= accuracy)
    }
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testEmptyProcessorList() async throws {
        let processorList = LogitsProcessorList(processors: [])

        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 3], scalars: [Float(1.0), Float(2.0), Float(3.0)], scalarType: Float.self)

        let result = await processorList(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars
        let expectedArray = await scores.shapedArray(of: Float.self).scalars

        // Should be unchanged
        #expect(resultArray == expectedArray)
    }

    // MARK: - Min-P Tests
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testMinPWarper() async throws {
        let warper = try MinPLogitsWarper(minP: 0.1)

        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        // Scores: [1.0, 2.0, 3.0, 4.0, 5.0]
        // After softmax, probabilities will be computed
        // Max prob will be for score=5.0
        // Min threshold = 0.1 * max_prob
        // Tokens with prob < threshold should be filtered
        let scores = MLTensor(shape: [1, 5], scalars: [Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)], scalarType: Float.self)

        let result = await warper(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        // Compute expected: softmax probabilities manually
        let scoresArray = await scores.shapedArray(of: Float.self).scalars
        let expScores = scoresArray.map { exp($0) }
        let sumExp = expScores.reduce(0, +)
        let probs = expScores.map { $0 / sumExp }
        let maxProb = probs.max()!
        let threshold = 0.1 * maxProb

        // Check that low probability tokens are filtered
        for (idx, prob) in probs.enumerated() {
            if prob < threshold {
                #expect(resultArray[idx].isInfinite && resultArray[idx] < 0, "Token \(idx) with prob \(prob) should be filtered")
            } else {
                #expect(abs(resultArray[idx] - scoresArray[idx]) <= accuracy, "Token \(idx) should not be filtered")
            }
        }
    }
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testMinPWarperKeepsMinTokens() async throws {
        // Even with aggressive minP, should keep at least minTokensToKeep tokens
        let warper = try MinPLogitsWarper(minP: 0.99, minTokensToKeep: 2)

        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 5], scalars: [Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)], scalarType: Float.self)

        let result = await warper(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        // Count non-infinite values
        let nonInfiniteCount = resultArray.filter { !$0.isInfinite }.count
        #expect(nonInfiniteCount >= 2, "Should keep at least 2 tokens")
    }
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testMinPWarperWithLowThreshold() async throws {
        // With very low minP, most tokens should pass
        let warper = try MinPLogitsWarper(minP: 0.001)

        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 5], scalars: [Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)], scalarType: Float.self)

        let result = await warper(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        // Most or all tokens should remain
        let nonInfiniteCount = resultArray.filter { !$0.isInfinite }.count
        #expect(nonInfiniteCount >= 4, "With low minP, most tokens should pass")
    }
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testMinPWarperInvalidParameters() {
        // Test invalid minP
        do {
            _ = try MinPLogitsWarper(minP: -0.1)
            Issue.record("Expected MinPLogitsWarper(minP: -0.1) to throw")
        } catch {}
        do {
            _ = try MinPLogitsWarper(minP: 1.5)
            Issue.record("Expected MinPLogitsWarper(minP: 1.5) to throw")
        } catch {}

        // Test invalid minTokensToKeep
        do {
            _ = try MinPLogitsWarper(minP: 0.1, minTokensToKeep: 0)
            Issue.record("Expected MinPLogitsWarper(minTokensToKeep: 0) to throw")
        } catch {}
        do {
            _ = try MinPLogitsWarper(minP: 0.1, minTokensToKeep: -1)
            Issue.record("Expected MinPLogitsWarper(minTokensToKeep: -1) to throw")
        } catch {}
    }

    // MARK: - Parameter Validation Tests
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testTemperatureWarperInvalidParameters() {
        // Test invalid temperature values
        do {
            _ = try TemperatureLogitsWarper(temperature: 0.0)
            Issue.record("Expected TemperatureLogitsWarper(temperature: 0.0) to throw")
        } catch {}
        do {
            _ = try TemperatureLogitsWarper(temperature: -1.0)
            Issue.record("Expected TemperatureLogitsWarper(temperature: -1.0) to throw")
        } catch {}
    }
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testTopKWarperInvalidParameters() {
        // Test invalid topK values
        do {
            _ = try TopKLogitsWarper(topK: 0)
            Issue.record("Expected TopKLogitsWarper(topK: 0) to throw")
        } catch {}
        do {
            _ = try TopKLogitsWarper(topK: -1)
            Issue.record("Expected TopKLogitsWarper(topK: -1) to throw")
        } catch {}

        // Test invalid minTokensToKeep
        do {
            _ = try TopKLogitsWarper(topK: 5, minTokensToKeep: 0)
            Issue.record("Expected TopKLogitsWarper(minTokensToKeep: 0) to throw")
        } catch {}
        do {
            _ = try TopKLogitsWarper(topK: 5, minTokensToKeep: -1)
            Issue.record("Expected TopKLogitsWarper(minTokensToKeep: -1) to throw")
        } catch {}
    }
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testTopPWarperInvalidParameters() {
        // Test invalid topP values
        do {
            _ = try TopPLogitsWarper(topP: -0.1)
            Issue.record("Expected TopPLogitsWarper(topP: -0.1) to throw")
        } catch {}
        do {
            _ = try TopPLogitsWarper(topP: 1.5)
            Issue.record("Expected TopPLogitsWarper(topP: 1.5) to throw")
        } catch {}

        // Test invalid minTokensToKeep
        do {
            _ = try TopPLogitsWarper(topP: 0.9, minTokensToKeep: 0)
            Issue.record("Expected TopPLogitsWarper(minTokensToKeep: 0) to throw")
        } catch {}
        do {
            _ = try TopPLogitsWarper(topP: 0.9, minTokensToKeep: -1)
            Issue.record("Expected TopPLogitsWarper(minTokensToKeep: -1) to throw")
        } catch {}
    }
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test
    func testRepetitionPenaltyInvalidParameters() {
        // Test invalid penalty values
        do {
            _ = try RepetitionPenaltyLogitsProcessor(penalty: 0.0)
            Issue.record("Expected RepetitionPenaltyLogitsProcessor(penalty: 0.0) to throw")
        } catch {}
        do {
            _ = try RepetitionPenaltyLogitsProcessor(penalty: -1.0)
            Issue.record("Expected RepetitionPenaltyLogitsProcessor(penalty: -1.0) to throw")
        } catch {}
    }
}

// MARK: - Test Helpers

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func assertMLTensorEqual(
    _ tensor: MLTensor,
    expected: [Float],
    accuracy: Float
) async {
    let actual = await tensor.shapedArray(of: Float.self).scalars
    #expect(actual.count == expected.count, "Tensor size mismatch")
    for (a, e) in zip(actual, expected) {
        #expect(abs(a - e) <= accuracy)
    }
}
#endif
