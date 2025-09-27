import CoreML
import XCTest

@testable import Generation

@available(macOS 15.0, iOS 18.0, *)
final class LogitsProcessorTests: XCTestCase {
    private let accuracy: Float = 0.0001

    // MARK: - Temperature Tests

    func testTemperatureWarper() async throws {
        let warper = TemperatureLogitsWarper(temperature: 2.0)

        // Create input: batch_size=1, seq_len=3
        let inputIds = MLTensor(shape: [1, 3], scalars: [Int32(1), Int32(2), Int32(3)], scalarType: Int32.self)
        // Create scores: batch_size=1, vocab_size=3
        let scores = MLTensor(shape: [1, 3], scalars: [Float(2.0), Float(4.0), Float(6.0)], scalarType: Float.self)

        let result = await warper(inputIds, scores)
        let expected: [Float] = [1.0, 2.0, 3.0] // Each score divided by 2.0

        await assertMLTensorEqual(result, expected: expected, accuracy: accuracy)
    }

    func testTemperatureWarperWithDifferentValues() async throws {
        // Test temperature < 1 (sharper distribution)
        let sharper = TemperatureLogitsWarper(temperature: 0.5)
        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 2], scalars: [Float(1.0), Float(2.0)], scalarType: Float.self)

        let result = await sharper(inputIds, scores)
        let expected: [Float] = [2.0, 4.0] // Divided by 0.5 = multiplied by 2

        await assertMLTensorEqual(result, expected: expected, accuracy: accuracy)
    }

    // MARK: - Top-K Tests

    func testTopKWarper() async throws {
        let warper = TopKLogitsWarper(topK: 3)

        let inputIds = MLTensor(shape: [1, 2], scalars: [Int32(1), Int32(2)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 5], scalars: [Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)], scalarType: Float.self)

        let result = await warper(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        // Top 3 tokens (5, 4, 3) should remain, others should be -inf
        XCTAssertTrue(resultArray[0].isInfinite && resultArray[0] < 0, "Token 0 should be -inf")
        XCTAssertTrue(resultArray[1].isInfinite && resultArray[1] < 0, "Token 1 should be -inf")
        XCTAssertEqual(resultArray[2], 3.0, accuracy: accuracy, "Token 2 should be kept")
        XCTAssertEqual(resultArray[3], 4.0, accuracy: accuracy, "Token 3 should be kept")
        XCTAssertEqual(resultArray[4], 5.0, accuracy: accuracy, "Token 4 should be kept")
    }

    func testTopKWarperWithSmallK() async throws {
        let warper = TopKLogitsWarper(topK: 1)

        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 3], scalars: [Float(1.0), Float(5.0), Float(3.0)], scalarType: Float.self)

        let result = await warper(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        // Only token with score 5.0 should remain
        XCTAssertTrue(resultArray[0].isInfinite && resultArray[0] < 0)
        XCTAssertEqual(resultArray[1], 5.0, accuracy: accuracy)
        XCTAssertTrue(resultArray[2].isInfinite && resultArray[2] < 0)
    }

    // MARK: - Top-P Tests

    func testTopPWarper() async throws {
        let warper = TopPLogitsWarper(topP: 0.9)

        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        // Create a distribution where top tokens dominate: [0.0, 1.0, 2.0, 3.0, 10.0]
        // After softmax, token 4 will have ~99.7% probability
        let scores = MLTensor(shape: [1, 5], scalars: [Float(0.0), Float(1.0), Float(2.0), Float(3.0), Float(10.0)], scalarType: Float.self)

        let result = await warper(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        // Token 4 (score 10.0) should definitely be kept (highest probability)
        XCTAssertFalse(resultArray[4].isInfinite, "Highest probability token should be kept")

        // Some lower tokens should be filtered to -inf
        let filteredCount = resultArray.filter { $0.isInfinite && $0 < 0 }.count
        XCTAssertTrue(filteredCount > 0, "Top-P should filter some low-probability tokens")
    }

    func testTopPWarperWithHighThreshold() async throws {
        // With topP=0.99, almost all tokens should be kept
        let warper = TopPLogitsWarper(topP: 0.99)

        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 5], scalars: [Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)], scalarType: Float.self)

        let result = await warper(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        // With high topP and relatively uniform distribution, most tokens should be kept
        let keptCount = resultArray.filter { !($0.isInfinite && $0 < 0) }.count
        XCTAssertTrue(keptCount >= 4, "High topP should keep most tokens")
    }

    // MARK: - Repetition Penalty Tests

    func testRepetitionPenaltyProcessor() async throws {
        let processor = RepetitionPenaltyLogitsProcessor(penalty: 2.0)

        // Input sequence with tokens [1, 2, 3]
        let inputIds = MLTensor(shape: [1, 3], scalars: [Int32(1), Int32(2), Int32(3)], scalarType: Int32.self)

        // Scores for vocab of size 5: [0.5, -0.5, 1.0, -1.0, 2.0]
        let scores = MLTensor(shape: [1, 5], scalars: [Float(0.5), Float(-0.5), Float(1.0), Float(-1.0), Float(2.0)], scalarType: Float.self)

        let result = await processor(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        // Token 0: not in sequence, unchanged
        XCTAssertEqual(resultArray[0], 0.5, accuracy: accuracy, "Token 0 should be unchanged")

        // Token 1 (score -0.5 < 0): multiplied by penalty = -1.0
        XCTAssertEqual(resultArray[1], -1.0, accuracy: accuracy, "Token 1 should be penalized (negative)")

        // Token 2 (score 1.0 > 0): divided by penalty = 0.5
        XCTAssertEqual(resultArray[2], 0.5, accuracy: accuracy, "Token 2 should be penalized (positive)")

        // Token 3 (score -1.0 < 0): multiplied by penalty = -2.0
        XCTAssertEqual(resultArray[3], -2.0, accuracy: accuracy, "Token 3 should be penalized (negative)")

        // Token 4: not in sequence, unchanged
        XCTAssertEqual(resultArray[4], 2.0, accuracy: accuracy, "Token 4 should be unchanged")
    }

    func testRepetitionPenaltyWithNoPenalty() async throws {
        let processor = RepetitionPenaltyLogitsProcessor(penalty: 1.0)

        let inputIds = MLTensor(shape: [1, 2], scalars: [Int32(1), Int32(2)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 5], scalars: [Float(1.0), Float(2.0), Float(3.0), Float(4.0), Float(5.0)], scalarType: Float.self)

        let result = await processor(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars
        let expectedArray = await scores.shapedArray(of: Float.self).scalars

        // With penalty=1.0, scores should be unchanged
        XCTAssertEqual(resultArray, expectedArray, "Penalty of 1.0 should not change scores")
    }

    func testRepetitionPenaltyWithRank3Scores() async throws {
        let processor = RepetitionPenaltyLogitsProcessor(penalty: 2.0)

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
        XCTAssertEqual(resultArray.count, expected.count, "Flattened tensor mismatch")
        for (value, exp) in zip(resultArray, expected) {
            XCTAssertEqual(value, exp, accuracy: accuracy)
        }
    }

    // MARK: - Processor List Tests

    func testLogitsProcessorList() async throws {
        let temp = TemperatureLogitsWarper(temperature: 2.0)
        let topK = TopKLogitsWarper(topK: 3)
        let processorList = LogitsProcessorList(processors: [temp, topK])

        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 5], scalars: [Float(2.0), Float(4.0), Float(6.0), Float(8.0), Float(10.0)], scalarType: Float.self)

        // First temperature divides by 2: [1, 2, 3, 4, 5]
        // Then top-k keeps top 3: [-inf, -inf, 3, 4, 5]
        let result = await processorList(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars

        XCTAssertTrue(resultArray[0].isInfinite && resultArray[0] < 0)
        XCTAssertTrue(resultArray[1].isInfinite && resultArray[1] < 0)
        XCTAssertEqual(resultArray[2], 3.0, accuracy: accuracy)
        XCTAssertEqual(resultArray[3], 4.0, accuracy: accuracy)
        XCTAssertEqual(resultArray[4], 5.0, accuracy: accuracy)
    }

    func testEmptyProcessorList() async throws {
        let processorList = LogitsProcessorList(processors: [])

        let inputIds = MLTensor(shape: [1, 1], scalars: [Int32(1)], scalarType: Int32.self)
        let scores = MLTensor(shape: [1, 3], scalars: [Float(1.0), Float(2.0), Float(3.0)], scalarType: Float.self)

        let result = await processorList(inputIds, scores)
        let resultArray = await result.shapedArray(of: Float.self).scalars
        let expectedArray = await scores.shapedArray(of: Float.self).scalars

        // Should be unchanged
        XCTAssertEqual(resultArray, expectedArray)
    }
}

// MARK: - Test Helpers

@available(macOS 15.0, iOS 18.0, *)
func assertMLTensorEqual(
    _ tensor: MLTensor,
    expected: [Float],
    accuracy: Float,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    let actual = await tensor.shapedArray(of: Float.self).scalars
    XCTAssertEqual(actual.count, expected.count, "Tensor size mismatch", file: file, line: line)
    for (a, e) in zip(actual, expected) {
        XCTAssertEqual(a, e, accuracy: accuracy, file: file, line: line)
    }
}
