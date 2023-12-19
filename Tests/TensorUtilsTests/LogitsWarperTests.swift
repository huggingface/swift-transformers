//
//  LogitsWarperTests.swift
//
//  Created by Jan Krukowski on 09/12/2023.
//

import XCTest
import CoreML
@testable import TensorUtils

final class LogitsWarperTests: XCTestCase {
    private let accuracy: Float = 0.00001

    func testTemperatureLogitsWarper() {
        let result1 = TemperatureLogitsWarper(temperature: 0.0)([])
        XCTAssertTrue(result1.indexes.isEmpty)
        XCTAssertTrue(result1.logits.isEmpty)

        let result2 = TemperatureLogitsWarper(temperature: 1.0)([])
        XCTAssertTrue(result2.indexes.isEmpty)
        XCTAssertTrue(result2.logits.isEmpty)

        let result3 = TemperatureLogitsWarper(temperature: 1.0)([2.0, 1.0])
        XCTAssertEqual(result3.indexes, [0, 1])
        XCTAssertEqual(result3.logits, [2.0, 1.0], accuracy: accuracy)

        let result4 = TemperatureLogitsWarper(temperature: 2.0)([2.0, 1.0])
        XCTAssertEqual(result4.indexes, [0, 1])
        XCTAssertEqual(result4.logits, [1.0, 0.5], accuracy: accuracy)

        let result5 = TemperatureLogitsWarper(temperature: 0.5)([2.0, 1.0])
        XCTAssertEqual(result5.indexes, [0, 1])
        XCTAssertEqual(result5.logits, [4.0, 2.0], accuracy: accuracy)
    }

    func testTopKLogitsWarper() {
        let result1 = TopKLogitsWarper(k: 0)([])
        XCTAssertTrue(result1.indexes.isEmpty)
        XCTAssertTrue(result1.logits.isEmpty)

        let result2 = TopKLogitsWarper(k: 3)([])
        XCTAssertTrue(result2.indexes.isEmpty)
        XCTAssertTrue(result2.logits.isEmpty)

        let result3 = TopKLogitsWarper(k: 3)([2.0, 1.0])
        XCTAssertEqual(result3.indexes, [0, 1])
        XCTAssertEqual(result3.logits, [2.0, 1.0], accuracy: accuracy)

        let result4 = TopKLogitsWarper(k: 3)([2.0, 1.0, 3.0])
        XCTAssertEqual(result4.indexes, [2, 0, 1])
        XCTAssertEqual(result4.logits, [3.0, 2.0, 1.0], accuracy: accuracy)

        let result5 = TopKLogitsWarper(k: 4)([2.0, 1.0, 3.0, -1.0, 123.0, 0.0])
        XCTAssertEqual(result5.indexes, [4, 2, 0, 1])
        XCTAssertEqual(result5.logits, [123.0, 3.0, 2.0, 1.0], accuracy: accuracy)
    }

    func testTopPLogitsWarper() {
        let result1 = TopPLogitsWarper(p: 0.99)([])
        XCTAssertTrue(result1.indexes.isEmpty)
        XCTAssertTrue(result1.logits.isEmpty)

        let result2 = TopPLogitsWarper(p: 0.99)((0 ..< 10).map { Float($0) })
        XCTAssertEqual(result2.indexes, [9, 8, 7, 6, 5])
        XCTAssertEqual(result2.logits, [9.0, 8.0, 7.0, 6.0, 5.0], accuracy: accuracy)

        let result3 = TopPLogitsWarper(p: 0.95)((0 ..< 10).map { Float($0) })
        XCTAssertEqual(result3.indexes, [9, 8, 7])
        XCTAssertEqual(result3.logits, [9.0, 8.0, 7.0], accuracy: accuracy)

        let result4 = TopPLogitsWarper(p: 0.6321493)((0 ..< 10).map { Float($0) })
        XCTAssertEqual(result4.indexes, [9, 8])
        XCTAssertEqual(result4.logits, [9.0, 8.0], accuracy: accuracy)
    }

    func testLogitsProcessor() {
        let processor1 = LogitsProcessor(logitsWarpers: [])
        let result1 = processor1([])
        XCTAssertTrue(result1.indexes.isEmpty)
        XCTAssertTrue(result1.logits.isEmpty)

        let processor2 = LogitsProcessor(logitsWarpers: [])
        let result2 = processor2([2.0, 1.0])
        XCTAssertEqual(result2.indexes, [0, 1])
        XCTAssertEqual(result2.logits, [2.0, 1.0], accuracy: accuracy)

        let processor3 = LogitsProcessor(
            logitsWarpers: [TopKLogitsWarper(k: 3)]
        )
        let result3 = processor3([2.0, 1.0, 3.0, -5.0])
        XCTAssertEqual(result3.indexes, [2, 0, 1])
        XCTAssertEqual(result3.logits, [3.0, 2.0, 1.0], accuracy: accuracy)

        let processor4 = LogitsProcessor(
            logitsWarpers: [TopKLogitsWarper(k: 3), TopPLogitsWarper(p: 0.99)]
        )
        let result4 = processor4([2.0, 1.0, 3.0, -5.0, -23.0, 12.5])
        XCTAssertEqual(result4.indexes, [0])
        XCTAssertEqual(result4.logits, [12.5], accuracy: accuracy)
    }
}
