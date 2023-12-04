//
//  TensorUtilsTests.swift
//
//  Created by Jan Krukowski on 25/11/2023.
//

import XCTest
import CoreML
@testable import TensorUtils

class TensorUtilsTests: XCTestCase {
    private let accuracy: Float = 0.00001

    func testTopK() {
        let result1 = Math.topK(arr: [], k: 0)

        XCTAssertEqual(result1.indexes, [])
        XCTAssertEqual(result1.probs, [])
        
        let result2 = Math.topK(arr: [], k: 3)

        XCTAssertEqual(result2.indexes, [])
        XCTAssertEqual(result2.probs, [])

        let result3 = Math.topK(arr: [2.0, 1.0], k: 3)

        XCTAssertEqual(result3.indexes, [0, 1])
        XCTAssertEqual(result3.probs, [0.7310586, 0.26894143], accuracy: accuracy)
        XCTAssertEqual(result3.probs.reduce(0, +), 1.0, accuracy: accuracy)

        let result4 = Math.topK(arr: [2.0, 1.0, 3.0], k: 3)

        XCTAssertEqual(result4.indexes, [2, 0, 1])
        XCTAssertEqual(result4.probs, [0.6652409, 0.24472845, 0.090030566], accuracy: accuracy)
        XCTAssertEqual(result4.probs.reduce(0, +), 1.0, accuracy: accuracy)

        let result5 = Math.topK(arr: [2.0, 1.0, 3.0, -1.0, 123.0, 0.0], k: 4)

        XCTAssertEqual(result5.indexes, [4, 2, 0, 1])
        XCTAssertEqual(result5.probs, [1.0, 0.0, 0.0, 0.0], accuracy: accuracy)
        XCTAssertEqual(result5.probs.reduce(0, +), 1.0, accuracy: accuracy)
    }

    func testTopP() {
        let result1 = Math.topP(arr: [], p: 0.99)
        XCTAssertTrue(result1.indexes.isEmpty)
        XCTAssertTrue(result1.probs.isEmpty)

        let result2 = Math.topP(arr: (0 ..< 10).map { Float($0) }, p: 0.99)
        XCTAssertEqual(result2.indexes, [9, 8, 7, 6, 5])
        XCTAssertEqual(result2.probs, [0.63640857, 0.23412164, 0.08612853, 0.031684916, 0.011656229], accuracy: accuracy)
        XCTAssertEqual(result2.probs.reduce(0, +), 1.0, accuracy: accuracy)

        let result3 = Math.topP(arr: (0 ..< 10).map { Float($0) }, p: 0.95)
        XCTAssertEqual(result3.indexes, [9, 8, 7])
        XCTAssertEqual(result3.probs, [0.6652409, 0.24472845, 0.090030566], accuracy: accuracy)
        XCTAssertEqual(result3.probs.reduce(0, +), 1.0, accuracy: accuracy)

        let result4 = Math.topP(arr: (0 ..< 10).map { Float($0) }, p: 0.6321493)
        XCTAssertEqual(result4.indexes, [9, 8])
        XCTAssertEqual(result4.probs, [0.7310586, 0.26894143], accuracy: accuracy)
        XCTAssertEqual(result4.probs.reduce(0, +), 1.0, accuracy: accuracy)
    }

    func testCumsum() {
        XCTAssertTrue(Math.cumsum([]).isEmpty)
        XCTAssertEqual(Math.cumsum([1]), [1])
        XCTAssertEqual(Math.cumsum([1, 2, 3, 4]), [1, 3, 6, 10])
    }

    func testArgMax() throws {
        let result1 = Math.argmax([3.0, 4.0, 1.0, 2.0] as [Float], count: 4)
        
        XCTAssertEqual(result1.0, 1)
        XCTAssertEqual(result1.1, 4.0)

        let result2 = Math.argmax32([3.0, 4.0, 1.0, 2.0], count: 4)
        
        XCTAssertEqual(result2.0, 1)
        XCTAssertEqual(result2.1, 4.0)

        let result3 = Math.argmax([3.0, 4.0, 1.0, 2.0] as [Double], count: 4)
        
        XCTAssertEqual(result3.0, 1)
        XCTAssertEqual(result3.1, 4.0)

        let result4 = Math.argmax32(try MLMultiArray([3.0, 4.0, 1.0, 2.0] as [Float]))
        XCTAssertEqual(result4.0, 1)
        XCTAssertEqual(result4.1, 4.0)

        let result5 = Math.argmax(try MLMultiArray([3.0, 4.0, 1.0, 2.0] as [Double]))
        XCTAssertEqual(result5.0, 1)
        XCTAssertEqual(result5.1, 4.0)

        let result6 = Math.argmax(MLShapedArray(scalars: [3.0, 4.0, 1.0, 2.0] as [Float], shape: [4]))
        XCTAssertEqual(result6.0, 1)
        XCTAssertEqual(result6.1, 4.0)
    }

    func testSoftmax() {
        XCTAssertEqual(Math.softmax([]),  [])
        
        let result1 = Math.softmax([3.0, 4.0, 1.0, 2.0])
        XCTAssertEqual(result1, [0.23688284, 0.6439143, 0.032058604, 0.08714432], accuracy: accuracy)
        XCTAssertEqual(result1.reduce(0, +), 1.0, accuracy: accuracy)
    }
}

func XCTAssertEqual<T: FloatingPoint>(
    _ expression1: @autoclosure () throws -> [T],
    _ expression2: @autoclosure () throws -> [T],
    accuracy: T,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) {
    do {
        let lhsEvaluated = try expression1()
        let rhsEvaluated = try expression2()
        XCTAssertEqual(lhsEvaluated.count, rhsEvaluated.count, file: file, line: line)
        for (lhs, rhs) in zip(lhsEvaluated, rhsEvaluated) {
            XCTAssertEqual(lhs, rhs, accuracy: accuracy, file: file, line: line)
        }
    } catch {
        XCTFail("Unexpected error: \(error)", file: file, line: line)
    }
}
