//
//  TensorUtilsTests.swift
//
//  Created by Jan Krukowski on 25/11/2023.
//

import CoreML
@testable import TensorUtils
import Testing

@Suite
final class TensorUtilsTests {
    private let accuracy: Float = 0.00001

    @Test
    func testCumsum() {
        #expect(Math.cumsum([]).isEmpty)
        #expect(Math.cumsum([1]) == [1])
        #expect(Math.cumsum([1, 2, 3, 4]) == [1, 3, 6, 10])
    }

    @Test
    func testArgMax() throws {
        let result1 = Math.argmax([3.0, 4.0, 1.0, 2.0] as [Float], count: 4)
        #expect(result1.0 == 1)
        #expect(result1.1 == 4.0)

        let result2 = Math.argmax32([3.0, 4.0, 1.0, 2.0], count: 4)
        #expect(result2.0 == 1)
        #expect(result2.1 == 4.0)

        let result3 = Math.argmax([3.0, 4.0, 1.0, 2.0] as [Double], count: 4)
        #expect(result3.0 == 1)
        #expect(result3.1 == 4.0)

        let result4 = try Math.argmax32(MLMultiArray([3.0, 4.0, 1.0, 2.0] as [Float]))
        #expect(result4.0 == 1)
        #expect(result4.1 == 4.0)

        let result5 = try Math.argmax(MLMultiArray([3.0, 4.0, 1.0, 2.0] as [Double]))
        #expect(result5.0 == 1)
        #expect(result5.1 == 4.0)

        let result6 = Math.argmax(MLShapedArray(scalars: [3.0, 4.0, 1.0, 2.0] as [Float], shape: [4]))
        #expect(result6.0 == 1)
        #expect(result6.1 == 4.0)
    }

    @Test
    func testSoftmax() {
        #expect(Math.softmax([]) == [])

        let result1 = Math.softmax([3.0, 4.0, 1.0, 2.0])
//        #expect(result1 == [0.23688284, 0.6439143, 0.032058604, 0.08714432], accuracy: accuracy)
//        #expect(result1.reduce(0, +) == 1.0, accuracy: accuracy)
    }
}
