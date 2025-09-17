//
//  MathTests.swift
//
//  Created by Jan Krukowski on 25/11/2023.
//

#if canImport(CoreML)
import CoreML
@testable import Generation
import Testing

@Suite("Math Tests")
struct MathTests {
    private let accuracy: Float = 0.00001

    @Test("Cumulative sum functionality")
    func cumsum() {
        #expect(Math.cumsum([]).isEmpty)
        #expect(Math.cumsum([1]) == [1])
        #expect(Math.cumsum([1, 2, 3, 4]) == [1, 3, 6, 10])
    }

    @Test("Argmax functionality")
    func argmax() throws {
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

    @Test("Softmax functionality")
    func softmax() {
        #expect(Math.softmax([]) == [])

        let result1 = Math.softmax([3.0, 4.0, 1.0, 2.0])
        #expect(isClose(result1, [0.23688284, 0.6439143, 0.032058604, 0.08714432], accuracy: accuracy))
        #expect(abs(result1.reduce(0, +) - 1.0) < accuracy)
    }
}
#endif // canImport(CoreML)
