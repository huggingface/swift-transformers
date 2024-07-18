import XCTest
import CoreML
@testable import TensorUtils

final class MLMultiArrayUtilsTests: XCTestCase {

    func testAdditionOfSameShape() throws {
        let array: [Float32] = [
            01, 02, 03, 04, 05, 06, 07, 08, 09, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27 ,28 ,29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        ]
        let stride = MemoryLayout<Float32>.stride
        let allocated = UnsafeMutableRawBufferPointer.allocate(byteCount: array.count * stride, alignment: MemoryLayout<Float32>.alignment)
        defer { allocated.deallocate() }
        _ = array.withUnsafeBufferPointer { ptr in
            memcpy(allocated.baseAddress!, ptr.baseAddress!, array.count * stride)
        }
        let multiArray = try MLMultiArray(dataPointer: allocated.baseAddress!, shape: [4, 10], dataType: .float32, strides: [10, 1])
        let output = multiArray + multiArray
        XCTAssertEqual(output.count, array.count)
        XCTAssertEqual(output.count, multiArray.count)

        for index in 0..<output.count {
            let expectedValue = array[index] + array[index]
            XCTAssertEqual(output[index].floatValue, expectedValue)
        }
    }

    func testAdditionRowBroadcasting() throws {
        let array: [Float32] = [
            01, 02, 03, 04, 05, 06, 07, 08, 09, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27 ,28 ,29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        ]
        let stride = MemoryLayout<Float32>.stride
        let allocA = UnsafeMutableRawBufferPointer.allocate(byteCount: array.count * stride, alignment: MemoryLayout<Float32>.alignment)
        defer { allocA.deallocate() }
        let allocB = UnsafeMutableRawBufferPointer.allocate(byteCount: 10 * stride, alignment: MemoryLayout<Float32>.alignment)
        defer { allocB.deallocate() }

        _ = array.withUnsafeBufferPointer { ptr in
            memcpy(allocA.baseAddress!, ptr.baseAddress!, array.count * stride)
        }
        _ = Array<Float32>(repeating: 10, count: 10).withUnsafeBufferPointer { ptr in
            memcpy(allocB.baseAddress!, ptr.baseAddress!, 10 * stride)
        }

        let A = try MLMultiArray(dataPointer: allocA.baseAddress!, shape: [4, 10], dataType: .float32, strides: [10, 1])
        XCTAssertEqual(A.count, 40)
        let B = try MLMultiArray(dataPointer: allocB.baseAddress!, shape: [1, 10], dataType: .float32, strides: [10, 1])
        XCTAssertEqual(B.count, 10)

        _ = A + B
        _ = A + B + B
        _ = A + B + B
        let expectedArray: [Float32] = [
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27 ,28 ,29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        ]
        let output = A + B
        XCTAssertEqual(output.floats, expectedArray)
    }

    func testAdditionRowReverseOrder() throws {
        let array: [Float32] = [
            01, 02, 03, 04, 05, 06, 07, 08, 09, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27 ,28 ,29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        ]
        let stride = MemoryLayout<Float32>.stride
        let allocA = UnsafeMutableRawBufferPointer.allocate(byteCount: array.count * stride, alignment: MemoryLayout<Float32>.alignment)
        defer { allocA.deallocate() }
        let allocB = UnsafeMutableRawBufferPointer.allocate(byteCount: 10 * stride, alignment: MemoryLayout<Float32>.alignment)
        defer { allocB.deallocate() }

        _ = array.withUnsafeBufferPointer { ptr in
            memcpy(allocA.baseAddress!, ptr.baseAddress!, array.count * stride)
        }
        _ = Array<Float32>(repeating: 10, count: 10).withUnsafeBufferPointer { ptr in
            memcpy(allocB.baseAddress!, ptr.baseAddress!, 10 * stride)
        }

        let A = try MLMultiArray(dataPointer: allocA.baseAddress!, shape: [4, 10], dataType: .float32, strides: [10, 1])
        XCTAssertEqual(A.count, 40)
        let B = try MLMultiArray(dataPointer: allocB.baseAddress!, shape: [1, 10], dataType: .float32, strides: [10, 1])
        XCTAssertEqual(B.count, 10)
        XCTAssertEqual(B + A, A + B)
        _ = A + B
        _ = A + B + B
        _ = A + B + B
        let expectedArray: [Float32] = [
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27 ,28 ,29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        ]
        let output = B + A
        XCTAssertEqual(output.floats, expectedArray)
    }
}
