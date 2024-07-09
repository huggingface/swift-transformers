import XCTest
import Accelerate
@testable import TensorUtils

class BNNSUtilsTests: XCTestCase {

    func testMakeMultiArrayFromDescriptor() throws {
        let rowCount = 4
        let dimSize = 6
        let dictData: [Float32] = [
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24,
        ]
        let dict = BNNSNDArrayDescriptor.allocate(initializingFrom: dictData, shape: .matrixColumnMajor(dimSize, rowCount))
        let shape: [NSNumber] = [
            NSNumber(value: rowCount),
            NSNumber(value: dimSize),
        ]
        let multiArray = try dict.makeMultiArray(of: Float32.self, shape: shape)
        XCTAssertEqual(multiArray.toArray(), dictData)
        XCTAssertEqual(multiArray.floats!, dictData)
    }

}
