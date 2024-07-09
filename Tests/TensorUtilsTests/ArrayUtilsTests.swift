import XCTest
@testable import TensorUtils

class ArrayUtilsTests: XCTestCase {

    func testPaddedArrayWhenNeedPadding() {
        let array = [1, 2, 3, 4]
        let paddedArray = array.padded(length: 7)
        XCTAssertEqual(paddedArray, [1, 2, 3, 4, 0, 0, 0])
    }

    func testNoPaddingForTheSamePaddingLength() {
        let array = [1, 2, 3, 4]
        let paddedArray = array.padded(length: 4)
        XCTAssertEqual(paddedArray, [1, 2, 3, 4])
    }

    func testNoPaddingForShorterPaddingLength() {
        let array = [1, 2, 3, 4]
        let paddedArray = array.padded(length: 2)
        XCTAssertEqual(paddedArray, [1, 2, 3, 4])
    }
}
