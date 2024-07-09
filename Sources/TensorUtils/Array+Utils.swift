import Foundation


extension Array where Element: Numeric {
    func padded(length maxLength: Int) -> Array<Element> {
        self + Array(repeating: 0, count: Swift.max(maxLength - count, 0))
    }
}
