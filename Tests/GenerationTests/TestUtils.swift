import Foundation

/// Check if two floating-point arrays are equal within a given accuracy
func isClose<T: FloatingPoint>(_ lhs: [T], _ rhs: [T], accuracy: T) -> Bool {
    guard lhs.count == rhs.count else { return false }
    return zip(lhs, rhs).allSatisfy { abs($0.0 - $0.1) <= accuracy }
}
