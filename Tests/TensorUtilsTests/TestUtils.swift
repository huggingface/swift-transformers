import Foundation
import Testing

func expectEqual<T: FloatingPoint>(
    _ expression1: @autoclosure () throws -> [T],
    _ expression2: @autoclosure () throws -> [T],
    accuracy: T,
    _ message: @autoclosure () -> String = "",
    sourceLocation: SourceLocation = #_sourceLocation
) {
    do {
        let lhsEvaluated = try expression1()
        let rhsEvaluated = try expression2()
        #expect(lhsEvaluated.count == rhsEvaluated.count, sourceLocation: sourceLocation)
        for (lhs, rhs) in zip(lhsEvaluated, rhsEvaluated) {
            #expect(abs(lhs - rhs) <= accuracy, sourceLocation: sourceLocation)
        }
    } catch {
        Issue.record("Unexpected error: \(error)", sourceLocation: sourceLocation)
    }
}
