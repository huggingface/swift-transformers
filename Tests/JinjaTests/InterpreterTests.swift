import Foundation
import Testing

@testable import Jinja

@Suite("Interpreter")
struct InterpreterTests {
    @Suite("Operators")
    struct OperatorTests {
        @Test("Floor division with integers")
        func floorDivisionIntegers() throws {
            let result = try Interpreter.evaluateBinaryValues(.floorDivide, .int(20), .int(7))
            #expect(result == .int(2))
        }

        @Test("Floor division with mixed types")
        func floorDivisionMixed() throws {
            let result = try Interpreter.evaluateBinaryValues(.floorDivide, .double(20.5), .int(7))
            #expect(result == .int(2))
        }

        @Test("Floor division by zero throws error")
        func floorDivisionByZero() throws {
            #expect(throws: JinjaError.self) {
                try Interpreter.evaluateBinaryValues(.floorDivide, .int(10), .int(0))
            }
        }

        @Test("Exponentiation with integers")
        func exponentiationIntegers() throws {
            let result = try Interpreter.evaluateBinaryValues(.power, .int(2), .int(3))
            #expect(result == .int(8))
        }

        @Test("Exponentiation with mixed types")
        func exponentiationMixed() throws {
            let result = try Interpreter.evaluateBinaryValues(.power, .int(2), .double(3.0))
            #expect(result == .double(8.0))
        }

        @Test("Exponentiation with negative exponent")
        func exponentiationNegative() throws {
            let result = try Interpreter.evaluateBinaryValues(.power, .int(2), .int(-2))
            #expect(result == .double(0.25))
        }
    }
}
