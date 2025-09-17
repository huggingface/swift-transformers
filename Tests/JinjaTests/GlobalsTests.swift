import Foundation
import Testing

@testable import Jinja

@Suite("Globals Tests")
struct GlobalsTests {
    let env = Environment()

    @Test("raise_exception without arguments")
    func raiseException() throws {
        #expect(throws: TemplateException.self) {
            try Globals.raiseException([], [:], env)
        }
    }

    @Test("raise_exception with custom message")
    func raiseExceptionWithMessage() throws {
        do {
            try Globals.raiseException(["Template error: invalid input"], [:], env)
        } catch let error as TemplateException {
            #expect(error.message == "Template error: invalid input")
        }
    }

    @Test("strftime_now with basic format")
    func strftimeNowBasic() throws {
        let result = try Globals.strftimeNow([.string("%Y-%m-%d")], [:], env)
        guard case let .string(dateString) = result else {
            #expect(Bool(false), "Expected string result")
            return
        }

        // Verify the result matches the expected format (YYYY-MM-DD)
        let regex = try NSRegularExpression(pattern: "^\\d{4}-\\d{2}-\\d{2}$")
        let range = NSRange(location: 0, length: dateString.utf16.count)
        let matches = regex.firstMatch(in: dateString, range: range)
        #expect(matches != nil, "Date string should match YYYY-MM-DD format")
    }

    @Test("strftime_now with time format")
    func strftimeNowTime() throws {
        let result = try Globals.strftimeNow([.string("%H:%M:%S")], [:], env)
        guard case let .string(timeString) = result else {
            #expect(Bool(false), "Expected string result")
            return
        }

        // Verify the result matches the expected format (HH:MM:SS)
        let regex = try NSRegularExpression(pattern: "^\\d{2}:\\d{2}:\\d{2}$")
        let range = NSRange(location: 0, length: timeString.utf16.count)
        let matches = regex.firstMatch(in: timeString, range: range)
        #expect(matches != nil, "Time string should match HH:MM:SS format")
    }

    @Test("strftime_now with weekday format")
    func strftimeNowWeekday() throws {
        let result = try Globals.strftimeNow([.string("%A")], [:], env)
        guard case let .string(weekdayString) = result else {
            #expect(Bool(false), "Expected string result")
            return
        }

        // Verify the result is a valid weekday name
        let weekdays = [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        ]
        #expect(weekdays.contains(weekdayString), "Result should be a valid weekday name")
    }

    @Test("strftime_now with complex format")
    func strftimeNowComplex() throws {
        let result = try Globals.strftimeNow([.string("%A, %B %d, %Y at %I:%M %p")], [:], env)
        guard case let .string(dateString) = result else {
            #expect(Bool(false), "Expected string result")
            return
        }

        // Should contain expected components
        #expect(dateString.contains(", "), "Should contain day/month separator")
        #expect(dateString.contains(" at "), "Should contain date/time separator")
        #expect(dateString.contains("AM") || dateString.contains("PM"), "Should contain AM/PM")

        // Check basic structure - should look like "Monday, September 15, 2025 at 01:30 PM"
        let regex = try NSRegularExpression(
            pattern: "\\w+, \\w+ \\d{1,2}, \\d{4} at \\d{1,2}:\\d{2} (AM|PM)")
        let range = NSRange(location: 0, length: dateString.utf16.count)
        let matches = regex.firstMatch(in: dateString, range: range)
        #expect(
            matches != nil,
            "Complex date format should match expected pattern, got: \(dateString)"
        )
    }

    @Test("strftime_now with literal percent")
    func strftimeNowLiteralPercent() throws {
        let result = try Globals.strftimeNow([.string("%%Y")], [:], env)
        guard case let .string(resultString) = result else {
            #expect(Bool(false), "Expected string result")
            return
        }

        #expect(resultString == "%Y", "Should handle literal percent correctly")
    }

    @Test("strftime_now with no arguments")
    func strftimeNowNoArguments() throws {
        #expect(throws: JinjaError.self) {
            try Globals.strftimeNow([], [:], env)
        }
    }

    @Test("strftime_now with too many arguments")
    func strftimeNowTooManyArguments() throws {
        #expect(throws: JinjaError.self) {
            try Globals.strftimeNow([.string("%Y"), .string("%m")], [:], env)
        }
    }

    @Test("strftime_now with non-string argument")
    func strftimeNowNonStringArgument() throws {
        #expect(throws: JinjaError.self) {
            try Globals.strftimeNow([.int(2024)], [:], env)
        }
    }
}
