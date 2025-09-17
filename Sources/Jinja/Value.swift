import Foundation
@_exported import OrderedCollections

/// Represents values in Jinja template expressions and variables.
///
/// Values are the runtime representation of data in Jinja templates,
/// supporting various types including primitives, collections, and callable objects.
public enum Value: Sendable {
    /// Null value representing absence of data.
    case null
    /// Undefined value for uninitialized variables.
    case undefined
    /// Boolean value (`true` or `false`).
    case boolean(Bool)
    /// Integer numeric value.
    case int(Int)
    /// Floating-point numeric value.
    case double(Double)
    /// String value containing text data.
    case string(String)
    /// Array containing ordered collection of values.
    case array([Value])
    /// Object containing key-value pairs with preserved insertion order.
    case object(OrderedDictionary<String, Value>)
    /// Function value that can be called with arguments.
    case function(@Sendable ([Value], [String: Value], Environment) throws -> Value)
    /// Macro value that can be invoked with arguments.
    case macro(Macro)

    /// Creates a Value from any Swift value.
    ///
    /// This initializer attempts to convert common Swift types to their Jinja equivalents.
    /// Supported types include strings, numbers, booleans, arrays, dictionaries, and macros.
    ///
    /// - Parameter value: The Swift value to convert
    /// - Throws: `JinjaError.runtime` if the value type cannot be converted
    public init(any value: Any?) throws {
        switch value {
        case nil:
            self = .null
        case let str as String:
            self = .string(str)
        case let int as Int:
            self = .int(int)
        case let double as Double:
            self = .double(double)
        case let float as Float:
            self = .double(Double(float))
        case let bool as Bool:
            self = .boolean(bool)
        case let array as [Any?]:
            let values = try array.map { try Value(any: $0) }
            self = .array(values)
        case let dict as [String: Any?]:
            var orderedDict = OrderedDictionary<String, Value>()
            for (key, value) in dict {
                orderedDict[key] = try Value(any: value)
            }
            self = .object(orderedDict)
        case let macro as Macro:
            self = .macro(macro)
        default:
            throw JinjaError.runtime(
                "Cannot convert value of type \(type(of: value)) to Jinja Value")
        }
    }

    // MARK: Convenience type checks

    /// Returns whether the value is `null`.
    public var isNull: Bool {
        self == .null
    }

    /// Returns whether the value is `undefined`.
    public var isUndefined: Bool {
        self == .undefined
    }

    /// Returns `true` if this value is a boolean.
    public var isBoolean: Bool {
        if case .boolean = self { return true }
        return false
    }

    /// Returns `true` if this value is an integer.
    public var isInt: Bool {
        if case .int = self { return true }
        return false
    }

    /// Returns `true` if this value is a floating-point number.
    public var isDouble: Bool {
        if case .double = self { return true }
        return false
    }

    /// Returns `true` if this value is a string.
    public var isString: Bool {
        if case .string = self { return true }
        return false
    }

    /// Returns `true` if this value is an array.
    public var isArray: Bool {
        if case .array = self { return true }
        return false
    }

    /// Returns `true` if this value is an object.
    public var isObject: Bool {
        if case .object = self { return true }
        return false
    }

    /// Returns `true` if this value is a function.
    public var isFunction: Bool {
        if case .function = self { return true }
        return false
    }

    /// Returns `true` if this value is a macro.
    public var isMacro: Bool {
        if case .macro = self { return true }
        return false
    }

    /// Returns `true` if this value can be iterated over (array, object, or string).
    public var isIterable: Bool {
        switch self {
        case .array, .object, .string: true
        default: false
        }
    }

    /// Returns `true` if this value is truthy in boolean context.
    public var isTruthy: Bool {
        switch self {
        case .null, .undefined: false
        case let .boolean(b): b
        case let .double(n): n != 0.0
        case let .int(i): i != 0
        case let .string(s): !s.isEmpty
        case let .array(a): !a.isEmpty
        case let .object(o): !o.isEmpty
        case .function: true
        case .macro: true
        }
    }

    // MARK: Operations

    /// Adds two values together.
    /// - Parameters:
    ///   - other: The value to add to the current value
    /// - Throws: `JinjaError.runtime` if the values cannot be added
    public func add(with other: Value) throws -> Value {
        switch (self, other) {
        case let (.int(a), .int(b)):
            return .int(a + b)
        case let (.double(a), .double(b)):
            return .double(a + b)
        case let (.int(a), .double(b)):
            return .double(Double(a) + b)
        case let (.double(a), .int(b)):
            return .double(a + Double(b))
        case let (.string(a), .string(b)):
            return .string(a + b)
        case let (.string(a), b):
            return .string(a + b.description)
        case let (a, .string(b)):
            return .string(a.description + b)
        case let (.array(a), .array(b)):
            return .array(a + b)
        default:
            throw JinjaError.runtime("Cannot add values of different types (\(self) and \(other))")
        }
    }

    /// Concatenates two values.
    /// - Parameters:
    ///   - other: The value to concatenate to the current value
    /// - Throws: `JinjaError.runtime` if the values cannot be concatenated
    public func concatenate(with other: Value) throws -> Value {
        switch (self, other) {
        case let (.string(a), .string(b)):
            return .string(a + b)
        case let (.string(a), b):
            return .string(a + b.description)
        case let (a, .string(b)):
            return .string(a.description + b)
        default:
            throw JinjaError.runtime(
                "Cannot concatenate values of different types (\(self) and \(other))")
        }
    }

    /// Subtracts another value from the current value.
    /// - Parameters:
    ///   - other: The value to subtract from the current value
    /// - Throws: `JinjaError.runtime` if the values cannot be subtracted
    public func subtract(by other: Value) throws -> Value {
        switch (self, other) {
        case let (.int(a), .int(b)):
            return .int(a - b)
        case let (.double(a), .double(b)):
            return .double(a - b)
        case let (.int(a), .double(b)):
            return .double(Double(a) - b)
        case let (.double(a), .int(b)):
            return .double(a - Double(b))
        default:
            throw JinjaError.runtime("Cannot subtract non-numeric values (\(self) and \(other))")
        }
    }

    /// Multiplies the current value by another value.
    /// - Parameters:
    ///   - other: The value to multiply the current value by
    /// - Throws: `JinjaError.runtime` if the values cannot be multiplied
    public func multiply(by other: Value) throws -> Value {
        switch (self, other) {
        case let (.int(a), .int(b)):
            return .int(a * b)
        case let (.double(a), .double(b)):
            return .double(a * b)
        case let (.int(a), .double(b)):
            return .double(Double(a) * b)
        case let (.double(a), .int(b)):
            return .double(a * Double(b))
        case let (.string(s), .int(n)):
            return .string(String(repeating: s, count: n))
        case let (.int(n), .string(s)):
            return .string(String(repeating: s, count: n))
        default:
            throw JinjaError.runtime("Cannot multiply values of these types (\(self) and \(other))")
        }
    }

    /// Divides the current value by another value.
    /// - Parameters:
    ///   - other: The value to divide the current value by
    /// - Throws: `JinjaError.runtime` if the values cannot be divided
    public func divide(by other: Value) throws -> Value {
        switch (self, other) {
        case let (.int(a), .int(b)):
            guard b != 0 else { throw JinjaError.runtime("Division by zero") }
            return .double(Double(a) / Double(b))
        case let (.double(a), .double(b)):
            guard b != 0 else { throw JinjaError.runtime("Division by zero") }
            return .double(a / b)
        case let (.int(a), .double(b)):
            guard b != 0 else { throw JinjaError.runtime("Division by zero") }
            return .double(Double(a) / b)
        case let (.double(a), .int(b)):
            guard b != 0 else { throw JinjaError.runtime("Division by zero") }
            return .double(a / Double(b))
        default:
            throw JinjaError.runtime("Cannot divide non-numeric values (\(self) and \(other))")
        }
    }

    /// Computes the modulo of the current value and another value.
    /// - Parameters:
    ///   - other: The value to compute the modulo of the current value by
    /// - Throws: `JinjaError.runtime` if the values cannot be moduloed
    public func modulo(by other: Value) throws -> Value {
        switch (self, other) {
        case let (.int(a), .int(b)):
            guard b != 0 else { throw JinjaError.runtime("Modulo by zero") }
            return .int(a % b)
        default:
            throw JinjaError.runtime("Modulo operation requires integers (\(self) and \(other))")
        }
    }

    /// Computes the floor division of the current value and another value.
    /// - Parameters:
    ///   - other: The value to compute the floor division of the current value by
    /// - Throws: `JinjaError.runtime` if the values cannot be floor divided
    public func floorDivide(by other: Value) throws -> Value {
        switch (self, other) {
        case let (.int(a), .int(b)):
            guard b != 0 else { throw JinjaError.runtime("Division by zero") }
            return .int(a / b) // Integer division in Swift already floors
        case let (.double(a), .double(b)):
            guard b != 0 else { throw JinjaError.runtime("Division by zero") }
            return .int(Int(floor(a / b)))
        case let (.int(a), .double(b)):
            guard b != 0 else { throw JinjaError.runtime("Division by zero") }
            return .int(Int(floor(Double(a) / b)))
        case let (.double(a), .int(b)):
            guard b != 0 else { throw JinjaError.runtime("Division by zero") }
            return .int(Int(floor(a / Double(b))))
        default:
            throw JinjaError.runtime(
                "Cannot floor divide non-numeric values (\(self) and \(other))")
        }
    }

    /// Raises the current value to the power of another value.
    /// - Parameters:
    ///   - other: The value to raise the current value to the power of
    /// - Throws: `JinjaError.runtime` if the values cannot be raised to a power
    public func power(by other: Value) throws -> Value {
        switch (self, other) {
        case let (.int(a), .int(b)):
            guard b >= 0 else {
                return .double(pow(Double(a), Double(b)))
            }
            return .int(Int(pow(Double(a), Double(b))))
        case let (.double(a), .double(b)):
            return .double(pow(a, b))
        case let (.int(a), .double(b)):
            return .double(pow(Double(a), b))
        case let (.double(a), .int(b)):
            return .double(pow(a, Double(b)))
        default:
            throw JinjaError.runtime(
                "Cannot raise non-numeric values to a power (\(self) and \(other))")
        }
    }

    /// Compares the current value to another value.
    /// - Parameters:
    ///   - other: The value to compare the current value to
    /// - Throws: `JinjaError.runtime` if the values cannot be compared
    public func compare(to other: Value) throws -> Int {
        switch (self, other) {
        case let (.int(a), .int(b)):
            return a < b ? -1 : a > b ? 1 : 0
        case let (.double(a), .double(b)):
            return a < b ? -1 : a > b ? 1 : 0
        case let (.int(a), .double(b)):
            let val = Double(a)
            return val < b ? -1 : val > b ? 1 : 0
        case let (.double(a), .int(b)):
            let val = Double(b)
            return a < val ? -1 : a > val ? 1 : 0
        case let (.string(a), .string(b)):
            return a < b ? -1 : a > b ? 1 : 0
        default:
            throw JinjaError.runtime(
                "Cannot compare values of different types (\(self) and \(other))")
        }
    }

    /// Checks if the current value is contained in another value.
    /// - Parameters:
    ///   - collection: The value to check if the current value is contained in
    /// - Throws: `JinjaError.runtime` if the values cannot be checked for containment
    public func isContained(in collection: Value) throws -> Bool {
        switch collection {
        case .undefined:
            return false
        case .null:
            return false
        case let .array(items):
            return items.contains { self == $0 }
        case let .string(str):
            guard case let .string(substr) = self else { return false }
            guard !substr.isEmpty else { return true } // '' in 'abc' -> true
            return str.contains(substr)
        case let .object(dict):
            guard case let .string(key) = self else { return false }
            return dict.keys.contains(key)
        default:
            throw JinjaError.runtime(
                "'in' operator requires iterable on right side (\(collection))")
        }
    }

    /// Checks if the current value is equivalent to another value.
    ///
    /// This is similar to `==`, but provides special handling for numeric types,
    /// allowing an `Int` and a `Double` to be considered equivalent if they represent
    /// the same number.
    ///
    /// - Parameters:
    ///   - other: The value to compare for equivalence.
    /// - Returns: `true` if the values are equivalent, otherwise `false`.
    public func isEquivalent(to other: Value) -> Bool {
        switch (self, other) {
        case let (.int(a), .int(b)):
            return a == b
        case let (.double(a), .double(b)):
            return a == b
        case let (.int(a), .double(b)):
            return Double(a) == b
        case let (.double(a), .int(b)):
            return a == Double(b)
        case let (.string(a), .string(b)):
            return a == b
        case let (.boolean(a), .boolean(b)):
            return a == b
        case (.null, .null):
            return true
        case (.undefined, .undefined):
            return true
        case let (.array(a), .array(b)):
            guard a.count == b.count else { return false }
            return zip(a, b).allSatisfy { $0.isEquivalent(to: $1) }
        case let (.object(a), .object(b)):
            guard a.count == b.count else { return false }
            for ((keyA, valueA), (keyB, valueB)) in zip(a, b) {
                if keyA != keyB || !valueA.isEquivalent(to: valueB) {
                    return false
                }
            }
            return true
        case let (.macro(a), .macro(b)):
            return a == b
        default:
            // .function and mixed types
            return false
        }
    }
}

// MARK: - CustomStringConvertible

extension Value: CustomStringConvertible {
    /// String representation of the value for template output.
    public var description: String {
        switch self {
        case let .string(s): return s
        case let .double(n): return String(n)
        case let .int(i): return String(i)
        case let .boolean(b): return String(b)
        case .null: return ""
        case .undefined: return ""
        case let .array(a):
            // Python-style representation of strings in the array
            let elements = a.map { element -> String in
                if case let .string(s) = element {
                    return "'\(s)'"
                } else {
                    return element.description
                }
            }
            return "[\(elements.joined(separator: ", "))]"
        case let .object(o):
            return "{\(o.map { "\($0.key): \($0.value.description)" }.joined(separator: ", "))}"
        case .function: return "[Function]"
        case let .macro(m): return "[Macro \(m.name)]"
        }
    }
}

// MARK: - Equatable

extension Value: Equatable {
    /// Compares two values for equality.
    public static func == (lhs: Value, rhs: Value) -> Bool {
        switch (lhs, rhs) {
        case let (.string(lhs), .string(rhs)): lhs == rhs
        case let (.double(lhs), .double(rhs)): lhs == rhs
        case let (.int(lhs), .int(rhs)): lhs == rhs
        case let (.boolean(lhs), .boolean(rhs)): lhs == rhs
        case (.null, .null): true
        case (.undefined, .undefined): true
        case let (.array(lhs), .array(rhs)): lhs == rhs
        case let (.object(lhs), .object(rhs)): lhs == rhs
        case (.function, .function): false
        case let (.macro(lhs), .macro(rhs)): lhs == rhs
        default: false
        }
    }
}

// MARK: - Hashable

extension Value: Hashable {
    /// Hashes the value into the given hasher.
    public func hash(into hasher: inout Hasher) {
        switch self {
        case let .string(value): hasher.combine(value)
        case let .double(value): hasher.combine(value)
        case let .int(value): hasher.combine(value)
        case let .boolean(value): hasher.combine(value)
        case .null: hasher.combine(0)
        case .undefined: hasher.combine(0)
        case let .array(value): hasher.combine(value)
        case let .object(value): hasher.combine(value)
        case .function: hasher.combine(0)
        case let .macro(m): hasher.combine(m)
        }
    }
}

// MARK: - Encodable

extension Value: Encodable {
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()

        switch self {
        case let .string(value):
            try container.encode(value)
        case let .double(value):
            try container.encode(value)
        case let .int(value):
            try container.encode(value)
        case let .boolean(value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        case .undefined:
            try container.encodeNil()
        case let .array(value):
            try container.encode(value)
        case let .object(value):
            var dictionary: [String: Value] = [:]
            for (key, value) in value {
                dictionary[key] = value
            }
            try container.encode(dictionary)
        case .function:
            throw EncodingError.invalidValue(
                self,
                EncodingError.Context(
                    codingPath: encoder.codingPath,
                    debugDescription: "Cannot encode function values"
                )
            )
        case let .macro(m):
            try container.encode(m)
        }
    }
}

// MARK: - Decodable

extension Value: Decodable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let string = try? container.decode(String.self) {
            self = .string(string)
        } else if let integer = try? container.decode(Int.self) {
            self = .int(integer)
        } else if let number = try? container.decode(Double.self) {
            self = .double(number)
        } else if let boolean = try? container.decode(Bool.self) {
            self = .boolean(boolean)
        } else if let value = try? container.decode([Value].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: Value].self) {
            var orderedDictionary: OrderedDictionary<String, Value> = [:]
            for key in value.keys.sorted() {
                orderedDictionary[key] = value[key]
            }
            self = .object(orderedDictionary)
        } else if let macro = try? container.decode(Macro.self) {
            self = .macro(macro)
        } else {
            throw DecodingError.typeMismatch(
                Value.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Cannot decode Value from any supported container type"
                )
            )
        }
    }
}

// MARK: - ExpressibleByNilLiteral

extension Value: ExpressibleByNilLiteral {
    public init(nilLiteral: ()) {
        self = .null
    }
}

// MARK: - ExpressibleByBooleanLiteral

extension Value: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) {
        self = .boolean(value)
    }
}

// MARK: - ExpressibleByIntegerLiteral

extension Value: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self = .int(value)
    }
}

// MARK: - ExpressibleByFloatLiteral

extension Value: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self = .double(value)
    }
}

// MARK: - ExpressibleByStringLiteral

extension Value: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) {
        self = .string(value)
    }
}

// MARK: - ExpressibleByArrayLiteral

extension Value: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: Value...) {
        self = .array(elements)
    }
}

// MARK: - ExpressibleByDictionaryLiteral

extension Value: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, Value)...) {
        var dict = OrderedDictionary<String, Value>()
        for (key, value) in elements {
            dict[key] = value
        }
        self = .object(dict)
    }
}
