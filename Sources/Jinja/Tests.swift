/// Built-in tests for Jinja template rendering.
///
/// Tests are used with the `is` operator to check conditions about values.
/// All test functions return a Boolean result and follow the same signature pattern,
/// accepting an array of values, optional keyword arguments, and an environment.
public enum Tests {
    // MARK: - Basic Tests

    /// Tests if the input is defined (not undefined).
    @Sendable public static func defined(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        return !input.isUndefined
    }

    /// Tests if the input is undefined.
    @Sendable public static func undefined(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return true }
        return input.isUndefined
    }

    /// Tests if the input is none/null.
    @Sendable public static func none(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        return input.isNull
    }

    /// Tests if the input is a string.
    @Sendable public static func string(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        return input.isString
    }

    /// Tests if the input is a number.
    @Sendable public static func number(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        return input.isInt || input.isDouble
    }

    /// Tests if the input is a Boolean.
    @Sendable public static func boolean(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        return input.isBoolean
    }

    /// Tests if the input is iterable.
    @Sendable public static func iterable(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        return input.isIterable
    }

    // MARK: - Numeric Tests

    /// Tests if a number is even.
    @Sendable public static func even(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        switch input {
        case let .int(num):
            return num % 2 == 0
        case let .double(num):
            return Int(num) % 2 == 0
        default:
            return false
        }
    }

    /// Tests if a number is odd.
    @Sendable public static func odd(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        switch input {
        case let .int(num):
            return num % 2 != 0
        case let .double(num):
            return Int(num) % 2 != 0
        default:
            return false
        }
    }

    /// Tests if a number is divisible by another number.
    @Sendable public static func divisibleby(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value", "divisor"]
        )

        guard let value = arguments["value"], let divisor = arguments["divisor"] else {
            return false
        }
        switch (value, divisor) {
        case let (.int(a), .int(b)):
            return b != 0 && a % b == 0
        case let (.double(a), .double(b)):
            return b != 0.0 && Int(a) % Int(b) == 0
        default:
            return false
        }
    }

    // MARK: - Comparison Tests

    /// Tests if a == b.
    @Sendable public static func eq(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["a", "b"]
        )

        guard let a = arguments["a"], let b = arguments["b"] else { return false }
        return a.isEquivalent(to: b)
    }

    /// Tests if a != b.
    @Sendable public static func ne(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["a", "b"]
        )

        guard let a = arguments["a"], let b = arguments["b"] else { return false }
        return !a.isEquivalent(to: b)
    }

    /// Tests if a > b.
    @Sendable public static func gt(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["a", "b"]
        )

        guard let a = arguments["a"], let b = arguments["b"] else { return false }
        do {
            return try a.compare(to: b) > 0
        } catch {
            return false
        }
    }

    /// Tests if a >= b.
    @Sendable public static func ge(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["a", "b"]
        )

        guard let a = arguments["a"], let b = arguments["b"] else { return false }
        do {
            return try a.compare(to: b) >= 0
        } catch {
            return false
        }
    }

    /// Tests if a < b.
    @Sendable public static func lt(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["a", "b"]
        )

        guard let a = arguments["a"], let b = arguments["b"] else { return false }
        do {
            return try a.compare(to: b) < 0
        } catch {
            return false
        }
    }

    /// Tests if a <= b.
    @Sendable public static func le(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["a", "b"]
        )

        guard let a = arguments["a"], let b = arguments["b"] else { return false }
        do {
            return try a.compare(to: b) <= 0
        } catch {
            return false
        }
    }

    /// Tests if the input is a mapping (dictionary/object).
    @Sendable public static func mapping(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        return input.isObject
    }

    /// Tests if the input is callable (function or macro).
    @Sendable public static func callable(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        return input.isFunction || input.isMacro
    }

    /// Tests if the input is an integer.
    @Sendable public static func integer(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        return input.isInt
    }

    /// Tests if a string is all lowercase.
    @Sendable public static func lower(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        if case let .string(str) = input {
            return str == str.lowercased()
        }
        return false
    }

    /// Tests if a string is all uppercase.
    @Sendable public static func upper(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        if case let .string(str) = input {
            return str == str.uppercased()
        }
        return false
    }

    /// Tests if the input is true.
    @Sendable public static func `true`(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        return input == .boolean(true)
    }

    /// Tests if the input is false.
    @Sendable public static func `false`(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        return input == .boolean(false)
    }

    /// Tests if the input is a float.
    @Sendable public static func float(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        return input.isDouble
    }

    /// Tests if the input is a sequence (array or string).
    @Sendable public static func sequence(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"] else { return false }
        switch input {
        case .array(_), .string:
            return true
        default:
            return false
        }
    }

    /// Tests if the input is escaped (always returns false for basic implementation).
    @Sendable public static func escaped(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        _ = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        // In basic implementation, inputs are not escaped by default
        return false
    }

    /// Tests if a filter exists by name.
    @Sendable public static func filter(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"], case let .string(filterName) = input else {
            return false
        }
        return Filters.builtIn[filterName] != nil
    }

    /// Tests if a test exists by name.
    @Sendable public static func test(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value"]
        )

        guard let input = arguments["value"], case let .string(testName) = input else {
            return false
        }
        return Tests.builtIn[testName] != nil
    }

    /// Tests if two inputs point to the same memory address (identity test).
    @Sendable public static func sameas(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["a", "b"]
        )

        guard let a = arguments["a"], let b = arguments["b"] else { return false }
        return try a.compare(to: b) == 0
    }

    /// Tests if the input is in a sequence.
    @Sendable public static func `in`(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Bool {
        let arguments = try resolveCallArguments(
            args: args,
            kwargs: kwargs,
            parameters: ["value", "container"]
        )

        guard let input = arguments["value"], let container = arguments["container"] else {
            return false
        }

        switch container {
        case let .array(arr):
            return try arr.contains { try $0.compare(to: input) == 0 }
        case let .string(str):
            if case let .string(searchStr) = input {
                return str.contains(searchStr)
            }
            return false
        case let .object(dict):
            if case let .string(key) = input {
                return dict[key] != nil
            }
            return false
        default:
            return false
        }
    }

    /// Dictionary of all built-in tests available for use in templates.
    ///
    /// Each test function accepts an array of values, optional keyword arguments,
    /// and the current environment, then returns a boolean result.
    public static let builtIn:
        [String: @Sendable ([Value], [String: Value], Environment) throws -> Bool] = [
            "defined": defined,
            "undefined": undefined,
            "none": none,
            "string": string,
            "number": number,
            "boolean": boolean,
            "iterable": iterable,
            "even": even,
            "odd": odd,
            "divisibleby": divisibleby,
            "mapping": mapping,
            "callable": callable,
            "integer": integer,
            "true": `true`,
            "false": `false`,
            "lower": lower,
            "upper": upper,
            "float": float,
            "sequence": sequence,
            "escaped": escaped,
            "filter": filter,
            "test": test,
            "sameas": sameas,
            "in": `in`,
            "eq": eq,
            "==": eq,
            "equalto": eq,
            "ne": ne,
            "!=": ne,
            "gt": gt,
            ">": gt,
            "greaterthan": gt,
            "ge": ge,
            ">=": ge,
            "lt": lt,
            "<": lt,
            "lessthan": lt,
            "le": le,
            "<=": le,
        ]
}
