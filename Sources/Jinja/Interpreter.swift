import Foundation
@_exported import OrderedCollections

// MARK: - Context

/// A context is a dictionary of variables and their values.
public typealias Context = [String: Value]

// MARK: - Environment

/// Execution environment that stores variables and provides context for template rendering.
///
/// The environment maintains the variable scope during template execution and provides
/// configuration options that affect rendering behavior.
public final class Environment: @unchecked Sendable {
    private let parent: Environment?
    private(set) var variables: [String: Value] = [:]

    // Options

    /// Whether leading spaces and tabs are stripped from the start of a line to a block.
    /// The default value is `false`.
    public var lstripBlocks: Bool = false

    /// Whether the first newline after a block is removed.
    /// This applies to block tags, not variable tags.
    /// The default value is `false`.
    public var trimBlocks: Bool = false

    // MARK: -

    /// Creates a new environment with optional parent and initial variables.
    ///
    /// - Parameters:
    ///   - parent: The parent environment to inherit variables from
    ///   - initial: The initial variables to set in this environment
    ///   - includeBuiltIns: Whether to include built-in functions (default: true)
    public init(
        parent: Environment? = nil,
        initial: [String: Value] = [:]
    ) {
        self.parent = parent
        variables = initial

        if parent == nil {
            // Only add built-ins to the root environment to avoid duplication
            for (name, value) in Globals.builtIn {
                variables[name] = value
            }
        }
    }

    /// Gets or sets a variable in the environment.
    ///
    /// When getting a variable, this looks in the current environment first,
    /// then in parent environments. Returns `.undefined` if the variable is not found.
    ///
    /// - Parameter name: The variable name
    /// - Returns: The value associated with the variable name, or `.undefined`
    public subscript(name: String) -> Value {
        get {
            if let value = variables[name] {
                return value
            }

            // Check parent environment
            if let parent {
                return parent[name]
            }

            return .undefined
        }
        set {
            variables[name] = newValue
        }
    }
}

// MARK: - Interpreter

/// Internal control flow exceptions for loop statements.
enum ControlFlow: Error, Sendable {
    /// Control flow exception for break statement.
    case `break`
    /// Control flow exception for continue statement.
    case `continue`
}

/// Executes parsed Jinja template nodes to produce rendered output.
public enum Interpreter {
    /// Interprets nodes and renders them to a string using the given environment.
    ///
    /// - Parameters:
    ///   - nodes: The AST nodes to interpret and render
    ///   - environment: The execution environment containing variables
    /// - Returns: The rendered template output as a string
    /// - Throws: `RuntimeError` if an error occurs during interpretation
    public static func interpret(_ nodes: [Node], environment: Environment) throws -> String {
        // Use the fast path with synchronous environment
        let env = Environment(initial: environment.variables)
        var buffer = ""
        buffer.reserveCapacity(1024)
        try interpret(nodes, env: env, into: &buffer)
        return buffer
    }

    // MARK: -

    static func interpret(
        _ nodes: [Node], env: Environment, into buffer: inout String
    )
        throws
    {
        for node in nodes {
            try interpretNode(node, env: env, into: &buffer)
        }
    }

    static func interpretNode(
        _ node: Node, env: Environment, into buffer: inout String
    )
        throws
    {
        switch node {
        case let .text(content):
            buffer.append(content)

        case .comment:
            // Comments are ignored during execution
            break

        case let .expression(expr):
            let value = try evaluateExpression(expr, env: env)
            buffer.append(value.description)

        case let .statement(stmt):
            try executeStatementWithOutput(stmt, env: env, into: &buffer)
        }
    }

    static func evaluateExpression(_ expr: Expression, env: Environment) throws -> Value {
        switch expr {
        case let .string(value):
            return .string(value)

        case let .number(value):
            return .double(value)

        case let .integer(value):
            return .int(value)

        case let .boolean(value):
            return .boolean(value)

        case .null:
            return .null

        case let .array(elements):
            let values = try elements.map { try evaluateExpression($0, env: env) }
            return .array(values)

        case let .tuple(elements):
            let values = try elements.map { try evaluateExpression($0, env: env) }
            return .array(values) // Tuples are represented as arrays in the runtime

        case let .object(pairs):
            let dict = try pairs.mapValues { try evaluateExpression($0, env: env) }
            return .object(dict)

        case let .identifier(name):
            return env[name]

        case let .unary(op, operand):
            let value = try evaluateExpression(operand, env: env)
            return try evaluateUnaryValue(op, value)

        case let .binary(op, left, right):
            let leftValue = try evaluateExpression(left, env: env)

            // Handle short-circuiting operators
            switch op {
            case .and:
                return leftValue.isTruthy ? try evaluateExpression(right, env: env) : leftValue
            case .or:
                return leftValue.isTruthy ? leftValue : try evaluateExpression(right, env: env)
            default:
                let rightValue = try evaluateExpression(right, env: env)
                return try evaluateBinaryValues(op, leftValue, rightValue)
            }

        case let .ternary(value, test, alternate):
            let testValue = try evaluateExpression(test, env: env)
            if testValue.isTruthy {
                return try evaluateExpression(value, env: env)
            } else if let alternate {
                return try evaluateExpression(alternate, env: env)
            } else {
                return .null
            }

        case let .member(object, property, computed):
            let objectValue = try evaluateExpression(object, env: env)

            if computed {
                let propertyValue = try evaluateExpression(property, env: env)
                return try evaluateComputedMember(objectValue, propertyValue)
            } else {
                guard case let .identifier(propertyName) = property else {
                    throw JinjaError.runtime("Property access requires identifier")
                }
                return try evaluatePropertyMember(objectValue, propertyName)
            }

        case let .filter(operand, filterName, args, kwargs):
            let operandValue = try evaluateExpression(operand, env: env)
            let argValues = try [operandValue] + args.map { try evaluateExpression($0, env: env) }
            let kwargValues = try kwargs.mapValues { try evaluateExpression($0, env: env) }
            return try evaluateFilter(filterName, argValues, kwargs: kwargValues, env: env)

        case let .test(operand, testName, args, negated):
            let operandValue = try evaluateExpression(operand, env: env)
            let argValues = try [operandValue] + args.map { try evaluateExpression($0, env: env) }
            let result = try evaluateTest(testName, argValues, env: env)
            return .boolean(negated ? !result : result)

        case let .call(callableExpr, argsExpr, kwargsExpr):
            let callableValue = try evaluateExpression(callableExpr, env: env)

            // Handle unpacking in arguments
            var argValues: [Value] = []
            for argExpr in argsExpr {
                if case let .unary(.splat, expr) = argExpr {
                    // Unpack the array/tuple
                    let value = try evaluateExpression(expr, env: env)
                    if case let .array(items) = value {
                        argValues.append(contentsOf: items)
                    } else {
                        throw JinjaError.runtime("Cannot unpack non-array value")
                    }
                } else {
                    try argValues.append(evaluateExpression(argExpr, env: env))
                }
            }

            let kwargs = try kwargsExpr.mapValues { try evaluateExpression($0, env: env) }

            switch callableValue {
            case let .function(function):
                return try function(argValues, kwargs, env)
            case let .macro(macro):
                return try callMacro(
                    macro: macro, arguments: argValues, keywordArguments: kwargs, env: env
                )
            default:
                throw JinjaError.runtime("Cannot call non-function value")
            }

        case let .slice(array, start, stop, step):
            let value = try evaluateExpression(array, env: env)
            return try evaluateSlice(value: value, start: start, stop: stop, step: step, env: env)
        }
    }

    /// Synchronous statement execution with output
    static func executeStatementWithOutput(
        _ statement: Statement, env: Environment, into buffer: inout String
    )
        throws
    {
        switch statement {
        case let .if(test, body, alternate):
            let testValue = try evaluateExpression(test, env: env)
            let nodesToExecute = testValue.isTruthy ? body : alternate

            for node in nodesToExecute {
                try interpretNode(node, env: env, into: &buffer)
            }

        case let .for(loopVar, iterable, body, elseBody, test):
            let iterableValue = try evaluateExpression(iterable, env: env)

            switch iterableValue {
            case let .array(items):
                if items.isEmpty {
                    // Execute else block
                    for node in elseBody {
                        try interpretNode(node, env: env, into: &buffer)
                    }
                } else {
                    let childEnv = Environment(parent: env)
                    for (index, item) in items.enumerated() {
                        // Set loop variables
                        switch loopVar {
                        case let .single(varName):
                            childEnv[varName] = item
                        case let .tuple(varNames):
                            if case let .array(tupleItems) = item {
                                for (i, varName) in varNames.enumerated() {
                                    let value = i < tupleItems.count ? tupleItems[i] : .undefined
                                    childEnv[varName] = value
                                }
                            }
                        }

                        childEnv["loop"] = makeLoopObject(index: index, totalCount: items.count)
                        if let test {
                            let testValue = try evaluateExpression(test, env: childEnv)
                            if !testValue.isTruthy { continue }
                        }

                        var shouldBreak = false
                        for node in body {
                            do {
                                try interpretNode(node, env: childEnv, into: &buffer)
                            } catch ControlFlow.break {
                                shouldBreak = true
                                break
                            } catch ControlFlow.continue {
                                break // Break from inner loop (current iteration)
                            }
                        }
                        if shouldBreak { break }
                    }
                }
            case let .object(dict):
                if dict.isEmpty {
                    for node in elseBody {
                        try interpretNode(node, env: env, into: &buffer)
                    }
                } else {
                    let childEnv = Environment(parent: env)
                    for (index, (key, value)) in dict.enumerated() {
                        switch loopVar {
                        case let .single(varName):
                            // Single variable gets the key
                            childEnv[varName] = .string(key)
                        case let .tuple(varNames):
                            // Tuple unpacking: first gets key, second gets value
                            if varNames.count >= 1 {
                                childEnv[varNames[0]] = .string(key)
                            }
                            if varNames.count >= 2 {
                                childEnv[varNames[1]] = value
                            }
                            // Set remaining variables to undefined
                            for i in 2..<varNames.count {
                                childEnv[varNames[i]] = .undefined
                            }
                        }
                        childEnv["loop"] = makeLoopObject(index: index, totalCount: dict.count)
                        if let test {
                            let testValue = try evaluateExpression(test, env: childEnv)
                            if !testValue.isTruthy { continue }
                        }
                        for node in body {
                            try interpretNode(node, env: childEnv, into: &buffer)
                        }
                    }
                }
            case let .string(str):
                let chars = str.map { Value.string(String($0)) }
                if chars.isEmpty {
                    for node in elseBody {
                        try interpretNode(node, env: env, into: &buffer)
                    }
                } else {
                    let childEnv = Environment(parent: env)
                    for (index, item) in chars.enumerated() {
                        switch loopVar {
                        case let .single(varName):
                            childEnv[varName] = item
                        case let .tuple(varNames):
                            for (i, varName) in varNames.enumerated() {
                                childEnv[varName] = i == 0 ? item : .undefined
                            }
                        }
                        childEnv["loop"] = makeLoopObject(index: index, totalCount: chars.count)
                        if let test {
                            let testValue = try evaluateExpression(test, env: childEnv)
                            if !testValue.isTruthy { continue }
                        }
                        for node in body {
                            try interpretNode(node, env: childEnv, into: &buffer)
                        }
                    }
                }
            default:
                throw JinjaError.runtime("Cannot iterate over non-iterable value")
            }

        case let .set(target, value, body):
            if let valueExpr = value {
                let evaluatedValue = try evaluateExpression(valueExpr, env: env)
                try assign(target: target, value: evaluatedValue, env: env)
            } else {
                var bodyBuffer = ""
                try interpret(body, env: env, into: &bodyBuffer)
                let renderedBody = bodyBuffer
                let valueToAssign = Value.string(renderedBody)
                try assign(target: target, value: valueToAssign, env: env)
            }

        case let .macro(name, parameters, defaults, body):
            try registerMacro(
                name: name, parameters: parameters, defaults: defaults, body: body, env: env
            )

        case let .program(nodes):
            try interpret(nodes, env: env, into: &buffer)

        case let .call(callExpr, callerParameters, body):
            let callable: Expression
            let args: [Expression]
            let kwargs: [String: Expression]
            switch callExpr {
            case let .call(c, a, k):
                callable = c
                args = a
                kwargs = k
            default:
                callable = callExpr
                args = []
                kwargs = [:]
            }

            let callableValue = try evaluateExpression(callable, env: env)

            let callTimeEnv = Environment(parent: env)
            callTimeEnv["caller"] = .function { callerArgs, _, _ in
                let bodyEnv = Environment(parent: env)
                for (paramName, value) in zip(callerParameters ?? [], callerArgs) {
                    guard case let .identifier(paramName) = paramName else {
                        throw JinjaError.runtime("Caller parameter must be an identifier")
                    }
                    bodyEnv[paramName] = value
                }
                var bodyBuffer = ""
                try interpret(body, env: bodyEnv, into: &bodyBuffer)
                return .string(bodyBuffer)
            }

            let finalArgs = try args.map { try evaluateExpression($0, env: env) }
            let finalKwargs = try kwargs.mapValues { try evaluateExpression($0, env: env) }

            switch callableValue {
            case let .function(function):
                let result = try function(finalArgs, finalKwargs, callTimeEnv)
                buffer.append(result.description)
            case let .macro(macro):
                let result = try callMacro(
                    macro: macro, arguments: finalArgs, keywordArguments: finalKwargs,
                    env: callTimeEnv
                )
                buffer.append(result.description)
            default:
                throw JinjaError.runtime("Cannot call non-function value")
            }

        case let .filter(filterExpr, body):
            var bodyBuffer = ""
            try interpret(body, env: env, into: &bodyBuffer)
            let renderedBody = bodyBuffer

            if case let .filter(_, name, args, _) = filterExpr {
                var filterArgs = [Value.string(renderedBody)]
                try filterArgs.append(contentsOf: args.map { try evaluateExpression($0, env: env) })
                // TODO: Handle kwargs in filters if necessary
                let filteredValue = try evaluateFilter(name, filterArgs, kwargs: [:], env: env)
                buffer.append(filteredValue.description)
            } else if case let .identifier(name) = filterExpr {
                let filteredValue = try evaluateFilter(
                    name, [.string(renderedBody)], kwargs: [:], env: env
                )
                buffer.append(filteredValue.description)
            } else {
                throw JinjaError.runtime("Invalid filter expression in filter statement")
            }

        case .break:
            throw ControlFlow.break

        case .continue:
            throw ControlFlow.continue
        }
    }

    static func executeStatement(_ statement: Statement, env: Environment) throws {
        switch statement {
        case let .set(target, value, body):
            if let valueExpr = value {
                let evaluatedValue = try evaluateExpression(valueExpr, env: env)
                try assign(target: target, value: evaluatedValue, env: env)
            } else {
                var bodyBuffer = ""
                try interpret(body, env: env, into: &bodyBuffer)
                let renderedBody = bodyBuffer
                let valueToAssign = Value.string(renderedBody)
                try assign(target: target, value: valueToAssign, env: env)
            }

        case let .macro(name, parameters, defaults, body):
            try registerMacro(
                name: name, parameters: parameters, defaults: defaults, body: body, env: env
            )

        // These statements do not produce output directly or are handled elsewhere.
        case .if, .for, .program, .break, .continue, .call, .filter:
            break
        }
    }

    static func assign(target: Expression, value: Value, env: Environment) throws {
        switch target {
        case let .identifier(name):
            env[name] = value
        case let .tuple(expressions):
            guard case let .array(values) = value else {
                throw JinjaError.runtime("Cannot unpack non-array value for tuple assignment.")
            }
            guard expressions.count == values.count else {
                throw JinjaError.runtime(
                    "Tuple assignment mismatch: \(expressions.count) variables and \(values.count) values."
                )
            }
            for (expr, val) in zip(expressions, values) {
                try assign(target: expr, value: val, env: env)
            }
        case let .member(objectExpr, propertyExpr, computed):
            // Handle property assignment like ns.foo = 'bar'
            let objectValue = try evaluateExpression(objectExpr, env: env)

            if computed {
                let propertyValue = try evaluateExpression(propertyExpr, env: env)
                guard case let .string(key) = propertyValue else {
                    throw JinjaError.runtime("Computed property key must be a string")
                }
                if case var .object(dict) = objectValue {
                    dict[key] = value
                    // Update the object in the environment
                    if case let .identifier(name) = objectExpr {
                        env[name] = .object(dict)
                    }
                }
            } else {
                guard case let .identifier(propertyName) = propertyExpr else {
                    throw JinjaError.runtime("Property assignment requires identifier")
                }
                if case var .object(dict) = objectValue {
                    dict[propertyName] = value
                    // Update the object in the environment
                    if case let .identifier(name) = objectExpr {
                        env[name] = .object(dict)
                    }
                }
            }
        default:
            throw JinjaError.runtime("Invalid target for assignment: \(target)")
        }
    }

    // MARK: -

    static func registerMacro(
        name: String, parameters: [String], defaults: OrderedDictionary<String, Expression>,
        body: [Node],
        env: Environment
    ) throws {
        env[name] = .macro(
            Macro(name: name, parameters: parameters, defaults: defaults, body: body))
    }

    static func callMacro(
        macro: Macro, arguments: [Value], keywordArguments: [String: Value], env: Environment
    ) throws -> Value {
        let macroEnv = Environment(parent: env)

        let caller = env["caller"]
        if caller != .undefined {
            macroEnv["caller"] = caller
        }

        // Start with defaults
        for (key, expr) in macro.defaults {
            // Evaluate defaults in current env
            let val = try evaluateExpression(expr, env: env)
            macroEnv[key] = val
        }

        // Bind positional args
        for (index, paramName) in macro.parameters.enumerated() {
            let value =
                index < arguments.count ? arguments[index] : macroEnv[paramName]
            macroEnv[paramName] = value
        }

        // Bind keyword args
        for (key, value) in keywordArguments {
            macroEnv[key] = value
        }

        var macroBuffer = ""
        try interpret(macro.body, env: macroEnv, into: &macroBuffer)
        return .string(macroBuffer)
    }

    static func evaluateBinaryValues(
        _ op: Expression.BinaryOp, _ left: Value, _ right: Value
    ) throws
        -> Value
    {
        switch op {
        case .add:
            try left.add(with: right)
        case .subtract:
            try left.subtract(by: right)
        case .multiply:
            try left.multiply(by: right)
        case .divide:
            try left.divide(by: right)
        case .floorDivide:
            try left.floorDivide(by: right)
        case .power:
            try left.power(by: right)
        case .modulo:
            try left.modulo(by: right)
        case .concat:
            try left.concatenate(with: right)
        case .equal:
            .boolean(left.isEquivalent(to: right))
        case .notEqual:
            .boolean(!left.isEquivalent(to: right))
        case .less:
            try .boolean(left.compare(to: right) < 0)
        case .lessEqual:
            try .boolean(left.compare(to: right) <= 0)
        case .greater:
            try .boolean(left.compare(to: right) > 0)
        case .greaterEqual:
            try .boolean(left.compare(to: right) >= 0)
        case .and:
            left.isTruthy ? right : left
        case .or:
            left.isTruthy ? left : right
        case .in:
            try .boolean(left.isContained(in: right))
        case .notIn:
            try .boolean(!(left.isContained(in: right)))
        }
    }

    static func evaluateUnaryValue(_ op: Expression.UnaryOp, _ value: Value) throws -> Value {
        switch op {
        case .not:
            return .boolean(!value.isTruthy)
        case .minus:
            switch value {
            case let .double(n):
                return .double(-n)
            case let .int(i):
                return .int(-i)
            default:
                throw JinjaError.runtime("Cannot negate non-numeric value")
            }
        case .plus:
            switch value {
            case .double, .int:
                return value
            default:
                throw JinjaError.runtime("Cannot apply unary plus to non-numeric value")
            }
        case .splat:
            // This should not be evaluated directly - it's only used for unpacking in calls
            throw JinjaError.runtime("Unpacking operator can only be used in function calls")
        }
    }

    static func evaluateComputedMember(_ object: Value, _ property: Value) throws -> Value {
        switch (object, property) {
        case let (.array(arr), .int(index)):
            let safeIndex = index < 0 ? arr.count + index : index
            guard safeIndex >= 0, safeIndex < arr.count else {
                return .undefined
            }
            return arr[safeIndex]

        case let (.object(obj), .string(key)):
            return obj[key] ?? .undefined

        case let (.string(str), .int(index)):
            let safeIndex = index < 0 ? str.count + index : index
            guard safeIndex >= 0, safeIndex < str.count else {
                return .undefined
            }
            let char = str[str.index(str.startIndex, offsetBy: safeIndex)]
            return .string(String(char))

        default:
            return .undefined
        }
    }

    static func evaluatePropertyMember(_ object: Value, _ propertyName: String) throws
        -> Value
    {
        switch object {
        case let .string(str):
            switch propertyName {
            case "upper":
                return .function { _, _, _ in .string(str.uppercased()) }
            case "lower":
                return .function { _, _, _ in .string(str.lowercased()) }
            case "title":
                return .function { _, _, _ in .string(str.capitalized) }
            case "strip":
                return .function { _, _, _ in
                    .string(str.trimmingCharacters(in: .whitespacesAndNewlines))
                }
            case "lstrip":
                return .function { _, _, _ in
                    let trimmed = str.drop(while: { $0.isWhitespace })
                    return .string(String(trimmed))
                }
            case "rstrip":
                return .function { _, _, _ in
                    let reversed = str.reversed().drop(while: { $0.isWhitespace })
                    return .string(String(reversed.reversed()))
                }
            case "split":
                return .function { args, _, _ in
                    if args.isEmpty {
                        // Split on whitespace
                        let components = str.split(separator: " ").map(String.init)
                        return .array(components.map(Value.string))
                    } else if case let .string(separator) = args[0] {
                        if args.count > 1, case let .int(limit) = args[1] {
                            // Split with limit: split at most 'limit' times
                            var components: [String] = []
                            var remaining = str
                            var splits = 0

                            while splits < limit, let range = remaining.range(of: separator) {
                                components.append(String(remaining[..<range.lowerBound]))
                                remaining = String(remaining[range.upperBound...])
                                splits += 1
                            }
                            // Add the remainder
                            components.append(remaining)
                            return .array(components.map(Value.string))
                        } else {
                            let components = str.components(separatedBy: separator)
                            return .array(components.map(Value.string))
                        }
                    }
                    return .array([.string(str)])
                }
            case "replace":
                return .function { args, kwargs, _ in
                    guard args.count >= 2,
                          case let .string(old) = args[0],
                          case let .string(new) = args[1]
                    else {
                        return .string(str)
                    }

                    // Check for count parameter in args or kwargs
                    var maxReplacements: Int? = nil
                    if args.count > 2, case let .int(count) = args[2] {
                        maxReplacements = count
                    } else if let countValue = kwargs["count"],
                              case let .int(count) = countValue
                    {
                        maxReplacements = count
                    }

                    // Special case: replacing empty string inserts at character boundaries
                    if old.isEmpty {
                        var result = ""
                        var replacements = 0
                        for char in str {
                            if let count = maxReplacements, replacements >= count {
                                result += String(char)
                            } else {
                                result += new + String(char)
                                replacements += 1
                            }
                        }
                        // Add final replacement if we haven't hit the count limit
                        if maxReplacements == nil || replacements < maxReplacements! {
                            result += new
                        }
                        return .string(result)
                    }

                    if let count = maxReplacements {
                        // Replace only the first 'count' occurrences
                        var result = str
                        var replacements = 0
                        while replacements < count, let range = result.range(of: old) {
                            result.replaceSubrange(range, with: new)
                            replacements += 1
                        }
                        return .string(result)
                    } else {
                        return .string(str.replacingOccurrences(of: old, with: new))
                    }
                }
            default:
                return .undefined
            }
        case let .object(obj):
            // Support Python-like dict.items() for iteration
            if propertyName == "items" {
                let fn: @Sendable ([Value], [String: Value], Environment) throws -> Value = {
                    _, _, _ in
                    let pairs = obj.map { key, value in Value.array([.string(key), value]) }
                    return .array(pairs)
                }
                return .function(fn)
            }
            // Support Python-like dict.get(key, default) method
            if propertyName == "get" {
                let fn: @Sendable ([Value], [String: Value], Environment) throws -> Value = {
                    args, _, _ in
                    guard !args.isEmpty else {
                        throw JinjaError.runtime("get() requires at least 1 argument")
                    }

                    let key: String = switch args[0] {
                    case let .string(s):
                        s
                    default:
                        args[0].description
                    }

                    let defaultValue = args.count > 1 ? args[1] : .null
                    return obj[key] ?? defaultValue
                }
                return .function(fn)
            }
            return obj[propertyName] ?? .undefined
        default:
            return .undefined
        }
    }

    static func evaluateTest(_ testName: String, _ argValues: [Value], env: Environment)
        throws -> Bool
    {
        // Try environment-provided tests first
        let testValue = env[testName]
        if case let .function(fn) = testValue {
            let result = try fn(argValues, [:], env)
            if case let .boolean(b) = result { return b }
            return result.isTruthy
        }

        // Fallback to built-in tests
        if let testFunction = Tests.builtIn[testName] {
            return try testFunction(argValues, [:], env)
        }

        throw JinjaError.runtime("Unknown test: \(testName)")
    }

    static func evaluateFilter(
        _ filterName: String, _ argValues: [Value], kwargs: [String: Value], env: Environment
    )
        throws -> Value
    {
        // Try environment-provided filters first
        let filterValue = env[filterName]
        if case let .function(fn) = filterValue {
            return try fn(argValues, kwargs, env)
        }

        // Fallback to built-in filters
        if let filterFunction = Filters.builtIn[filterName] {
            return try filterFunction(argValues, kwargs, env)
        }

        throw JinjaError.runtime("Unknown filter: \(filterName)")
    }

    private static func makeLoopObject(index: Int, totalCount: Int) -> Value {
        var loopContext: OrderedDictionary<String, Value> = [
            "index": .int(index + 1),
            "index0": .int(index),
            "first": .boolean(index == 0),
            "last": .boolean(index == totalCount - 1),
            "length": .int(totalCount),
            "revindex": .int(totalCount - index),
            "revindex0": .int(totalCount - index - 1),
        ]

        loopContext["cycle"] = .function { args, _, _ in
            guard !args.isEmpty else { return .string("") }
            let cycleIndex = index % args.count
            return args[cycleIndex]
        }

        return .object(loopContext)
    }

    private static func evaluateSlice(
        value: Value, start: Expression?, stop: Expression?, step: Expression?, env: Environment
    ) throws -> Value {
        let startVal = try start.map { try evaluateExpression($0, env: env) }
        let stopVal = try stop.map { try evaluateExpression($0, env: env) }
        let stepVal = try step.map { try evaluateExpression($0, env: env) }

        let step: Int
        if let s = stepVal, case let .int(val) = s {
            if val == 0 { throw JinjaError.runtime("Slice step cannot be zero") }
            step = val
        } else {
            step = 1
        }

        switch value {
        case let .array(items):
            let count = items.count

            let startIdx: Int = if let s = startVal, case let .int(val) = s {
                val >= 0 ? val : count + val
            } else {
                step > 0 ? 0 : count - 1
            }

            let stopIdx: Int = if let s = stopVal, case let .int(val) = s {
                val >= 0 ? val : count + val
            } else {
                step > 0 ? count : -1
            }

            var result: [Value] = []
            for i in stride(from: startIdx, to: stopIdx, by: step) {
                if i >= 0, i < count {
                    result.append(items[i])
                }
            }
            return .array(result)

        case let .string(str):
            let count = str.count

            let startIdx: Int = if let s = startVal, case let .int(val) = s {
                val >= 0 ? val : count + val
            } else {
                step > 0 ? 0 : count - 1
            }

            let stopIdx: Int = if let s = stopVal, case let .int(val) = s {
                val >= 0 ? val : count + val
            } else {
                step > 0 ? count : -1
            }

            var result = ""
            for i in stride(from: startIdx, to: stopIdx, by: step) {
                if i >= 0, i < count {
                    let index = str.index(str.startIndex, offsetBy: i)
                    result.append(str[index])
                }
            }
            return .string(String(result))

        default:
            throw JinjaError.runtime("Slice requires array or string")
        }
    }
}
