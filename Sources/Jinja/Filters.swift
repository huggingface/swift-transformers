import Foundation

/// Built-in filters for Jinja template rendering.
///
/// Filters transform values in template expressions using the pipe syntax (`|`).
/// All filter functions follow the same signature pattern, accepting an array of values
/// (with the filtered value as the first element), optional keyword arguments, and an environment.
public enum Filters {
    // MARK: - Basic String Filters

    /// Converts a string to uppercase.
    @Sendable public static func upper(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .string(str) = args.first else {
            throw JinjaError.runtime("upper filter requires string")
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        return .string(str.uppercased())
    }

    /// Converts a string to lowercase.
    @Sendable public static func lower(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .string(str) = args.first else {
            throw JinjaError.runtime("lower filter requires string")
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        return .string(str.lowercased())
    }

    /// Returns the length of a string, array, or object.
    @Sendable public static func length(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        switch args.first {
        case let .string(str):
            return .int(str.count)
        case let .array(arr):
            return .int(arr.count)
        case let .object(obj):
            return .int(obj.count)
        default:
            throw JinjaError.runtime("length filter requires string, array, or object")
        }
    }

    /// Joins an array of values with a separator.
    @Sendable public static func join(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .array(array) = args.first else {
            throw JinjaError.runtime("join filter requires array")
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["separator"],
            defaults: ["separator": .string("")]
        )

        guard case let .string(separator) = arguments["separator"] else {
            throw JinjaError.runtime("join filter requires string separator")
        }

        let strings = array.map { $0.description }
        return .string(strings.joined(separator: separator))
    }

    /// Returns a default value if the input is undefined,
    /// or if the input is false and the second / `boolean` argument is `true`.
    @Sendable public static func `default`(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        let input = args.first ?? .undefined

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["default_value", "boolean"],
            defaults: ["boolean": .boolean(false)]
        )

        let defaultValue = arguments["default_value"]!
        let boolean = arguments["boolean"]!.isTruthy

        // If input is undefined, return default value
        if input == .undefined {
            return defaultValue
        }

        // If boolean is true and input is false, return default value
        if boolean, input == false {
            return defaultValue
        }

        // Otherwise return the input value
        return input
    }

    // MARK: - Array Filters

    /// Returns the first item from an array.
    @Sendable public static func first(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else {
            return .undefined
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        switch value {
        case let .array(arr):
            return arr.first ?? .undefined
        case let .string(str):
            return str.first.map { .string(String($0)) } ?? .undefined
        default:
            throw JinjaError.runtime("first filter requires array or string")
        }
    }

    /// Returns the last item from an array.
    @Sendable public static func last(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else {
            return .undefined
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        switch value {
        case let .array(arr):
            return arr.last ?? .undefined
        case let .string(str):
            return str.last.map { .string(String($0)) } ?? .undefined
        default:
            throw JinjaError.runtime("last filter requires array or string")
        }
    }

    /// Returns a random item from an array.
    @Sendable public static func random(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else {
            return .undefined
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        switch value {
        case let .array(arr):
            return arr.randomElement() ?? .undefined
        case let .string(str):
            return str.randomElement().map { .string(String($0)) } ?? .undefined
        case let .object(dict):
            if dict.isEmpty { return .undefined }
            let randomIndex = dict.keys.indices.randomElement()!
            let randomKey = dict.keys[randomIndex]
            return .string(randomKey)
        default:
            return .undefined
        }
    }

    /// Reverses an array or string.
    @Sendable public static func reverse(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else {
            return .undefined
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        switch value {
        case let .array(arr):
            return .array(arr.reversed())
        case let .string(str):
            return .string(String(str.reversed()))
        default:
            throw JinjaError.runtime("reverse filter requires array or string")
        }
    }

    /// Sorts an array.
    @Sendable public static func sort(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .array(items) = value else {
            return .array([])
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["reverse", "case_sensitive", "attribute"],
            defaults: [
                "reverse": .boolean(false),
                "case_sensitive": .boolean(true),
                "attribute": .null,
            ]
        )

        let reverse = arguments["reverse"]!.isTruthy
        let caseSensitive = arguments["case_sensitive"]!.isTruthy

        let sortedItems: [Value] = if case let .string(attribute) = arguments["attribute"] {
            try items.sorted { a, b in
                let aValue = try Interpreter.evaluatePropertyMember(a, attribute)
                let bValue = try Interpreter.evaluatePropertyMember(b, attribute)
                let comparison = try aValue.compare(to: bValue)
                return reverse ? comparison > 0 : comparison < 0
            }
        } else {
            try items.sorted { a, b in
                let comparison: Int = if !caseSensitive, case let .string(aStr) = a, case let .string(bStr) = b {
                    aStr.lowercased() < bStr.lowercased()
                        ? -1 : aStr.lowercased() > bStr.lowercased() ? 1 : 0
                } else {
                    try a.compare(to: b)
                }
                return reverse ? comparison > 0 : comparison < 0
            }
        }

        return .array(sortedItems)
    }

    /// Groups items by a given attribute.
    @Sendable public static func groupby(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .array(items) = value else {
            return .array([])
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["attribute"],
            defaults: [:]
        )

        guard case let .string(attribute) = arguments["attribute"] else {
            throw JinjaError.runtime("groupby filter requires attribute parameter")
        }

        var groups = OrderedDictionary<Value, [Value]>()
        for item in items {
            let key = try Interpreter.evaluatePropertyMember(item, attribute)
            groups[key, default: []].append(item)
        }
        let result = groups.map { key, value in
            Value.object([
                "grouper": key,
                "list": .array(value),
            ])
        }
        return .array(result)
    }

    /// Slices an array into multiple slices.
    @Sendable public static func slice(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .array(items) = value else {
            return .array([])
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["numSlices", "fillWith"],
            defaults: ["fillWith": .null]
        )

        guard case let .int(numSlices) = arguments["numSlices"], numSlices > 0 else {
            throw JinjaError.runtime("slice filter requires positive integer numSlices parameter")
        }

        let fillWith = arguments["fillWith"]!
        var result = Array(repeating: [Value](), count: numSlices)
        let itemsPerSlice = (items.count + numSlices - 1) / numSlices

        for i in 0..<itemsPerSlice {
            for j in 0..<numSlices {
                let index = i * numSlices + j
                if index < items.count {
                    result[j].append(items[index])
                } else {
                    result[j].append(fillWith)
                }
            }
        }

        return .array(result.map { .array($0) })
    }

    /// Maps items through a filter or extracts attribute values.
    @Sendable public static func map(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .array(items) = value else {
            return .array([])
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["filterName", "attribute"],
            defaults: ["filterName": .null, "attribute": .null]
        )

        if case let .string(filterName) = arguments["filterName"] {
            return try .array(
                items.map {
                    try Interpreter.evaluateFilter(filterName, [$0], kwargs: [:], env: env)
                })
        } else if case let .string(attribute) = arguments["attribute"] {
            return try .array(
                items.map {
                    try Interpreter.evaluatePropertyMember($0, attribute)
                })
        }

        return .array([])
    }

    /// Selects items that pass a test.
    @Sendable public static func select(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .array(items) = value else {
            return .array([])
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["testName"],
            defaults: [:]
        )

        guard case let .string(testName) = arguments["testName"] else {
            throw JinjaError.runtime("select filter requires testName parameter")
        }

        let testArgs = Array(args.dropFirst(2))
        return try .array(
            items.filter {
                try Interpreter.evaluateTest(testName, [$0] + testArgs, env: env)
            })
    }

    /// Rejects items that pass a test.
    @Sendable public static func reject(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .array(items) = value else {
            return .array([])
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["testName"],
            defaults: [:]
        )

        guard case let .string(testName) = arguments["testName"] else {
            throw JinjaError.runtime("reject filter requires testName parameter")
        }

        let testArgs = Array(args.dropFirst(2))
        return try .array(
            items.filter {
                try !Interpreter.evaluateTest(testName, [$0] + testArgs, env: env)
            })
    }

    /// Selects items with an attribute that passes a test.
    /// If no test is specified,
    /// the attribute's value will be evaluated as a Boolean.
    @Sendable public static func selectattr(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .array(items)? = args.first else {
            return .array([])
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["attribute", "testName"],
            defaults: ["testName": .null]
        )

        guard case let .string(attribute) = arguments["attribute"] else {
            throw JinjaError.runtime("selectattr filter requires attribute parameter")
        }

        let testArgs = Array(args.dropFirst(2))
        return try .array(
            items.filter {
                let attrValue = try Interpreter.evaluatePropertyMember($0, attribute)
                guard case let .string(testName) = arguments["testName"] else {
                    return attrValue.isTruthy
                }

                return try Interpreter.evaluateTest(
                    testName, [attrValue] + testArgs.dropFirst(1), env: env
                )
            })
    }

    /// Rejects items with an attribute that passes a test.
    /// If no test is specified,
    /// the attribute's value will be evaluated as a Boolean.
    @Sendable public static func rejectattr(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .array(items) = value else {
            return .array([])
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["attribute", "testName"],
            defaults: ["testName": .null]
        )

        guard case let .string(attribute) = arguments["attribute"] else {
            throw JinjaError.runtime("rejectattr filter requires attribute parameter")
        }

        let testArgs = Array(args.dropFirst(2))
        return try .array(
            items.filter {
                let attrValue = try Interpreter.evaluatePropertyMember($0, attribute)
                guard case let .string(testName) = arguments["testName"] else {
                    return !attrValue.isTruthy
                }

                return try !Interpreter.evaluateTest(
                    testName, [attrValue] + testArgs.dropFirst(1), env: env
                )
            })
    }

    // MARK: - Object Filters

    /// Gets an attribute from an object.
    @Sendable public static func attr(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let obj = args.first else {
            return .undefined
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["attribute"],
            defaults: [:]
        )

        guard case let .string(attribute) = arguments["attribute"] else {
            throw JinjaError.runtime("attr filter requires attribute parameter")
        }

        return try Interpreter.evaluatePropertyMember(obj, attribute)
    }

    /// Sorts a dictionary by keys and returns key-value pairs.
    @Sendable public static func dictsort(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .object(dict) = args.first else {
            return .array([])
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["case_sensitive", "by", "reverse"],
            defaults: [
                "case_sensitive": .boolean(false),
                "by": .string("key"),
                "reverse": .boolean(false),
            ]
        )

        let caseSensitive = arguments["case_sensitive"]!.isTruthy
        let by: String = if case let .string(s) = arguments["by"] {
            s
        } else {
            "key"
        }
        let reverse = arguments["reverse"]!.isTruthy

        let sortedPairs: [(key: String, value: Value)] = if by == "value" {
            dict.sorted { a, b in
                let comparison =
                    caseSensitive
                        ? a.value.description.compare(b.value.description)
                        : a.value.description.localizedCaseInsensitiveCompare(b.value.description)
                return reverse ? comparison == .orderedDescending : comparison == .orderedAscending
            }
        } else {
            dict.sorted { a, b in
                let comparison =
                    caseSensitive
                        ? a.key.compare(b.key)
                        : a.key.localizedCaseInsensitiveCompare(b.key)
                return reverse ? comparison == .orderedDescending : comparison == .orderedAscending
            }
        }

        let resultArray = sortedPairs.map { key, value in
            Value.array([.string(key), value])
        }
        return .array(resultArray)
    }

    // MARK: - String Processing Filters

    /// Escapes HTML characters.
    @Sendable public static func forceescape(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .string(str) = args.first else {
            return .string("")
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        let escaped =
            str
                .replacingOccurrences(of: "&", with: "&amp;")
                .replacingOccurrences(of: "<", with: "&lt;")
                .replacingOccurrences(of: ">", with: "&gt;")
                .replacingOccurrences(of: "\"", with: "&quot;")
                .replacingOccurrences(of: "'", with: "&#39;")
        return .string(escaped)
    }

    /// Marks a string as safe (no-op for basic implementation).
    @Sendable public static func safe(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        return args.first ?? .string("")
    }

    /// Strips HTML tags from a string.
    @Sendable public static func striptags(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .string(str) = args.first else {
            return .string("")
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        let pattern = "<[^>]+>"
        let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        let nsRange = NSRange(str.startIndex..<str.endIndex, in: str)
        let noTags = regex?.stringByReplacingMatches(in: str, options: [], range: nsRange, withTemplate: "") ?? str
        let components = noTags.components(separatedBy: .whitespacesAndNewlines)
        return .string(components.filter { !$0.isEmpty }.joined(separator: " "))
    }

    /// Basic string formatting.
    @Sendable public static func format(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard args.count > 1, case let .string(formatString) = args[0] else {
            return args.first ?? .string("")
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        let formatArgs = Array(args.dropFirst())
        var result = ""
        var formatIdx = formatString.startIndex
        var argIdx = 0
        while formatIdx < formatString.endIndex {
            let char = formatString[formatIdx]
            if char == "%", argIdx < formatArgs.count {
                formatIdx = formatString.index(after: formatIdx)
                if formatIdx < formatString.endIndex {
                    let specifier = formatString[formatIdx]
                    if specifier == "s" {
                        result += formatArgs[argIdx].description
                        argIdx += 1
                    } else {
                        result.append("%")
                        result.append(specifier)
                    }
                } else {
                    result.append("%")
                }
            } else {
                result.append(char)
            }
            formatIdx = formatString.index(after: formatIdx)
        }
        return .string(result)
    }

    /// Wraps text to a specified width.
    @Sendable public static func wordwrap(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .string(str) = value else {
            return .string("")
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["width", "break_long_words"],
            defaults: [
                "width": .int(79),
                "break_long_words": .boolean(true),
            ]
        )

        let width: Int = if case let .int(w) = arguments["width"] {
            w
        } else {
            79
        }
        _ = arguments["break_long_words"]!.isTruthy

        var lines = [String]()
        let paragraphs = str.components(separatedBy: .newlines)
        for paragraph in paragraphs {
            var line = ""
            let words = paragraph.components(separatedBy: .whitespaces)
            for word in words {
                if line.isEmpty {
                    line = word
                } else if line.count + word.count + 1 <= width {
                    line += " \(word)"
                } else {
                    lines.append(line)
                    line = word
                }
            }
            if !line.isEmpty {
                lines.append(line)
            }
        }
        return .string(lines.joined(separator: "\n"))
    }

    /// Formats file size in human readable format.
    @Sendable public static func filesizeformat(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else {
            return .string("")
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["binary"],
            defaults: ["binary": .boolean(false)]
        )

        guard case let .double(num) = value else {
            return .string("")
        }

        let binary = arguments["binary"]!.isTruthy
        let bytes = num
        let unit: Double = binary ? 1024 : 1000
        if bytes < unit {
            return .string("\(Int(bytes)) Bytes")
        }
        let exp = Int(log(bytes) / log(unit))
        let pre = (binary ? "KMGTPEZY" : "kMGTPEZY")
        let preIndex = pre.index(pre.startIndex, offsetBy: exp - 1)
        let preChar = pre[preIndex]
        let suffix = binary ? "iB" : "B"
        return .string(
            String(format: "%.1f %s\(suffix)", bytes / pow(unit, Double(exp)), String(preChar)))
    }

    /// Formats object attributes as XML attributes.
    @Sendable public static func xmlattr(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .object(dict) = value else {
            return .string("")
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["autospace"],
            defaults: ["autospace": .boolean(true)]
        )

        let autospace = arguments["autospace"]!.isTruthy
        var result = ""
        for (key, value) in dict {
            if value == .null || value == .undefined { continue }
            // Validate key doesn't contain invalid characters
            if key.contains(" ") || key.contains("/") || key.contains(">") || key.contains("=") {
                throw JinjaError.runtime("Invalid character in XML attribute key: '\(key)'")
            }
            let escapedValue = value.description
                .replacingOccurrences(of: "&", with: "&amp;")
                .replacingOccurrences(of: "<", with: "&lt;")
                .replacingOccurrences(of: ">", with: "&gt;")
                .replacingOccurrences(of: "\"", with: "&quot;")
            result += "\(key)=\"\(escapedValue)\""
        }
        if autospace, !result.isEmpty {
            result = " " + result
        }
        return .string(result)
    }

    /// Converts a value to a string.
    @Sendable public static func string(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else { return .string("") }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        return .string(value.description)
    }

    // MARK: - Additional Filters

    /// Trims whitespace from a string.
    @Sendable public static func trim(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .string(str) = args.first else {
            return .string("")
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        return .string(str.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Escapes HTML characters (alias for forceescape).
    @Sendable public static func escape(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        return try forceescape(args, kwargs: kwargs, env: env)
    }

    /// Converts value to JSON string.
    @Sendable public static func tojson(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else { return .string("null") }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["indent"],
            defaults: ["indent": .null]
        )

        let encoder = JSONEncoder()
        if let indent = arguments["indent"],
           case let .int(count) = indent,
           count > 0
        {
            encoder.outputFormatting = .prettyPrinted
        }

        if let jsonData = (try? encoder.encode(value)),
           let jsonString = String(data: jsonData, encoding: .utf8)
        {
            return .string(jsonString)
        } else {
            return .string("null")
        }
    }

    /// Returns absolute value of a number.
    @Sendable public static func abs(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else {
            return .int(0)
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        switch value {
        case let .int(i):
            return .int(Swift.abs(i))
        case let .double(n):
            return .double(Swift.abs(n))
        default:
            throw JinjaError.runtime("abs filter requires number or integer")
        }
    }

    /// Capitalizes the first letter and lowercases the rest.
    @Sendable public static func capitalize(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .string(str) = args.first else {
            return .string("")
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        return .string(str.prefix(1).uppercased() + str.dropFirst().lowercased())
    }

    /// Centers a string within a specified width.
    @Sendable public static func center(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .string(str) = args.first else {
            return args.first ?? .string("")
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["width"],
            defaults: [:]
        )

        guard case let .int(width) = arguments["width"] else {
            throw JinjaError.runtime("center filter requires width parameter")
        }

        let padCount = width - str.count
        if padCount <= 0 {
            return .string(str)
        }
        let leftPad = String(repeating: " ", count: padCount / 2)
        let rightPad = String(repeating: " ", count: padCount - (padCount / 2))
        return .string(leftPad + str + rightPad)
    }

    /// Converts a value to float.
    @Sendable public static func float(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else { return .double(0.0) }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        switch value {
        case let .int(i):
            return .double(Double(i))
        case let .double(n):
            return .double(n)
        case let .string(s):
            return .double(Double(s) ?? 0.0)
        default:
            return .double(0.0)
        }
    }

    /// Converts a value to integer.
    @Sendable public static func int(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else { return .int(0) }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        switch value {
        case let .int(i):
            return .int(i)
        case let .double(n):
            return .int(Int(n))
        case let .string(s):
            return .int(Int(s) ?? 0)
        default:
            return .int(0)
        }
    }

    /// Converts a value to list.
    /// If it was a string the returned list will be a list of characters.
    @Sendable public static func list(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else { return .array([]) }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        switch value {
        case let .array(arr):
            return .array(arr)
        case let .string(str):
            return .array(str.map { .string(String($0)) })
        case let .object(dict):
            return .array(dict.values.map { $0 })
        default:
            return .array([])
        }
    }

    /// Returns the maximum value from an array.
    @Sendable public static func max(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .array(items) = value else { return .undefined }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        return items.max(by: { a, b in
            do {
                return try a.compare(to: b) < 0
            } catch {
                return false
            }
        }) ?? .undefined
    }

    /// Returns the minimum value from an array.
    @Sendable public static func min(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .array(items) = value else { return .undefined }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        return items.min(by: { a, b in
            do {
                return try a.compare(to: b) < 0
            } catch {
                return false
            }
        }) ?? .undefined
    }

    /// Rounds a number to specified precision.
    @Sendable public static func round(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else { return .double(0.0) }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["precision", "method"],
            defaults: ["precision": .int(0), "method": .string("common")]
        )

        let number: Double
        switch value {
        case let .int(intValue):
            number = Double(intValue)
        case let .double(doubleValue):
            number = doubleValue
        default:
            return value // Or throw error
        }

        let precision: Int = if case let .int(p) = arguments["precision"]! {
            p
        } else {
            0
        }

        let method: String = if case let .string(m) = arguments["method"]! {
            m
        } else {
            "common"
        }

        if method == "common" {
            let divisor = pow(10.0, Double(precision))
            return .double((number * divisor).rounded() / divisor)
        } else if method == "ceil" {
            let divisor = pow(10.0, Double(precision))
            return .double(ceil(number * divisor) / divisor)
        } else if method == "floor" {
            let divisor = pow(10.0, Double(precision))
            return .double(floor(number * divisor) / divisor)
        }
        return .double(number)
    }

    /// Capitalizes each word in a string.
    @Sendable public static func title(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .string(str) = args.first else {
            return .string("")
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        return .string(str.capitalized)
    }

    /// Counts words in a string.
    @Sendable public static func wordcount(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .string(str) = args.first else {
            return .int(0)
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        let words = str.split { $0.isWhitespace || $0.isNewline }
        return .int(words.count)
    }

    /// Return string with all occurrences of a substring replaced with a new one.
    /// The first argument is the substring that should be replaced,
    /// the second is the replacement string.
    /// If the optional third argument count is given,
    /// only the first count occurrences are replaced.
    @Sendable public static func replace(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .string(str) = args.first else {
            return args.first ?? .string("")
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["old", "new", "count"],
            defaults: ["count": .null]
        )

        guard case let .string(old) = arguments["old"],
              case let .string(new) = arguments["new"]
        else {
            throw JinjaError.runtime("replace() requires 'old' and 'new' string arguments.")
        }

        // Handle count parameter - can be positional (3rd arg) or named (count=)
        let count: Int? = if case let .int(c) = arguments["count"] {
            c
        } else {
            nil
        }

        // Special case: replacing empty string inserts at character boundaries
        if old.isEmpty {
            var result = ""
            var replacements = 0

            // Insert at the beginning
            if count == nil || replacements < count! {
                result += new
                replacements += 1
            }

            // Insert between each character
            for char in str {
                result += String(char)
                if count == nil || replacements < count! {
                    result += new
                    replacements += 1
                }
            }

            return .string(result)
        }

        // Regular case: replace occurrences of the substring
        var result = ""
        var remaining = str
        var replacements = 0

        while let range = remaining.range(of: old) {
            if let count, replacements >= count {
                break
            }

            result += remaining[..<range.lowerBound]
            result += new
            remaining = String(remaining[range.upperBound...])
            replacements += 1
        }

        result += remaining
        return .string(result)
    }

    /// URL encodes a string or object.
    @Sendable public static func urlencode(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else {
            return .string("")
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        let str: String
        if case let .string(s) = value {
            str = s
        } else if case let .object(dict) = value {
            str = dict.map { key, value in
                let encodedKey =
                    key.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
                let encodedValue =
                    value.description.addingPercentEncoding(
                        withAllowedCharacters: .urlQueryAllowed) ?? ""
                return "\(encodedKey)=\(encodedValue)"
            }.joined(separator: "&")
        } else {
            return .string("")
        }

        return .string(str.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")
    }

    /// Batches items into groups.
    @Sendable public static func batch(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .array(items) = value else {
            return .array([])
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["batchSize", "fillWith"],
            defaults: ["fillWith": .null]
        )

        guard case let .int(batchSize) = arguments["batchSize"], batchSize > 0 else {
            throw JinjaError.runtime("batch filter requires positive integer batchSize parameter")
        }

        let fillWith = arguments["fillWith"]!

        var result = [Value]()
        var batch = [Value]()
        for item in items {
            batch.append(item)
            if batch.count == batchSize {
                result.append(.array(batch))
                batch = []
            }
        }
        if !batch.isEmpty {
            while batch.count < batchSize {
                batch.append(fillWith)
            }
            result.append(.array(batch))
        }
        return .array(result)
    }

    /// Sums values in an array.
    @Sendable public static func sum(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .array(items) = value else {
            return .int(0)
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["attribute", "start"],
            defaults: ["attribute": .null, "start": .int(0)]
        )

        let start = arguments["start"]!

        let valuesToSum: [Value] = if case let .string(attribute) = arguments["attribute"] {
            try items.map { item in
                try Interpreter.evaluatePropertyMember(item, attribute)
            }
        } else {
            items
        }

        let sum = try valuesToSum.reduce(start) { acc, next in
            try acc.add(with: next)
        }
        return sum
    }

    /// Truncates a string to a specified length.
    @Sendable public static func truncate(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .string(str) = args.first else {
            return .string("")
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["length", "killwords", "end"],
            defaults: ["length": .int(255), "killwords": .boolean(false), "end": .string("...")]
        )

        let length: Int = if case let .int(l) = arguments["length"]! {
            l
        } else {
            255
        }

        let killwords = arguments["killwords"]!.isTruthy

        let end: String = if case let .string(e) = arguments["end"]! {
            e
        } else {
            "..."
        }

        if str.count <= length {
            return .string(str)
        }

        if killwords {
            return .string(str.prefix(length) + end)
        } else {
            let truncated = str.prefix(length)
            if let lastSpace = truncated.lastIndex(where: { $0.isWhitespace }) {
                return .string(truncated[..<lastSpace] + end)
            } else {
                return .string(truncated + end)
            }
        }
    }

    /// Returns unique items from an array.
    @Sendable public static func unique(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first, case let .array(items) = value else {
            return .array([])
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        var seen = Set<Value>()
        var result = [Value]()
        for item in items {
            if !seen.contains(item) {
                seen.insert(item)
                result.append(item)
            }
        }
        return .array(result)
    }

    /// Indents text.
    @Sendable public static func indent(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .string(str) = args.first else {
            return .string("")
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["width", "first", "blank"],
            defaults: ["width": .int(4), "first": .boolean(false), "blank": .boolean(false)]
        )

        let widthString: String = switch arguments["width"] {
        case let .int(n):
            String(repeating: " ", count: n)
        case let .string(s):
            s
        default:
            "    "
        }

        let first = arguments["first"]!.isTruthy
        let blank = arguments["blank"]!.isTruthy

        let lines = str.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var result = ""

        for (i, line) in lines.enumerated() {
            if i == 0, !first {
                result += line
            } else if line.isEmpty, !blank {
                result += line
            } else {
                result += widthString + line
            }
            if i < lines.count - 1 {
                result += "\n"
            }
        }
        return .string(result)
    }

    /// Returns items (key-value pairs) of a dictionary/object.
    @Sendable public static func items(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else {
            return .array([])
        }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        if case let .object(obj) = value {
            let pairs = obj.map { key, value in
                Value.array([.string(key), value])
            }
            return .array(pairs)
        }

        return .array([])
    }

    /// Pretty prints a variable (useful for debugging).
    @Sendable public static func pprint(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard let value = args.first else { return .string("") }

        _ = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: [],
            defaults: [:]
        )

        func prettyPrint(_ val: Value, indent: Int = 0) -> String {
            let indentString = String(repeating: "  ", count: indent)
            switch val {
            case let .array(arr):
                if arr.isEmpty { return "[]" }
                let items = arr.map { prettyPrint($0, indent: indent + 1) }
                return "[\n" + items.map { "\(indentString)  \($0)" }.joined(separator: ",\n")
                    + "\n\(indentString)]"
            case let .object(dict):
                if dict.isEmpty { return "{}" }
                let items = dict.map { key, value in
                    "\(indentString)  \"\(key)\": \(prettyPrint(value, indent: indent + 1))"
                }
                return "{\n" + items.joined(separator: ",\n") + "\n\(indentString)}"
            case let .string(str):
                return "\"\(str)\""
            default:
                return val.description
            }
        }

        return .string(prettyPrint(value))
    }

    /// Converts URLs in text into clickable links.
    @Sendable public static func urlize(
        _ args: [Value], kwargs: [String: Value] = [:], env: Environment
    ) throws -> Value {
        guard case let .string(text) = args.first else {
            return .string("")
        }

        let arguments = try resolveCallArguments(
            args: Array(args.dropFirst()),
            kwargs: kwargs,
            parameters: ["trim_url_limit", "nofollow", "target", "rel"],
            defaults: [
                "trim_url_limit": .null,
                "nofollow": .boolean(false),
                "target": .null,
                "rel": .null,
            ]
        )

        let trimUrlLimit: Int? = if case let .int(limit)? = arguments["trim_url_limit"] {
            limit
        } else {
            nil
        }

        let nofollow = arguments["nofollow"]!.isTruthy

        let target: String? = if case let .string(t)? = arguments["target"] {
            t
        } else {
            nil
        }

        let rel: String? = if case let .string(r)? = arguments["rel"] {
            r
        } else {
            nil
        }

        func buildAttributes() -> String {
            var attributes = ""
            if nofollow { attributes += " rel=\"nofollow\"" }
            if let target { attributes += " target=\"\(target)\"" }
            if let rel { attributes += " rel=\"\(rel)\"" }
            return attributes
        }

        // Use URLComponents to detect and validate URLs
        var result = text
        let words = text.components(separatedBy: .whitespacesAndNewlines)

        for word in words {
            // Check if the word looks like a URL
            guard word.hasPrefix("http://") || word.hasPrefix("https://") else { continue }

            // Validate using URLComponents
            guard let urlComponents = URLComponents(string: word),
                  let scheme = urlComponents.scheme,
                  scheme == "http" || scheme == "https",
                  urlComponents.host != nil
            else { continue }

            let url = word
            let displayUrl =
                trimUrlLimit != nil && url.count > trimUrlLimit!
                    ? String(url.prefix(trimUrlLimit!)) + "..." : url
            let replacement = "<a href=\"\(url)\"\(buildAttributes())>\(displayUrl)</a>"
            result = result.replacingOccurrences(of: word, with: replacement)
        }

        return .string(result)
    }

    /// Dictionary of all built-in filters available for use in templates.
    ///
    /// Each filter function accepts an array of values (with the input as the first element),
    /// optional keyword arguments, and the current environment, then returns a transformed value.
    public static let builtIn:
        [String: @Sendable ([Value], [String: Value], Environment) throws -> Value] = [
            "upper": upper,
            "lower": lower,
            "length": length,
            "count": length, // alias for length
            "join": join,
            "default": `default`,
            "first": first,
            "last": last,
            "random": random,
            "reverse": reverse,
            "sort": sort,
            "groupby": groupby,
            "slice": slice,
            "map": map,
            "select": select,
            "reject": reject,
            "selectattr": selectattr,
            "rejectattr": rejectattr,
            "attr": attr,
            "dictsort": dictsort,
            "forceescape": forceescape,
            "safe": safe,
            "striptags": striptags,
            "format": format,
            "wordwrap": wordwrap,
            "filesizeformat": filesizeformat,
            "xmlattr": xmlattr,
            "string": string,
            "trim": trim,
            "escape": escape,
            "e": escape, // alias for escape
            "tojson": tojson,
            "abs": abs,
            "capitalize": capitalize,
            "center": center,
            "float": float,
            "int": int,
            "list": list,
            "max": max,
            "min": min,
            "round": round,
            "title": title,
            "wordcount": wordcount,
            "replace": replace,
            "urlencode": urlencode,
            "batch": batch,
            "sum": sum,
            "truncate": truncate,
            "unique": unique,
            "indent": indent,
            "items": items,
            "pprint": pprint,
            "urlize": urlize,
        ]
}
