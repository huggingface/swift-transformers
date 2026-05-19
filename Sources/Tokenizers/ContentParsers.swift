//
//  ContentParsers.swift
//  swift-transformers
//
//  Per-field content parsers. Each parser takes a captured region body and
//  collapses it into a single `ParsedValue`. Fully agent-portable from the
//  Python implementation; the behaviour matches one-to-one.
//

import Foundation

public enum ContentParserKind: String, Sendable {
    case text
    case int
    case float
    case bool
    case json
    case xmlInline = "xml-inline"
    case kvLines = "kv-lines"
}

public struct ContentParserSpec: @unchecked Sendable {
    public let kind: ContentParserKind
    public let args: [String: Any]

    public init(kind: ContentParserKind, args: [String: Any] = [:]) {
        self.kind = kind
        self.args = args
    }
}

/// Parsers whose chunks contribute verbatim to the final value: streaming
/// consumers can render them as they arrive (`dirty = false`). Structured
/// parsers (`json`, `xml-inline`, `kv-lines`) only yield a meaningful value on
/// close, so their chunks stream raw bytes with `dirty = true` and the final
/// `ParsedValue` is delivered via `region_close`.
let streamableParsers: Set<ContentParserKind> = [.text, .int, .float, .bool]

/// Closure type for resolving optional `transform` expressions (e.g. jmespath).
/// Receives the expression source and a context dict (`{...captures, "content": value}`).
public typealias ResponseTransform = @Sendable (_ expression: String, _ context: [String: ParsedValue]) throws -> ParsedValue

/// Error thrown by transform resolvers when they don't recognize an expression.
public struct UnsupportedTransform: Error, CustomStringConvertible {
    public let expression: String
    public init(_ expression: String) { self.expression = expression }
    public var description: String { "Unsupported transform expression: \(expression)" }
}

/// A namespace for common `ResponseTransform` implementations without a jmespath evaluator.
public enum ResponseTransforms {
    public static let builtin: ResponseTransform = { expression, context in
        switch expression {
        // JSON body contains `name` and `arguments` (Hermes-like, SmolLM3)
        case "{type: 'function', function: content}":
            return .object([
                "type": .string("function"),
                "function": context["content"] ?? .null,
            ])
        // Gemma 4 / Qwen 3 style: function name comes from a named capture,
        // `arguments` from parsed JSON payload.
        case "{type: 'function', function: {name: name, arguments: content}}":
            return .object([
                "type": .string("function"),
                "function": .object([
                    "name": context["name"] ?? .null,
                    "arguments": context["content"] ?? .null,
                ]),
            ])
        default:
            throw UnsupportedTransform(expression)
        }
    }
}

func parseContent(_ text: String, spec: ContentParserSpec, fieldName: String) throws -> ParsedValue {
    switch spec.kind {
    case .text: return parseText(text, args: spec.args)
    case .int: return try parseInt(text, args: spec.args, fieldName: fieldName)
    case .float: return try parseFloat(text, args: spec.args, fieldName: fieldName)
    case .bool: return parseBool(text, args: spec.args)
    case .json: return try parseJSON(text, args: spec.args, fieldName: fieldName)
    case .xmlInline: return try parseXMLInline(text, args: spec.args, fieldName: fieldName)
    case .kvLines: return try parseKVLines(text, args: spec.args, fieldName: fieldName)
    }
}

func processField(
    body: String,
    field: ResponseField,
    captures: [String: String],
    transform: ResponseTransform?
) throws -> ParsedValue {
    let value = try parseContent(body, spec: field.content, fieldName: field.name)
    return try applyTransform(field: field, captures: captures, content: value, transform: transform)
}

private func applyTransform(
    field: ResponseField,
    captures: [String: String],
    content: ParsedValue,
    transform: ResponseTransform?
) throws -> ParsedValue {
    guard let expr = field.transform else { return content }
    guard let transform else {
        throw ResponseParserError.transformUnavailable(field: field.name)
    }
    var context: [String: ParsedValue] = [:]
    for (k, v) in captures { context[k] = .string(v) }
    context["content"] = content
    return try transform(expr, context)
}

// MARK: - text / int / float / bool

private func parseText(_ text: String, args: [String: Any]) -> ParsedValue {
    let strip = (args["strip"] as? Bool) ?? true
    return .string(strip ? text.trimmingCharacters(in: .whitespacesAndNewlines) : text)
}

private func parseInt(_ text: String, args: [String: Any], fieldName: String) throws -> ParsedValue {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let value = Int64(trimmed) else {
        throw ResponseParserError.contentParseFailed(field: fieldName, reason: "could not parse '\(trimmed)' as int")
    }
    return .int(value)
}

private func parseFloat(_ text: String, args: [String: Any], fieldName: String) throws -> ParsedValue {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let value = Double(trimmed) else {
        throw ResponseParserError.contentParseFailed(field: fieldName, reason: "could not parse '\(trimmed)' as float")
    }
    return .double(value)
}

private func parseBool(_ text: String, args: [String: Any]) -> ParsedValue {
    let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    return .bool(normalized == "true" || normalized == "1")
}

// MARK: - json (with optional dialect knobs)

/// Sentinel characters used to pre-extract custom-delimited strings before JSON parsing.
private let laxOpen = "\u{0001}"
private let laxClose = "\u{0002}"

private func parseJSON(_ text: String, args: [String: Any], fieldName: String) throws -> ParsedValue {
    var stringDelims: [(String, String)] = []
    if let delimsAny = args["string_delims"] as? [Any] {
        for entry in delimsAny {
            if let pair = entry as? [String], pair.count == 2 {
                stringDelims.append((pair[0], pair[1]))
            } else if let pair = entry as? [Any], pair.count == 2,
                      let open = pair[0] as? String, let close = pair[1] as? String
            {
                stringDelims.append((open, close))
            }
        }
    }
    let unquotedKeys = (args["unquoted_keys"] as? Bool) ?? false
    let allowNonJSON = (args["allow_non_json"] as? Bool) ?? false

    if !stringDelims.isEmpty, text.contains(laxOpen) || text.contains(laxClose) {
        throw ResponseParserError.contentParseFailed(
            field: fieldName,
            reason: "input contains reserved sentinel characters (\\x01/\\x02); cannot parse safely"
        )
    }

    var working = text
    var captured: [String] = []
    for (openD, closeD) in stringDelims {
        let escOpen = NSRegularExpression.escapedPattern(for: openD)
        let escClose = NSRegularExpression.escapedPattern(for: closeD)
        let pattern = "\(escOpen)(.*?)\(escClose)"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) else { continue }
        working = substitute(in: working, regex: regex) { match, ns in
            guard match.numberOfRanges >= 2 else { return nil }
            let r = match.range(at: 1)
            guard r.location != NSNotFound else { return nil }
            captured.append(ns.substring(with: r))
            return "\(laxOpen)\(captured.count - 1)\(laxClose)"
        }
    }

    if unquotedKeys {
        if let regex = try? NSRegularExpression(pattern: #"(?<=[{,])(\w+):"#, options: []) {
            working = substitute(in: working, regex: regex) { match, ns in
                guard match.numberOfRanges >= 2 else { return nil }
                let key = ns.substring(with: match.range(at: 1))
                return "\"\(key)\":"
            }
        }
    }

    for (i, s) in captured.enumerated() {
        let placeholder = "\(laxOpen)\(i)\(laxClose)"
        if let encoded = jsonEncodeString(s) {
            working = working.replacingOccurrences(of: placeholder, with: encoded)
        }
    }

    guard let data = working.data(using: .utf8) else {
        if allowNonJSON {
            return .string(text.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        throw ResponseParserError.contentParseFailed(field: fieldName, reason: "could not encode JSON working text as utf-8")
    }
    do {
        let object = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        return jsonObjectToParsedValue(object)
    } catch {
        if allowNonJSON {
            return .string(text.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        throw ResponseParserError.contentParseFailed(
            field: fieldName,
            reason: "JSON parse failed: \(error.localizedDescription); content: \(text)"
        )
    }
}

private func jsonEncodeString(_ s: String) -> String? {
    guard let data = try? JSONSerialization.data(withJSONObject: [s], options: [.fragmentsAllowed]) else { return nil }
    guard let str = String(data: data, encoding: .utf8) else { return nil }
    // Strip the surrounding "[" and "]" to leave a quoted JSON string.
    let trimmed = str.dropFirst().dropLast()
    return String(trimmed)
}

private func jsonObjectToParsedValue(_ object: Any) -> ParsedValue {
    if object is NSNull { return .null }
    if let n = object as? NSNumber {
        // NSNumber doesn't preserve "was this a bool" cleanly; check the CFType.
        if CFGetTypeID(n) == CFBooleanGetTypeID() {
            return .bool(n.boolValue)
        }
        if let asInt = Int64(exactly: n.doubleValue), asInt == n.int64Value, Double(asInt) == n.doubleValue {
            return .int(asInt)
        }
        return .double(n.doubleValue)
    }
    if let s = object as? String { return .string(s) }
    if let arr = object as? [Any] { return .array(arr.map(jsonObjectToParsedValue)) }
    if let dict = object as? [String: Any] {
        var out: [String: ParsedValue] = [:]
        for (k, v) in dict { out[k] = jsonObjectToParsedValue(v) }
        return .object(out)
    }
    return .null
}

// MARK: - xml-inline

private func parseXMLInline(_ text: String, args: [String: Any], fieldName: String) throws -> ParsedValue {
    guard let tagPattern = args["tag_pattern"] as? String else {
        throw ResponseParserError.contentParseFailed(field: fieldName, reason: "xml-inline: 'tag_pattern' is required")
    }
    let valueParser = args["value_parser"] as? [String: Any]
    let mergeDuplicates = (args["merge_duplicates"] as? Bool) ?? false

    let regex: NSRegularExpression
    do {
        regex = try NSRegularExpression(pattern: tagPattern, options: [.dotMatchesLineSeparators])
    } catch {
        throw ResponseParserError.contentParseFailed(
            field: fieldName,
            reason: "xml-inline: invalid tag_pattern '\(tagPattern)': \(error.localizedDescription)"
        )
    }

    var out: [String: ParsedValue] = [:]
    var encounteredError: Error?
    let ns = text as NSString
    let full = NSRange(location: 0, length: ns.length)
    regex.enumerateMatches(in: text, options: [], range: full) { match, _, stop in
        guard let m = match, encounteredError == nil else { return }
        let keyRange = m.range(withName: "key")
        guard keyRange.location != NSNotFound else {
            encounteredError = ResponseParserError.contentParseFailed(
                field: fieldName,
                reason: "xml-inline: tag_pattern must have a named group 'key'"
            )
            stop.pointee = true
            return
        }
        let key = ns.substring(with: keyRange)
        let valueRange = m.range(withName: "value")
        let rawValue: String
        if valueRange.location != NSNotFound {
            rawValue = ns.substring(with: valueRange)
        } else {
            rawValue = ""
        }
        let parsedValue: ParsedValue
        do {
            parsedValue = try subParse(rawValue, spec: valueParser, fieldName: fieldName)
        } catch {
            encounteredError = error
            stop.pointee = true
            return
        }

        if let existing = out[key], mergeDuplicates {
            if case .array(var list) = existing {
                list.append(parsedValue)
                out[key] = .array(list)
            } else {
                out[key] = .array([existing, parsedValue])
            }
        } else {
            out[key] = parsedValue
        }
    }
    if let encounteredError { throw encounteredError }
    return .object(out)
}

// MARK: - kv-lines

private func parseKVLines(_ text: String, args: [String: Any], fieldName: String) throws -> ParsedValue {
    let lineSep = (args["line_sep"] as? String) ?? "\n"
    let kvSep = (args["kv_sep"] as? String) ?? ":"
    let valueParser = args["value_parser"] as? [String: Any]
    let strip = (args["strip"] as? Bool) ?? true

    var out: [String: ParsedValue] = [:]
    for rawLine in text.components(separatedBy: lineSep) {
        var line = rawLine
        if strip { line = line.trimmingCharacters(in: .whitespacesAndNewlines) }
        if line.isEmpty || !line.contains(kvSep) { continue }
        guard let sepRange = line.range(of: kvSep) else { continue }
        var k = String(line[..<sepRange.lowerBound])
        var v = String(line[sepRange.upperBound...])
        if strip {
            k = k.trimmingCharacters(in: .whitespacesAndNewlines)
            v = v.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        out[k] = try subParse(v, spec: valueParser, fieldName: fieldName)
    }
    return .object(out)
}

// MARK: - shared helpers

private func subParse(_ raw: String, spec: [String: Any]?, fieldName: String) throws -> ParsedValue {
    guard let spec else { return .string(raw) }
    let name = (spec["name"] as? String) ?? "text"
    guard let kind = ContentParserKind(rawValue: name) else {
        throw ResponseParserError.unknownContentParser(name: name, field: fieldName)
    }
    let args = (spec["args"] as? [String: Any]) ?? [:]
    return try parseContent(raw, spec: ContentParserSpec(kind: kind, args: args), fieldName: fieldName)
}

private func substitute(
    in text: String,
    regex: NSRegularExpression,
    using replace: (NSTextCheckingResult, NSString) -> String?
) -> String {
    let ns = text as NSString
    let full = NSRange(location: 0, length: ns.length)
    let matches = regex.matches(in: text, options: [], range: full)
    if matches.isEmpty { return text }

    var out = ""
    var cursor = 0
    for m in matches {
        let r = m.range
        if r.location > cursor {
            out += ns.substring(with: NSRange(location: cursor, length: r.location - cursor))
        }
        if let replacement = replace(m, ns) {
            out += replacement
        } else {
            out += ns.substring(with: r)
        }
        cursor = r.location + r.length
    }
    if cursor < ns.length {
        out += ns.substring(with: NSRange(location: cursor, length: ns.length - cursor))
    }
    return out
}
