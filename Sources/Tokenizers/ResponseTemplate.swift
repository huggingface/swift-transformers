//
//  ResponseTemplate.swift
//  swift-transformers
//
//  Declarative spec that drives `ResponseParser`. Counterpart of chat templates:
//  chat templates turn structured messages into a token stream; response templates
//  turn a model's token stream back into a structured message.
//

import Foundation
import Hub

/// Errors raised while loading a `ResponseTemplate` or while parsing a response.
public enum ResponseParserError: Error, Sendable, Equatable {
    case unsupportedVersion(Int)
    case invalidSpec(String)
    case invalidRegex(field: String, pattern: String, message: String)
    case unknownContentParser(name: String, field: String)
    case missingRequiredFields([String])
    case missingTransform(field: String)
    case transformUnavailable(field: String)
    case contentParseFailed(field: String, reason: String)
}

/// A JSON-shaped value tree — the type a parsed field collapses into.
public enum ParsedValue: Sendable, Hashable {
    case null
    case bool(Bool)
    case int(Int64)
    case double(Double)
    case string(String)
    case array([ParsedValue])
    case object([String: ParsedValue])
}

public extension ParsedValue {
    /// Structurally empty: `.null`, or an empty string / array / object.
    /// Primitives like `.bool(false)` and `.int(0)` are not considered empty.
    var isEmpty: Bool {
        switch self {
        case .null: return true
        case .string(let s): return s.isEmpty
        case .array(let arr): return arr.isEmpty
        case .object(let dict): return dict.isEmpty
        case .bool, .int, .double: return false
        }
    }
}

/// The structured output of a parse: keys are field names declared in the template.
public typealias ParsedMessage = [String: ParsedValue]

/// A region boundary — either a literal alternation or a regex pattern.
public struct ResponseAnchor: @unchecked Sendable {
    /// Compiled boundary. `NSRegularExpression` is documented as thread-safe.
    let regex: NSRegularExpression
    /// Non-nil iff the anchor was specified as a literal (or list of literals).
    let literals: [String]?
    /// True iff one literal in the list is a strict prefix of another, in which
    /// case a match at the buffer edge could still grow with more input.
    let literalCanExtend: Bool
    /// Named capture group names declared in the pattern (always empty for literal anchors).
    let namedGroups: [String]
}

/// A single field declared in a `ResponseTemplate`.
public struct ResponseField: Sendable {
    public let name: String
    /// Pattern that opens the region. `nil` marks this as the implicit (leftover-sink) field.
    public let open: ResponseAnchor?
    /// Pattern that closes the region. May be `nil` for an implicit field with no terminator.
    public let close: ResponseAnchor?
    public let content: ContentParserSpec
    public let repeats: Bool
    public let optional: Bool
    /// Optional jmespath-style expression evaluated against `{...captures, "content": value}`
    /// after the content parser runs. Resolved by a caller-supplied closure.
    public let transform: String?
}

public struct ResponseTemplate: @unchecked Sendable {
    public let defaults: ParsedMessage
    public let fields: [String: ResponseField]
    /// Name of the implicit field (the one without an `open`/`open_pattern`), if any.
    public let implicit: String?
    public let startAnchor: ResponseAnchor?

    /// Load a template from a `response_template` Config (as nested in `tokenizer_config.json`).
    public init(from config: Config) throws {
        try self.init(spec: try Self.configToAny(config))
    }

    /// Load a template from a plain spec dictionary (test/inline use).
    public init(spec: [String: Any]) throws {
        let allowedTopLevel: Set<String> = ["version", "defaults", "fields", "start_anchor", "start_anchor_pattern"]
        for key in spec.keys where !allowedTopLevel.contains(key) {
            throw ResponseParserError.invalidSpec("Unknown response_template key: '\(key)'")
        }

        let version = (spec["version"] as? Int) ?? 1
        guard version == 1 else { throw ResponseParserError.unsupportedVersion(version) }

        let defaultsRaw = spec["defaults"] as? [String: Any] ?? [:]
        var defaults: ParsedMessage = [:]
        for (k, v) in defaultsRaw {
            defaults[k] = Self.anyToParsedValue(v)
        }

        guard let fieldsRaw = spec["fields"] as? [String: Any], !fieldsRaw.isEmpty else {
            throw ResponseParserError.invalidSpec("response_template.fields must be a non-empty dict")
        }

        var fields: [String: ResponseField] = [:]
        var implicitFields: [String] = []
        for (name, fieldAny) in fieldsRaw {
            guard let fieldDict = fieldAny as? [String: Any] else {
                throw ResponseParserError.invalidSpec("Field '\(name)' must be a dict")
            }

            let allowedFieldKeys: Set<String> = [
                "open", "open_pattern", "close", "close_pattern",
                "content", "content_args", "repeats", "optional", "transform",
            ]
            for k in fieldDict.keys where !allowedFieldKeys.contains(k) {
                throw ResponseParserError.invalidSpec("Field '\(name)': unknown key '\(k)'")
            }

            let contentName = (fieldDict["content"] as? String) ?? "text"
            guard let contentKind = ContentParserKind(rawValue: contentName) else {
                throw ResponseParserError.unknownContentParser(name: contentName, field: name)
            }
            let contentArgs = (fieldDict["content_args"] as? [String: Any]) ?? [:]
            let contentSpec = ContentParserSpec(kind: contentKind, args: contentArgs)

            let openAnchor = try Self.compileAnchor(scope: "Field '\(name)'", field: fieldDict, litKey: "open", patKey: "open_pattern")
            let closeAnchor = try Self.compileAnchor(scope: "Field '\(name)'", field: fieldDict, litKey: "close", patKey: "close_pattern")

            if openAnchor == nil {
                implicitFields.append(name)
            }

            let transform = fieldDict["transform"] as? String
            if transform == nil {
                var capturedNames: Set<String> = []
                if let openAnchor { capturedNames.formUnion(Self.namedGroupKeys(in: openAnchor.regex)) }
                if let closeAnchor { capturedNames.formUnion(Self.namedGroupKeys(in: closeAnchor.regex)) }
                if !capturedNames.isEmpty {
                    throw ResponseParserError.invalidSpec(
                        "Field '\(name)': named capture group(s) \(capturedNames.sorted()) declared but no 'transform'"
                    )
                }
            }

            fields[name] = ResponseField(
                name: name,
                open: openAnchor,
                close: closeAnchor,
                content: contentSpec,
                repeats: (fieldDict["repeats"] as? Bool) ?? false,
                optional: (fieldDict["optional"] as? Bool) ?? true,
                transform: transform
            )
        }

        if implicitFields.count > 1 {
            throw ResponseParserError.invalidSpec(
                "At most one field may omit 'open'/'open_pattern'; found: \(implicitFields.joined(separator: ", "))"
            )
        }

        let startAnchor = try Self.compileAnchor(scope: "response_template", field: spec, litKey: "start_anchor", patKey: "start_anchor_pattern")

        self.defaults = defaults
        self.fields = fields
        self.implicit = implicitFields.first
        self.startAnchor = startAnchor
    }

    /// Right-truncate `text` to the position right after the last `start_anchor` match,
    /// or return it unchanged if no anchor is configured.
    func truncatePastLastAnchor(_ text: String) -> String {
        guard let anchor = startAnchor else { return text }
        let ns = text as NSString
        let full = NSRange(location: 0, length: ns.length)
        var lastEnd: Int? = nil
        anchor.regex.enumerateMatches(in: text, options: [], range: full) { match, _, _ in
            if let m = match { lastEnd = m.range.location + m.range.length }
        }
        guard let end = lastEnd else { return text }
        return ns.substring(from: end)
    }

    // MARK: - Helpers

    private static func compileAnchor(scope: String, field: [String: Any], litKey: String, patKey: String) throws -> ResponseAnchor? {
        let hasLit = field[litKey] != nil
        let hasPat = field[patKey] != nil
        if hasLit && hasPat {
            throw ResponseParserError.invalidSpec("\(scope): cannot specify both '\(litKey)' and '\(patKey)'")
        }
        if hasLit {
            let raw = field[litKey]!
            var lits: [String]
            if let s = raw as? String {
                lits = [s]
            } else if let arr = raw as? [Any] {
                if arr.isEmpty {
                    throw ResponseParserError.invalidSpec("\(scope): '\(litKey)' list must contain at least one literal")
                }
                lits = []
                for item in arr {
                    guard let s = item as? String else {
                        throw ResponseParserError.invalidSpec("\(scope): '\(litKey)' list must contain only strings")
                    }
                    lits.append(s)
                }
                // Dedupe preserving first-seen order.
                var seen: Set<String> = []
                lits = lits.filter { seen.insert($0).inserted }
            } else {
                throw ResponseParserError.invalidSpec("\(scope): '\(litKey)' must be a string or list of strings")
            }
            if lits.contains(where: { $0.isEmpty }) {
                throw ResponseParserError.invalidSpec("\(scope): '\(litKey)' literals cannot be empty strings")
            }
            // "eos" magic literal → end-of-stream anchor.
            if lits.contains("eos") {
                if lits.count > 1 {
                    throw ResponseParserError.invalidSpec("\(scope): the 'eos' literal cannot be combined with other literals")
                }
                let regex = try NSRegularExpression(pattern: #"\z"#, options: [.dotMatchesLineSeparators])
                return ResponseAnchor(regex: regex, literals: lits, literalCanExtend: false, namedGroups: [])
            }
            // Longest-first so the alternation prefers the longer alternative on ties.
            let ordered = lits.sorted { $0.count > $1.count }
            var canExtend = false
            for a in lits {
                for b in lits where a != b {
                    if a.hasPrefix(b) { canExtend = true }
                }
            }
            let pattern = ordered.map { NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "|")
            let regex: NSRegularExpression
            do {
                regex = try NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators])
            } catch {
                throw ResponseParserError.invalidRegex(field: scope, pattern: pattern, message: "\(error)")
            }
            return ResponseAnchor(regex: regex, literals: lits, literalCanExtend: canExtend, namedGroups: [])
        }
        if hasPat {
            guard let pat = field[patKey] as? String else {
                throw ResponseParserError.invalidSpec("\(scope): '\(patKey)' must be a string")
            }
            // Templates are commonly authored against Python's `re`, which
            // uses `(?P<name>...)` for named captures. ICU (NSRegularExpression)
            // uses `(?<name>...)`. Normalize so authors can write either.
            let normalized = Self.normalizePythonNamedGroups(pat)
            do {
                let regex = try NSRegularExpression(pattern: normalized, options: [.dotMatchesLineSeparators])
                let groups = Self.namedGroupKeys(in: regex)
                return ResponseAnchor(regex: regex, literals: nil, literalCanExtend: false, namedGroups: Array(groups))
            } catch {
                throw ResponseParserError.invalidRegex(field: scope, pattern: pat, message: "\(error)")
            }
        }
        return nil
    }

    /// Rewrite Python-style named captures `(?P<name>...)` to ICU style
    /// `(?<name>...)`. NSRegularExpression accepts only the latter, but
    /// templates are often authored against Python's `re`. Leaves
    /// `(?P=name)` backreferences alone — also Python-only, but ICU has no
    /// direct equivalent, so we'd surface that as a normal regex error.
    private static func normalizePythonNamedGroups(_ pattern: String) -> String {
        guard let scanner = try? NSRegularExpression(pattern: #"\(\?P<"#) else { return pattern }
        let ns = pattern as NSString
        let range = NSRange(location: 0, length: ns.length)
        return scanner.stringByReplacingMatches(in: pattern, options: [], range: range, withTemplate: "(?<")
    }

    /// Best-effort extraction of named capture group names from a compiled regex.
    /// NSRegularExpression doesn't expose them directly, so we re-scan the pattern.
    private static func namedGroupKeys(in regex: NSRegularExpression) -> Set<String> {
        var out: Set<String> = []
        let pat = regex.pattern as NSString
        let scanner = try? NSRegularExpression(pattern: #"\(\?P?<([A-Za-z_][A-Za-z0-9_]*)>"#)
        guard let scanner else { return [] }
        let range = NSRange(location: 0, length: pat.length)
        scanner.enumerateMatches(in: regex.pattern, options: [], range: range) { m, _, _ in
            guard let m, m.numberOfRanges >= 2 else { return }
            let r = m.range(at: 1)
            if r.location != NSNotFound { out.insert(pat.substring(with: r)) }
        }
        return out
    }

    /// Bridge a `Config` to the plain `[String: Any]` shape `init(spec:)` expects.
    private static func configToAny(_ config: Config) throws -> [String: Any] {
        guard let dict = config.dictionary() else {
            throw ResponseParserError.invalidSpec("response_template must be a dictionary")
        }
        var spec: [String: Any] = [:]
        for (k, v) in dict {
            spec[k.string] = configValueToAny(v)
        }
        return spec
    }

    private static func configValueToAny(_ value: Config) -> Any {
        if value.isNull() { return NSNull() }
        if let s = value.string() { return s }
        if let b = value.boolean() { return b }
        if let i = value.integer() { return i }
        if let f = value.floating() { return f }
        if let arr = value.array() { return arr.map(configValueToAny) }
        if let dict = value.dictionary() {
            var out: [String: Any] = [:]
            for (k, v) in dict { out[k.string] = configValueToAny(v) }
            return out
        }
        return NSNull()
    }

    static func anyToParsedValue(_ any: Any) -> ParsedValue {
        if any is NSNull { return .null }
        if let s = any as? String { return .string(s) }
        if let b = any as? Bool { return .bool(b) }
        if let i = any as? Int { return .int(Int64(i)) }
        if let i = any as? Int64 { return .int(i) }
        if let d = any as? Double { return .double(d) }
        if let f = any as? Float { return .double(Double(f)) }
        if let arr = any as? [Any] { return .array(arr.map(anyToParsedValue)) }
        if let dict = any as? [String: Any] {
            var out: [String: ParsedValue] = [:]
            for (k, v) in dict { out[k] = anyToParsedValue(v) }
            return .object(out)
        }
        return .null
    }
}
