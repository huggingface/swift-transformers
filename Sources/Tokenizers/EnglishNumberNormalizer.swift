//
//  EnglishNumberNormalizer.swift
//
//  Converts spelled-out English numbers ("twenty-seven", "five hundred and
//  twenty-three") into their arabic-numeral form. Mirrors
//  `whisper.normalizers.EnglishNumberNormalizer` from openai/whisper.
//
//  The implementation is a faithful Swift port of the Python state machine:
//  - `process(words:)` ↔ `process_words`
//  - `preprocess` ↔ `preprocess`
//  - `postprocess` ↔ `postprocess`
//
//  Internal numeric state is kept as `Int` (64-bit). All multipliers up to
//  quintillion (10^18) fit; sextillion (10^21) and above would overflow,
//  but they do not appear in real ASR evaluation corpora.
//

import Foundation

public struct EnglishNumberNormalizer: Sendable {
    public init() {}

    // MARK: - Tables (built once at type init)

    static let zeros: Set<String> = ["o", "oh", "zero"]

    static let onesList: [String] = [
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]

    static let ones: [String: Int] = {
        var d = [String: Int]()
        for (i, name) in onesList.enumerated() { d[name] = i + 1 }
        return d
    }()

    static let onesPlural: [String: (value: Int, suffix: String)] = {
        var d = [String: (Int, String)]()
        for (name, value) in ones {
            let plural = (name == "six") ? "sixes" : name + "s"
            d[plural] = (value, "s")
        }
        return d
    }()

    static let onesOrdinal: [String: (value: Int, suffix: String)] = {
        var d: [String: (Int, String)] = [
            "zeroth": (0, "th"),
            "first": (1, "st"),
            "second": (2, "nd"),
            "third": (3, "rd"),
            "fifth": (5, "th"),
            "twelfth": (12, "th"),
        ]
        for (name, value) in ones where value > 3 && value != 5 && value != 12 {
            let key = name + (name.hasSuffix("t") ? "h" : "th")
            d[key] = (value, "th")
        }
        return d
    }()

    static let onesSuffixed: [String: (value: Int, suffix: String)] = {
        var d = onesPlural
        for (k, v) in onesOrdinal { d[k] = v }
        return d
    }()

    static let tens: [String: Int] = [
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    ]

    static let tensPlural: [String: (value: Int, suffix: String)] = {
        var d = [String: (Int, String)]()
        for (name, value) in tens {
            // "twenty" → "twenties", "thirty" → "thirties" ...
            let key = name.replacingOccurrences(of: "y", with: "ies")
            d[key] = (value, "s")
        }
        return d
    }()

    static let tensOrdinal: [String: (value: Int, suffix: String)] = {
        var d = [String: (Int, String)]()
        for (name, value) in tens {
            // "twenty" → "twentieth", "thirty" → "thirtieth" ...
            let key = name.replacingOccurrences(of: "y", with: "ieth")
            d[key] = (value, "th")
        }
        return d
    }()

    static let tensSuffixed: [String: (value: Int, suffix: String)] = {
        var d = tensPlural
        for (k, v) in tensOrdinal { d[k] = v }
        return d
    }()

    // Sextillion (10^21) and above overflow Int64; they are uncommon in
    // ASR text, so we stop at quintillion to avoid an arbitrary-precision
    // dependency in the normalizer.
    static let multipliers: [String: Int] = [
        "hundred": 100,
        "thousand": 1_000,
        "million": 1_000_000,
        "billion": 1_000_000_000,
        "trillion": 1_000_000_000_000,
        "quadrillion": 1_000_000_000_000_000,
        "quintillion": 1_000_000_000_000_000_000,
    ]

    static let multipliersPlural: [String: (value: Int, suffix: String)] = {
        var d = [String: (Int, String)]()
        for (name, value) in multipliers { d[name + "s"] = (value, "s") }
        return d
    }()

    static let multipliersOrdinal: [String: (value: Int, suffix: String)] = {
        var d = [String: (Int, String)]()
        for (name, value) in multipliers { d[name + "th"] = (value, "th") }
        return d
    }()

    static let multipliersSuffixed: [String: (value: Int, suffix: String)] = {
        var d = multipliersPlural
        for (k, v) in multipliersOrdinal { d[k] = v }
        return d
    }()

    static let decimals: Set<String> = {
        var s = zeros
        for k in ones.keys { s.insert(k) }
        for k in tens.keys { s.insert(k) }
        return s
    }()

    static let precedingPrefixers: [String: String] = [
        "minus": "-", "negative": "-",
        "plus": "+", "positive": "+",
    ]

    static let followingPrefixers: [String: String] = [
        "pound": "£", "pounds": "£",
        "euro": "€", "euros": "€",
        "dollar": "$", "dollars": "$",
        "cent": "¢", "cents": "¢",
    ]

    static let prefixes: Set<Character> = {
        var s = Set<Character>()
        for v in precedingPrefixers.values { if let c = v.first { s.insert(c) } }
        for v in followingPrefixers.values { if let c = v.first { s.insert(c) } }
        return s
    }()

    enum Suffixer {
        case direct(String)
        case context([String: String])
    }

    static let suffixers: [String: Suffixer] = [
        "per": .context(["cent": "%"]),
        "percent": .direct("%"),
    ]

    static let specials: Set<String> = ["and", "double", "triple", "point"]

    static let words: Set<String> = {
        var s = Set<String>()
        s.formUnion(zeros)
        for k in ones.keys { s.insert(k) }
        for k in onesSuffixed.keys { s.insert(k) }
        for k in tens.keys { s.insert(k) }
        for k in tensSuffixed.keys { s.insert(k) }
        for k in multipliers.keys { s.insert(k) }
        for k in multipliersSuffixed.keys { s.insert(k) }
        for k in precedingPrefixers.keys { s.insert(k) }
        for k in followingPrefixers.keys { s.insert(k) }
        for k in suffixers.keys { s.insert(k) }
        s.formUnion(specials)
        return s
    }()

    // MARK: - Public entrypoint

    public func callAsFunction(_ s: String) -> String {
        normalize(s)
    }

    public func normalize(_ s: String) -> String {
        var out = preprocess(s)
        let outputWords = process(words: out.split(separator: " ").map(String.init))
        out = outputWords.joined(separator: " ")
        out = postprocess(out)
        return out
    }

    // MARK: - State machine

    enum NumValue: Equatable {
        case none
        case integer(Int)
        case string(String)

        var stringForm: String {
            switch self {
            case .none: return ""
            case .integer(let i): return String(i)
            case .string(let s): return s
            }
        }

        var isNone: Bool {
            if case .none = self { return true }
            return false
        }
    }

    /// Sliding state for `process(words:)`. Threading the state through
    /// a single `inout` parameter avoids the captured-closure exclusivity
    /// pitfall we would otherwise hit by passing `inout value` alongside
    /// an `emit` closure that also mutates `value`.
    struct State {
        var output: [String] = []
        var prefix: String? = nil
        var value: NumValue = .none
        var skip: Bool = false

        mutating func emit(_ result: String) {
            var s = result
            if let p = prefix {
                s = p + s
                prefix = nil
            }
            output.append(s)
            value = .none
        }

        mutating func emitValue() {
            switch value {
            case .none: return
            case .integer(let i): emit(String(i))
            case .string(let s): emit(s)
            }
        }
    }

    private static let arabicNumberRegex = CompiledRegex(#"^\d+(\.\d+)?$"#)

    /// Returns `true` if `s` matches `^\d+(\.\d+)?$`.
    private static func isArabicNumber(_ s: String) -> Bool {
        let range = NSRange(s.startIndex..<s.endIndex, in: s)
        return arabicNumberRegex.nsRegex.firstMatch(in: s, options: [], range: range) != nil
    }

    /// Returns the integer value when `s` is a whole arabic number ("23"),
    /// or `nil` for decimals ("23.5") and non-numbers.
    private static func wholeIntFromArabic(_ s: String) -> Int? {
        guard !s.contains(".") else { return nil }
        return Int(s)
    }

    /// Python: `for prev, current, next in windowed([None] + words + [None], 3)`.
    func process(words: [String]) -> [String] {
        guard !words.isEmpty else { return [] }

        var state = State()
        let padded: [String?] = [nil] + words.map(Optional.some) + [nil]

        for i in 0..<(padded.count - 2) {
            if state.skip {
                state.skip = false
                continue
            }
            let prev = padded[i]
            guard let current = padded[i + 1] else { continue }
            let next = padded[i + 2]

            let nextIsNumeric = (next != nil) && Self.isArabicNumber(next!)
            let hasPrefix = current.first.map { Self.prefixes.contains($0) } ?? false
            let currentWithoutPrefix = hasPrefix ? String(current.dropFirst()) : current

            // Branch: arabic number (possibly with sign/decimal)
            if Self.isArabicNumber(currentWithoutPrefix) {
                if !state.value.isNone {
                    if case .string(let s) = state.value, s.hasSuffix(".") {
                        // continue concatenating decimals / ip-address pieces
                        state.value = .string(s + current)
                        continue
                    } else {
                        state.emitValue()
                    }
                }
                if hasPrefix, let c = current.first { state.prefix = String(c) }
                if let whole = Self.wholeIntFromArabic(currentWithoutPrefix) {
                    state.value = .integer(whole)
                } else {
                    state.value = .string(currentWithoutPrefix)
                }
                continue
            }

            // Branch: non-numeric word
            if !Self.words.contains(current) {
                state.emitValue()
                state.emit(current)
                continue
            }

            // Branch: zero words ("o", "oh", "zero")
            if Self.zeros.contains(current) {
                state.value = .string(state.value.stringForm + "0")
                continue
            }

            // Branch: ones (one..nineteen)
            if let onesV = Self.ones[current] {
                processOnes(onesValue: onesV, prev: prev, state: &state)
                continue
            }

            // Branch: ones_suffixed (ordinal/cardinal, e.g. "fifth", "ones")
            if let entry = Self.onesSuffixed[current] {
                processOnesSuffixed(
                    onesValue: entry.value, suffix: entry.suffix,
                    prev: prev, state: &state)
                continue
            }

            // Branch: tens (twenty..ninety)
            if let tensV = Self.tens[current] {
                processTens(tensValue: tensV, state: &state)
                continue
            }

            // Branch: tens_suffixed
            if let entry = Self.tensSuffixed[current] {
                processTensSuffixed(
                    tensValue: entry.value, suffix: entry.suffix,
                    state: &state)
                continue
            }

            // Branch: multipliers (hundred, thousand, ...)
            if let mult = Self.multipliers[current] {
                processMultiplier(multiplier: mult, state: &state)
                continue
            }

            // Branch: multipliers_suffixed
            if let entry = Self.multipliersSuffixed[current] {
                processMultiplierSuffixed(
                    multiplier: entry.value, suffix: entry.suffix,
                    state: &state)
                continue
            }

            // Branch: preceding prefixers (minus, plus, ...)
            if let p = Self.precedingPrefixers[current] {
                state.emitValue()
                if (next.map { Self.words.contains($0) } ?? false) || nextIsNumeric {
                    state.prefix = p
                } else {
                    state.emit(current)
                }
                continue
            }

            // Branch: following prefixers (dollars, cents, ...)
            if let p = Self.followingPrefixers[current] {
                if !state.value.isNone {
                    state.prefix = p
                    state.emitValue()
                } else {
                    state.emit(current)
                }
                continue
            }

            // Branch: suffixers (percent)
            if let suffixer = Self.suffixers[current] {
                if !state.value.isNone {
                    switch suffixer {
                    case .direct(let suffix):
                        state.emit(state.value.stringForm + suffix)
                    case .context(let map):
                        if let next, let suffix = map[next] {
                            state.emit(state.value.stringForm + suffix)
                            state.skip = true
                        } else {
                            state.emitValue()
                            state.emit(current)
                        }
                    }
                } else {
                    state.emit(current)
                }
                continue
            }

            // Branch: specials (and, double, triple, point)
            if Self.specials.contains(current) {
                let nextNotInWords = !(next.map { Self.words.contains($0) } ?? false)
                if nextNotInWords && !nextIsNumeric {
                    state.emitValue()
                    state.emit(current)
                } else if current == "and" {
                    let prevIsMultiplier =
                        prev.map { Self.multipliers[$0] != nil } ?? false
                    if !prevIsMultiplier {
                        state.emitValue()
                        state.emit(current)
                    }
                } else if current == "double" || current == "triple" {
                    let nextIsOnesOrZero =
                        (next.map { Self.ones[$0] != nil } ?? false)
                        || (next.map { Self.zeros.contains($0) } ?? false)
                    if nextIsOnesOrZero, let next {
                        let repeats = current == "double" ? 2 : 3
                        let digit = Self.ones[next] ?? 0
                        let repeated = String(repeating: String(digit), count: repeats)
                        state.value = .string(state.value.stringForm + repeated)
                        state.skip = true
                    } else {
                        state.emitValue()
                        state.emit(current)
                    }
                } else if current == "point" {
                    let nextIsDecimal = next.map { Self.decimals.contains($0) } ?? false
                    if nextIsDecimal || nextIsNumeric {
                        state.value = .string(state.value.stringForm + ".")
                    }
                }
                continue
            }
        }

        if !state.value.isNone {
            state.emitValue()
        }

        return state.output
    }

    // MARK: - per-word helpers (split out for readability)

    private func processOnes(onesValue: Int, prev: String?, state: inout State) {
        switch state.value {
        case .none:
            state.value = .integer(onesValue)
        case .string(let s):
            if let prev, Self.tens[prev] != nil, onesValue < 10 {
                // replace the trailing zero of "...0" with the digit
                if s.hasSuffix("0") {
                    state.value = .string(String(s.dropLast()) + String(onesValue))
                } else {
                    state.value = .string(s + String(onesValue))
                }
            } else {
                state.value = .string(s + String(onesValue))
            }
        case .integer(let n):
            let prevInOnes = prev.map { Self.ones[$0] != nil } ?? false
            if prevInOnes {
                let prevInTens = prev.map { Self.tens[$0] != nil } ?? false
                if prevInTens && onesValue < 10 {
                    var s = String(n)
                    if s.hasSuffix("0") { s.removeLast() }
                    state.value = .string(s + String(onesValue))
                } else {
                    state.value = .string(String(n) + String(onesValue))
                }
            } else if onesValue < 10 {
                if n % 10 == 0 {
                    state.value = .integer(n + onesValue)
                } else {
                    state.value = .string(String(n) + String(onesValue))
                }
            } else {
                // eleven..nineteen
                if n % 100 == 0 {
                    state.value = .integer(n + onesValue)
                } else {
                    state.value = .string(String(n) + String(onesValue))
                }
            }
        }
    }

    private func processOnesSuffixed(
        onesValue: Int, suffix: String, prev: String?,
        state: inout State
    ) {
        switch state.value {
        case .none:
            state.emit(String(onesValue) + suffix)
        case .string(let s):
            if let prev, Self.tens[prev] != nil, onesValue < 10 {
                if s.hasSuffix("0") {
                    state.emit(String(s.dropLast()) + String(onesValue) + suffix)
                } else {
                    state.emit(s + String(onesValue) + suffix)
                }
            } else {
                state.emit(s + String(onesValue) + suffix)
            }
        case .integer(let n):
            let prevInOnes = prev.map { Self.ones[$0] != nil } ?? false
            if prevInOnes {
                let prevInTens = prev.map { Self.tens[$0] != nil } ?? false
                if prevInTens && onesValue < 10 {
                    var s = String(n)
                    if s.hasSuffix("0") { s.removeLast() }
                    state.emit(s + String(onesValue) + suffix)
                } else {
                    state.emit(String(n) + String(onesValue) + suffix)
                }
            } else if onesValue < 10 {
                if n % 10 == 0 {
                    state.emit(String(n + onesValue) + suffix)
                } else {
                    state.emit(String(n) + String(onesValue) + suffix)
                }
            } else {
                if n % 100 == 0 {
                    state.emit(String(n + onesValue) + suffix)
                } else {
                    state.emit(String(n) + String(onesValue) + suffix)
                }
            }
        }
        state.value = .none
    }

    private func processTens(tensValue: Int, state: inout State) {
        switch state.value {
        case .none:
            state.value = .integer(tensValue)
        case .string(let s):
            state.value = .string(s + String(tensValue))
        case .integer(let n):
            if n % 100 == 0 {
                state.value = .integer(n + tensValue)
            } else {
                state.value = .string(String(n) + String(tensValue))
            }
        }
    }

    private func processTensSuffixed(
        tensValue: Int, suffix: String,
        state: inout State
    ) {
        switch state.value {
        case .none:
            state.emit(String(tensValue) + suffix)
        case .string(let s):
            state.emit(s + String(tensValue) + suffix)
        case .integer(let n):
            if n % 100 == 0 {
                state.emit(String(n + tensValue) + suffix)
            } else {
                state.emit(String(n) + String(tensValue) + suffix)
            }
        }
        state.value = .none
    }

    private func processMultiplier(multiplier: Int, state: inout State) {
        switch state.value {
        case .none:
            state.value = .integer(multiplier)
        case .string(let s):
            if let f = Double(s), let product = exactProduct(f, multiplier) {
                state.value = .integer(product)
            } else {
                state.emit(s)
                state.value = .integer(multiplier)
            }
        case .integer(0):
            // Python: isinstance(value, str) or value == 0 → same branch.
            // For integer 0, Fraction("0") * multiplier = 0; denominator 1.
            state.value = .integer(0)
        case .integer(let n):
            let before = n / 1000 * 1000
            let residual = n % 1000
            state.value = .integer(before + residual * multiplier)
        }
    }

    private func processMultiplierSuffixed(
        multiplier: Int, suffix: String,
        state: inout State
    ) {
        switch state.value {
        case .none:
            state.emit(String(multiplier) + suffix)
        case .string(let s):
            if let f = Double(s), let product = exactProduct(f, multiplier) {
                state.emit(String(product) + suffix)
            } else {
                state.emit(s)
                state.emit(String(multiplier) + suffix)
            }
        case .integer(let n):
            let before = n / 1000 * 1000
            let residual = n % 1000
            let combined = before + residual * multiplier
            state.emit(String(combined) + suffix)
        }
        state.value = .none
    }

    /// Returns `f * m` as an exact Int when the product is an integer
    /// representable in Int range. Mirrors Python's `Fraction(value) * multiplier`
    /// check for `denominator == 1`.
    private func exactProduct(_ f: Double, _ m: Int) -> Int? {
        let product = f * Double(m)
        guard product.isFinite, product == product.rounded(),
            product >= Double(Int.min), product <= Double(Int.max)
        else { return nil }
        return Int(product)
    }

    // MARK: - preprocess / postprocess

    private static let andAHalfRegex = CompiledRegex(#"\band\s+a\s+half\b"#)
    private static let lettersBeforeDigitsRegex = CompiledRegex(#"([a-z])([0-9])"#)
    private static let digitsBeforeLettersRegex = CompiledRegex(#"([0-9])([a-z])"#)
    private static let digitOrdinalSuffixRegex =
        CompiledRegex(#"([0-9])\s+(st|nd|rd|th|s)\b"#)
    private static let combineCentsRegex =
        CompiledRegex(#"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b"#)
    private static let extractCentsRegex =
        CompiledRegex(#"[€£$]0\.([0-9]{1,2})\b"#)
    private static let oneReadabilityRegex = CompiledRegex(#"\b1(s?)\b"#)

    private func preprocess(_ s: String) -> String {
        // Replace "<number> and a half" with "<number> point five".
        var results: [String] = []
        let segments = Self.split(s, by: Self.andAHalfRegex)
        for (i, segment) in segments.enumerated() {
            if segment.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                continue
            }
            if i == segments.count - 1 {
                results.append(segment)
            } else {
                results.append(segment)
                // Inspect last whitespace-delimited word of the segment.
                let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
                let lastWord = trimmed.split(separator: " ").last.map(String.init) ?? ""
                if Self.decimals.contains(lastWord) || Self.multipliers[lastWord] != nil {
                    results.append("point five")
                } else {
                    results.append("and a half")
                }
            }
        }
        var out = results.joined(separator: " ")

        out = Self.lettersBeforeDigitsRegex.replace(out, with: "$1 $2")
        out = Self.digitsBeforeLettersRegex.replace(out, with: "$1 $2")
        out = Self.digitOrdinalSuffixRegex.replace(out, with: "$1$2")
        return out
    }

    private func postprocess(_ s: String) -> String {
        var out = s
        // "$2 and ¢7" → "$2.07"
        out = Self.replaceCombineCents(out)
        // "$0.07" written as "$0.07"... actually `[€£$]0.([0-9]{1,2})\b` → "¢7"
        out = Self.replaceExtractCents(out)
        // "1" → "one", "1s" → "ones" (for readability)
        out = Self.oneReadabilityRegex.replace(out, with: "one$1")
        return out
    }

    /// Python `re.split` returns the segments between matches.
    /// `re.split(r"\band\s+a\s+half\b", "X and a half Y")` → `["X ", " Y"]`.
    /// Captures are not used here, so we can rely on `NSRegularExpression`'s
    /// split-by-match positions.
    private static func split(_ text: String, by regex: CompiledRegex) -> [String] {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = regex.nsRegex.matches(in: text, options: [], range: range)
        if matches.isEmpty { return [text] }
        var parts: [String] = []
        var cursor = text.startIndex
        for m in matches {
            guard let r = Range(m.range, in: text) else { continue }
            parts.append(String(text[cursor..<r.lowerBound]))
            cursor = r.upperBound
        }
        parts.append(String(text[cursor..<text.endIndex]))
        return parts
    }

    /// Apply `combine_cents` callback: `([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})`
    /// → `"<curr><int>.<cc>"` with the cents zero-padded to two digits.
    private static func replaceCombineCents(_ text: String) -> String {
        let regex = combineCentsRegex.nsRegex
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = regex.matches(in: text, options: [], range: range)
        guard !matches.isEmpty else { return text }
        var result = text
        for m in matches.reversed() {
            guard let mRange = Range(m.range, in: result),
                let g1 = Range(m.range(at: 1), in: result),
                let g2 = Range(m.range(at: 2), in: result),
                let g3 = Range(m.range(at: 3), in: result)
            else { continue }
            let currency = String(result[g1])
            let integer = String(result[g2])
            guard let cents = Int(result[g3]) else { continue }
            let replacement = "\(currency)\(integer).\(String(format: "%02d", cents))"
            result.replaceSubrange(mRange, with: replacement)
        }
        return result
    }

    /// Apply `extract_cents`: `[€£$]0.([0-9]{1,2})` → `¢<int>`.
    private static func replaceExtractCents(_ text: String) -> String {
        let regex = extractCentsRegex.nsRegex
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = regex.matches(in: text, options: [], range: range)
        guard !matches.isEmpty else { return text }
        var result = text
        for m in matches.reversed() {
            guard let mRange = Range(m.range, in: result),
                let g1 = Range(m.range(at: 1), in: result),
                let cents = Int(result[g1])
            else { continue }
            result.replaceSubrange(mRange, with: "¢\(cents)")
        }
        return result
    }
}
