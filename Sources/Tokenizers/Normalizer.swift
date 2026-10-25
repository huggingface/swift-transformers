//
//  Normalizer.swift
//
//
//  Created by Pedro Cuenca on 17/7/23.
//

import Foundation
import Hub

/// A protocol for text normalization operations applied before tokenization.
///
/// Normalizers perform text preprocessing such as lowercasing, accent removal,
/// Unicode normalization, and other text cleaning operations to standardize
/// input before the main tokenization process.
public protocol Normalizer {
    /// Normalizes the input text according to the normalizer's rules.
    ///
    /// - Parameter text: The input text to normalize
    /// - Returns: The normalized text
    func normalize(text: String) -> String

    /// Function call syntax for text normalization.
    ///
    /// - Parameter text: The input text to normalize
    /// - Returns: The normalized text
    func callAsFunction(text: String) -> String

    /// Initializes the normalizer from configuration.
    ///
    /// - Parameter config: The configuration for this normalizer
    /// - Throws: An error if a configured regular expression cannot be compiled.
    init(config: Config) throws
}

extension Normalizer {
    func callAsFunction(text: String) -> String {
        normalize(text: text)
    }
}

enum NormalizerType: String {
    case Sequence
    case Prepend
    case Replace
    case Lowercase
    case NFD
    case NFC
    case NFKD
    case NFKC
    case Bert
    case BertNormalizer
    case Precompiled
    case StripAccents
    case Strip
    case Unknown = ""
}

struct NormalizerFactory {
    static func fromConfig(config: Config?) throws -> Normalizer? {
        guard let config else { return nil }
        guard let typeName = config.type.string() else { return nil }
        let type = NormalizerType(rawValue: typeName)
        switch type {
        case .Sequence: return try NormalizerSequence(config: config)
        case .Prepend: return PrependNormalizer(config: config)
        case .Replace: return try ReplaceNormalizer(config: config)
        case .Lowercase: return LowercaseNormalizer(config: config)
        case .NFD: return NFDNormalizer(config: config)
        case .NFC: return NFCNormalizer(config: config)
        case .NFKD: return NFKDNormalizer(config: config)
        case .NFKC: return NFKCNormalizer(config: config)
        case .Bert, .BertNormalizer: return BertNormalizer(config: config)
        case .Precompiled: return PrecompiledNormalizer(config: config)
        case .StripAccents: return StripAccentsNormalizer(config: config)
        case .Strip: return StripNormalizer(config: config)
        default: fatalError("Unsupported Normalizer type: \(typeName)")
        }
    }
}

class NormalizerSequence: Normalizer {
    let normalizers: [Normalizer]

    required init(config: Config) throws {
        guard let configs = config.normalizers.array() else {
            fatalError("No normalizers in Sequence")
        }
        normalizers = try configs.compactMap { try NormalizerFactory.fromConfig(config: $0) }
    }

    func normalize(text: String) -> String {
        normalizers.reduce(text) { current, normalizer in
            normalizer(text: current)
        }
    }
}

class PrependNormalizer: Normalizer {
    let prepend: String

    required init(config: Config) {
        prepend = config.prepend.string(or: "")
    }

    func normalize(text: String) -> String {
        prepend + text
    }
}

class ReplaceNormalizer: Normalizer {
    let pattern: StringReplacePattern?

    required init(config: Config) throws {
        pattern = try StringReplacePattern.from(config: config)
    }

    func normalize(text: String) -> String {
        guard let pattern else { return text }
        return pattern.replace(text)
    }
}

class LowercaseNormalizer: Normalizer {
    required init(config: Config) {}

    func normalize(text: String) -> String {
        text.lowercased()
    }
}

class NFDNormalizer: Normalizer {
    required init(config: Config) {}

    func normalize(text: String) -> String {
        text.decomposedStringWithCanonicalMapping
    }
}

class NFCNormalizer: Normalizer {
    required init(config: Config) {}

    func normalize(text: String) -> String {
        text.precomposedStringWithCanonicalMapping
    }
}

class NFKDNormalizer: Normalizer {
    required init(config: Config) {}

    func normalize(text: String) -> String {
        text.decomposedStringWithCompatibilityMapping
    }
}

class NFKCNormalizer: Normalizer {
    required init(config: Config) {}

    func normalize(text: String) -> String {
        text.precomposedStringWithCompatibilityMapping
    }
}

class BertNormalizer: Normalizer {
    let shouldCleanText: Bool
    let shouldHandleChineseChars: Bool
    let shouldStripAccents: Bool
    let shouldLowercase: Bool

    required init(config: Config) {
        shouldCleanText = config.cleanText.boolean(or: true)
        shouldHandleChineseChars = config.handleChineseChars.boolean(or: true)
        shouldLowercase = config.lowercase.boolean(or: true)
        shouldStripAccents = config.stripAccents.boolean(or: shouldLowercase)
    }

    func normalize(text: String) -> String {
        var output = text
        if shouldCleanText {
            output = cleanText(text: output)
        }
        if shouldHandleChineseChars {
            output = handleChineseChars(text: output)
        }
        if shouldStripAccents {
            output = stripAccents(text: output)
        }
        if shouldLowercase {
            output = output.lowercased()
        }

        return output
    }

    private func cleanText(text: String) -> String {
        text.map { c in
            guard let scalar = c.unicodeScalars.first,
                scalar.value != 0x0,
                scalar.value != 0xFFFD,
                !isControl(scalar)
            else { return "\(c)" }

            // Replace whitespace: \t, \n, \r
            if scalar.value == 0x009 || scalar.value == 0x00A || scalar.value == 0x000D {
                return " "
            } else {
                return "\(c)"
            }
        }
        .joined()
    }

    private func isControl(_ c: UnicodeScalar) -> Bool {
        if c.value == 0x009 || c.value == 0x00A || c.value == 0x000D {
            // Except \t, \n, \r that will be spaces.
            false
        } else {
            // https://unicode.org/reports/tr44/#GC_Values_Table
            // Other Cc | Cf | Cs | Co | Cn
            isOther(c.properties.generalCategory)
        }
    }

    private func isOther(_ c: Unicode.GeneralCategory) -> Bool {
        c == .control || c == .format || c == .surrogate || c == .privateUse || c == .unassigned
    }

    private func handleChineseChars(text: String) -> String {
        text.map { c in
            if let scalar = c.unicodeScalars.first, scalar.isCJKUnifiedIdeograph {
                " \(c) "
            } else {
                "\(c)"
            }
        }
        .joined()
    }

    private func stripAccents(text: String) -> String {
        // Equivalent of HF Python's `_run_strip_accents`: NFD-decompose then drop every
        // nonspacing-mark scalar (Unicode general category Mn). Filtering only
        // U+0300..U+036F handled Latin diacritics but missed marks in other blocks —
        // notably the Japanese voiced-kana combining marks U+3099/U+309A.
        // This matches Rust `tokenizers` and HF Python.
        String(
            String.UnicodeScalarView(
                text.decomposedStringWithCanonicalMapping.unicodeScalars.filter {
                    $0.properties.generalCategory != .nonspacingMark
                }
            )
        )
    }
}

extension UnicodeScalar {
    /// https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    var isCJKUnifiedIdeograph: Bool {
        (value >= 0x4E00 && value <= 0x9FFF)
            || (value >= 0x3400 && value <= 0x4DBF)
            || (value >= 0x20000 && value <= 0x2A6DF)
            || (value >= 0x2A700 && value <= 0x2B73F)
            || (value >= 0x2B740 && value <= 0x2B81F)
            || (value >= 0x2B820 && value <= 0x2CEAF)
            || (value >= 0xF900 && value <= 0xFAFF)
            || (value >= 0x2F800 && value <= 0x2FA1F)
    }
}

/// SentencePiece `precompiled_charsmap`: a Darts double-array trie keyed by UTF-8
/// bytes, whose leaves hold offsets into a blob of NUL-terminated replacements.
///
/// Port of https://github.com/huggingface/spm_precompiled
struct PrecompiledCharsmap {
    private let trie: [UInt32]
    private let replacements: [UInt8]

    /// Layout: `[u32 LE trie size in bytes][trie: u32 LE units][replacements]`.
    init?(_ data: Data) {
        let bytes = [UInt8](data)
        guard bytes.count >= 4 else { return nil }
        func u32(at offset: Int) -> UInt32 {
            UInt32(bytes[offset]) | UInt32(bytes[offset + 1]) << 8 | UInt32(bytes[offset + 2]) << 16 | UInt32(bytes[offset + 3]) << 24
        }
        let unitCount = Int(u32(at: 0)) / 4
        let blobStart = 4 + unitCount * 4
        guard unitCount > 0, blobStart <= bytes.count else { return nil }
        trie = (0..<unitCount).map { u32(at: 4 + $0 * 4) }
        replacements = Array(bytes[blobStart...])
    }

    /// The replacement for the *shortest* prefix of `key` found in the trie, as
    /// `Precompiled::transform` returns the first `common_prefix_search` result.
    func transform(_ key: some Sequence<UInt8>) -> String? {
        func hasLeaf(_ unit: UInt32) -> Bool { (unit >> 8) & 1 == 1 }
        func value(_ unit: UInt32) -> Int { Int(unit & ((1 << 31) - 1)) }
        func label(_ unit: UInt32) -> UInt32 { unit & ((1 << 31) | 0xFF) }
        func offset(_ unit: UInt32) -> Int { Int((unit >> 10) << ((unit & (1 << 9)) >> 6)) }

        var position = offset(trie[0])
        for byte in key {
            if byte == 0 { break }
            position ^= Int(byte)
            guard position < trie.count, label(trie[position]) == UInt32(byte) else { return nil }
            let unit = trie[position]
            position ^= offset(unit)
            if hasLeaf(unit) {
                guard position < trie.count else { return nil }
                let start = value(trie[position])
                guard start <= replacements.count else { return nil }
                let end = replacements[start...].firstIndex(of: 0) ?? replacements.count
                return String(decoding: replacements[start..<end], as: UTF8.self)
            }
        }
        return nil
    }
}

class PrecompiledNormalizer: Normalizer {
    private let charsmap: PrecompiledCharsmap?

    required init(config: Config) {
        charsmap = config.precompiledCharsmap.string()
            .flatMap { Data(base64Encoded: $0) }
            .flatMap { PrecompiledCharsmap($0) }
    }

    /// https://github.com/huggingface/tokenizers/blob/v0.23.2/tokenizers/src/normalizers/precompiled.rs
    ///
    /// A grapheme cluster shorter than 6 UTF-8 bytes is replaced as a whole when any prefix
    /// of it is in the charsmap, so "\u{2581}\u{0301}" becomes " ". Otherwise each scalar
    /// is transformed on its own.
    func normalize(text: String) -> String {
        guard let charsmap else { return approximateNormalize(text: text) }
        var output = ""
        for grapheme in text {
            if grapheme.utf8.count < 6, let replacement = charsmap.transform(grapheme.utf8) {
                output += replacement
                continue
            }
            for scalar in grapheme.unicodeScalars {
                if let replacement = charsmap.transform(scalar.utf8) {
                    output += replacement
                } else {
                    output.unicodeScalars.append(scalar)
                }
            }
        }
        return output
    }

    /// Approximation for configurations without a `precompiled_charsmap`.
    private func approximateNormalize(text: String) -> String {
        // - The following comments also apply here:
        // https://github.com/xenova/transformers.js/blob/main/src/tokenizers.js#L2237-L2247
        var output = ""
        var hasFullwidthTilde = false

        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x0001...0x0008, 0x000B, 0x000E...0x001F, 0x007F, 0x008F, 0x009F:
                // Non-printing control characters
                output.append("")
            case 0x0009, 0x000A, 0x000C, 0x000D, 0x1680, 0x200B...0x200F, 0x2028, 0x2029, 0x2581,
                0xFEFF, 0xFFFD:
                // Separators
                output.append(" ")
            case 0xFF5E:
                hasFullwidthTilde = true
                fallthrough
            default:
                output.append(Character(scalar))
            }
        }

        if hasFullwidthTilde {
            return
                output
                .split(by: "\u{FF5E}")
                .map { $0.precomposedStringWithCompatibilityMapping }
                .joined(separator: "\u{FF5E}")
        } else {
            return output.precomposedStringWithCompatibilityMapping
        }
    }
}

class StripAccentsNormalizer: Normalizer {
    required init(config: Config) {}

    /// Removes combining marks (Unicode general category M), like `StripAccents` in Rust `tokenizers`.
    /// Precomposed characters are kept, so it is usually preceded by an `NFD` normalizer.
    func normalize(text: String) -> String {
        String(
            String.UnicodeScalarView(
                text.unicodeScalars.filter {
                    switch $0.properties.generalCategory {
                    case .nonspacingMark, .spacingMark, .enclosingMark: false
                    default: true
                    }
                }
            )
        )
    }
}

class StripNormalizer: Normalizer {
    let leftStrip: Bool
    let rightStrip: Bool

    required init(config: Config) {
        leftStrip = config.stripLeft.boolean(or: true)
        rightStrip = config.stripRight.boolean(or: true)
    }

    func normalize(text: String) -> String {
        var result = text

        if leftStrip {
            result = String(result.drop(while: { $0.isWhitespace }))
        }

        if rightStrip {
            result = String(result.reversed().drop(while: { $0.isWhitespace }).reversed())
        }

        return result
    }
}

enum StringReplacePattern {
    case regexp(regexp: NSRegularExpression, replacement: String)
    case string(pattern: String, replacement: String)
}

extension StringReplacePattern {
    func replace(_ text: String) -> String {
        switch self {
        case let .regexp(regexp, replacement):
            let range = NSRange(text.startIndex..., in: text)
            let replaced = regexp.stringByReplacingMatches(
                in: text, options: [], range: range, withTemplate: replacement
            )
            return replaced
        case let .string(toReplace, replacement):
            return text.replacingOccurrences(of: toReplace, with: replacement, options: .literal)
        }
    }
}

extension StringReplacePattern {
    static func from(config: Config) throws -> StringReplacePattern? {
        guard let replacement = config.content.string() else { return nil }
        if let pattern = config.pattern.String.string() {
            return StringReplacePattern.string(pattern: pattern, replacement: replacement)
        }
        if let pattern = config.pattern.Regex.string() {
            do {
                return StringReplacePattern.regexp(regexp: try NSRegularExpression(pattern: pattern), replacement: replacement)
            } catch {
                throw TokenizerError.invalidRegex(pattern: pattern, underlyingError: error)
            }
        }
        return nil
    }
}
