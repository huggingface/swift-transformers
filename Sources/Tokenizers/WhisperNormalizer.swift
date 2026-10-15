//
//  WhisperNormalizer.swift
//
//  Whisper text normalization for ASR evaluation (WER/CER), ported from
//  https://github.com/openai/whisper/tree/main/whisper/normalizers
//  and kept byte-identical to `transformers.models.whisper.english_normalizer`.
//
//  These normalizers are NOT applied during regular tokenization. They are
//  opt-in text post-processors intended for evaluation pipelines that need
//  to compare hypotheses against references in a canonical form.
//

import Foundation

// MARK: - BasicTextNormalizer

/// Language-agnostic Whisper text normalizer.
///
/// Lowercases input, removes bracketed/parenthesized spans, drops or
/// space-replaces Unicode marks/symbols/punctuation, optionally splits
/// the text into grapheme clusters, and collapses whitespace.
///
/// Mirrors `whisper.normalizers.BasicTextNormalizer`.
public struct BasicTextNormalizer: Sendable {
    /// When `true`, applies NFKD decomposition and drops combining marks
    /// (and a handful of non-decomposable diacritics such as `œ`, `ß`).
    public let removeDiacritics: Bool

    /// When `true`, splits the output into grapheme clusters separated by
    /// single spaces (used by CER-style evaluations).
    public let splitLetters: Bool

    public init(removeDiacritics: Bool = false, splitLetters: Bool = false) {
        self.removeDiacritics = removeDiacritics
        self.splitLetters = splitLetters
    }

    public func callAsFunction(_ text: String) -> String {
        normalize(text)
    }

    public func normalize(_ text: String) -> String {
        var s = text.lowercased()
        s = WhisperNormalizerRegex.bracketed.replace(s, with: "")
        s = WhisperNormalizerRegex.parenthesized.replace(s, with: "")
        s =
            removeDiacritics
            ? WhisperNormalizerSupport.removeSymbolsAndDiacritics(s, keep: "")
            : WhisperNormalizerSupport.removeSymbols(s)
        s = s.lowercased()

        if splitLetters {
            s = String(s.map(Character.init).map(String.init).joined(separator: " "))
        }

        s = WhisperNormalizerRegex.whitespace.replace(s, with: " ")
        return s
    }
}

// MARK: - EnglishSpellingNormalizer

/// Applies British→American spelling substitutions to whitespace-separated
/// words. Mirrors `whisper.normalizers.EnglishSpellingNormalizer`.
///
/// The mapping table is generated from `openai/whisper/english.json` by
/// `Tools/generate_whisper_normalizer_baselines.py`.
public struct EnglishSpellingNormalizer: Sendable {
    public let mapping: [String: String]

    public init(mapping: [String: String] = englishSpellingTable) {
        self.mapping = mapping
    }

    public func callAsFunction(_ text: String) -> String {
        normalize(text)
    }

    public func normalize(_ text: String) -> String {
        text.split(separator: " ", omittingEmptySubsequences: false)
            .map { word in mapping[String(word)] ?? String(word) }
            .joined(separator: " ")
    }
}

// MARK: - EnglishTextNormalizer

/// English-specific Whisper text normalizer used for WER evaluation on
/// the Open ASR Leaderboard and similar benchmarks.
///
/// Pipeline (in order):
/// 1. Lowercase.
/// 2. Remove bracketed/parenthesized spans.
/// 3. Strip filler words (`hmm`, `mm`, `mhm`, `mmm`, `uh`, `um`).
/// 4. Remove whitespace before apostrophes.
/// 5. Apply contraction and title replacers (`won't` → `will not`, `mr` → `mister`, ...).
/// 6. Strip commas between digits; turn periods that don't precede digits into spaces.
/// 7. Remove symbols/diacritics (keeping `.%$¢€£`).
/// 8. Normalize spelled-out numbers and currency.
/// 9. Apply British→American spelling substitutions.
/// 10. Drop residual numeric symbols not adjacent to digits.
/// 11. Collapse whitespace.
///
/// Mirrors `whisper.normalizers.EnglishTextNormalizer`.
public struct EnglishTextNormalizer: Sendable {
    public let spellingNormalizer: EnglishSpellingNormalizer
    public let numberNormalizer: EnglishNumberNormalizer

    public init(spellingMapping: [String: String] = englishSpellingTable) {
        self.spellingNormalizer = EnglishSpellingNormalizer(mapping: spellingMapping)
        self.numberNormalizer = EnglishNumberNormalizer()
    }

    public func callAsFunction(_ text: String) -> String {
        normalize(text)
    }

    public func normalize(_ text: String) -> String {
        var s = text.lowercased()

        s = WhisperNormalizerRegex.bracketed.replace(s, with: "")
        s = WhisperNormalizerRegex.parenthesized.replace(s, with: "")
        s = WhisperNormalizerRegex.fillerWords.replace(s, with: "")
        s = WhisperNormalizerRegex.spaceBeforeApostrophe.replace(s, with: "'")

        for replacer in WhisperNormalizerRegex.englishReplacers {
            s = replacer.regex.replace(s, with: replacer.replacement)
        }

        s = WhisperNormalizerRegex.commaBetweenDigits.replace(s, with: "$1$2")
        s = WhisperNormalizerRegex.periodNotBeforeDigit.replace(s, with: " $1")
        s = WhisperNormalizerSupport.removeSymbolsAndDiacritics(s, keep: ".%$¢€£")

        s = numberNormalizer(s)
        s = spellingNormalizer(s)

        s = WhisperNormalizerRegex.numericSymbolNotBeforeDigit.replace(s, with: " $1")
        s = WhisperNormalizerRegex.percentNotAfterDigit.replace(s, with: "$1 ")

        s = WhisperNormalizerRegex.whitespace.replace(s, with: " ")
        return s
    }
}

// MARK: - Shared support

enum WhisperNormalizerSupport {
    /// Non-ASCII letters that are not separated by NFKD normalization, and
    /// therefore need explicit ASCII substitutions for `removeDiacritics`.
    static let additionalDiacritics: [UnicodeScalar: String] = [
        "œ": "oe", "Œ": "OE",
        "ø": "o", "Ø": "O",
        "æ": "ae", "Æ": "AE",
        "ß": "ss", "ẞ": "SS",
        "đ": "d", "Đ": "D",
        "ð": "d", "Ð": "D",
        "þ": "th", "Þ": "th",
        "ł": "l", "Ł": "L",
    ]

    /// Replaces marks, symbols, and punctuation with a space, drops
    /// non-spacing diacritics, and applies `additionalDiacritics`.
    ///
    /// Mirrors `remove_symbols_and_diacritics(s, keep="")` after NFKD.
    static func removeSymbolsAndDiacritics(_ s: String, keep: String) -> String {
        let keepSet = Set(keep.unicodeScalars)
        let decomposed = s.decomposedStringWithCompatibilityMapping
        var result = ""
        result.reserveCapacity(decomposed.unicodeScalars.count)
        for scalar in decomposed.unicodeScalars {
            if keepSet.contains(scalar) {
                result.unicodeScalars.append(scalar)
                continue
            }
            if let mapped = additionalDiacritics[scalar] {
                result += mapped
                continue
            }
            let category = scalar.properties.generalCategory
            if category == .nonspacingMark {
                continue
            }
            if WhisperNormalizerSupport.isMarkSymbolOrPunctuation(category) {
                result += " "
                continue
            }
            result.unicodeScalars.append(scalar)
        }
        return result
    }

    /// Replaces marks, symbols, and punctuation with a space, keeping
    /// composed diacritics. Mirrors `remove_symbols(s)` after NFKC.
    static func removeSymbols(_ s: String) -> String {
        let composed = s.precomposedStringWithCompatibilityMapping
        var result = ""
        result.reserveCapacity(composed.unicodeScalars.count)
        for scalar in composed.unicodeScalars {
            if WhisperNormalizerSupport.isMarkSymbolOrPunctuation(
                scalar.properties.generalCategory)
            {
                result += " "
            } else {
                result.unicodeScalars.append(scalar)
            }
        }
        return result
    }

    /// True if `category` is any Unicode Mark / Symbol / Punctuation class
    /// (Python's `unicodedata.category(c)[0] in "MSP"`).
    static func isMarkSymbolOrPunctuation(_ category: Unicode.GeneralCategory) -> Bool {
        switch category {
        case .nonspacingMark, .spacingMark, .enclosingMark,
            .mathSymbol, .currencySymbol, .modifierSymbol, .otherSymbol,
            .connectorPunctuation, .dashPunctuation,
            .openPunctuation, .closePunctuation,
            .initialPunctuation, .finalPunctuation, .otherPunctuation:
            return true
        default:
            return false
        }
    }
}

// MARK: - Regex registry

/// File-scope `NSRegularExpression` cache for the Whisper normalizers.
/// Building these once at module load avoids reparsing every call.
enum WhisperNormalizerRegex {
    static let bracketed = CompiledRegex(#"[<\[][^>\]]*[>\]]"#)
    static let parenthesized = CompiledRegex(#"\(([^)]+?)\)"#)
    static let fillerWords = CompiledRegex(#"\b(hmm|mm|mhm|mmm|uh|um)\b"#)
    static let spaceBeforeApostrophe = CompiledRegex(#"\s+'"#)
    static let commaBetweenDigits = CompiledRegex(#"(\d),(\d)"#)
    static let periodNotBeforeDigit = CompiledRegex(#"\.([^0-9]|$)"#)
    static let numericSymbolNotBeforeDigit = CompiledRegex(#"[.$¢€£]([^0-9])"#)
    static let percentNotAfterDigit = CompiledRegex(#"([^0-9])%"#)
    static let whitespace = CompiledRegex(#"\s+"#)

    struct Replacer {
        let regex: CompiledRegex
        let replacement: String
    }

    /// Ordered contraction / title replacers from
    /// `EnglishTextNormalizer.replacers`. Order is significant.
    static let englishReplacers: [Replacer] = [
        // common contractions
        Replacer(regex: CompiledRegex(#"\bwon't\b"#), replacement: "will not"),
        Replacer(regex: CompiledRegex(#"\bcan't\b"#), replacement: "can not"),
        Replacer(regex: CompiledRegex(#"\blet's\b"#), replacement: "let us"),
        Replacer(regex: CompiledRegex(#"\bain't\b"#), replacement: "aint"),
        Replacer(regex: CompiledRegex(#"\by'all\b"#), replacement: "you all"),
        Replacer(regex: CompiledRegex(#"\bwanna\b"#), replacement: "want to"),
        Replacer(regex: CompiledRegex(#"\bgotta\b"#), replacement: "got to"),
        Replacer(regex: CompiledRegex(#"\bgonna\b"#), replacement: "going to"),
        Replacer(regex: CompiledRegex(#"\bi'ma\b"#), replacement: "i am going to"),
        Replacer(regex: CompiledRegex(#"\bimma\b"#), replacement: "i am going to"),
        Replacer(regex: CompiledRegex(#"\bwoulda\b"#), replacement: "would have"),
        Replacer(regex: CompiledRegex(#"\bcoulda\b"#), replacement: "could have"),
        Replacer(regex: CompiledRegex(#"\bshoulda\b"#), replacement: "should have"),
        Replacer(regex: CompiledRegex(#"\bma'am\b"#), replacement: "madam"),
        // titles
        Replacer(regex: CompiledRegex(#"\bmr\b"#), replacement: "mister "),
        Replacer(regex: CompiledRegex(#"\bmrs\b"#), replacement: "missus "),
        Replacer(regex: CompiledRegex(#"\bst\b"#), replacement: "saint "),
        Replacer(regex: CompiledRegex(#"\bdr\b"#), replacement: "doctor "),
        Replacer(regex: CompiledRegex(#"\bprof\b"#), replacement: "professor "),
        Replacer(regex: CompiledRegex(#"\bcapt\b"#), replacement: "captain "),
        Replacer(regex: CompiledRegex(#"\bgov\b"#), replacement: "governor "),
        Replacer(regex: CompiledRegex(#"\bald\b"#), replacement: "alderman "),
        Replacer(regex: CompiledRegex(#"\bgen\b"#), replacement: "general "),
        Replacer(regex: CompiledRegex(#"\bsen\b"#), replacement: "senator "),
        Replacer(regex: CompiledRegex(#"\brep\b"#), replacement: "representative "),
        Replacer(regex: CompiledRegex(#"\bpres\b"#), replacement: "president "),
        Replacer(regex: CompiledRegex(#"\brev\b"#), replacement: "reverend "),
        Replacer(regex: CompiledRegex(#"\bhon\b"#), replacement: "honorable "),
        Replacer(regex: CompiledRegex(#"\basst\b"#), replacement: "assistant "),
        Replacer(regex: CompiledRegex(#"\bassoc\b"#), replacement: "associate "),
        Replacer(regex: CompiledRegex(#"\blt\b"#), replacement: "lieutenant "),
        Replacer(regex: CompiledRegex(#"\bcol\b"#), replacement: "colonel "),
        Replacer(regex: CompiledRegex(#"\bjr\b"#), replacement: "junior "),
        Replacer(regex: CompiledRegex(#"\bsr\b"#), replacement: "senior "),
        Replacer(regex: CompiledRegex(#"\besq\b"#), replacement: "esquire "),
        // perfect tenses
        Replacer(regex: CompiledRegex(#"'d been\b"#), replacement: " had been"),
        Replacer(regex: CompiledRegex(#"'s been\b"#), replacement: " has been"),
        Replacer(regex: CompiledRegex(#"'d gone\b"#), replacement: " had gone"),
        Replacer(regex: CompiledRegex(#"'s gone\b"#), replacement: " has gone"),
        Replacer(regex: CompiledRegex(#"'d done\b"#), replacement: " had done"),
        Replacer(regex: CompiledRegex(#"'s got\b"#), replacement: " has got"),
        // general contractions
        Replacer(regex: CompiledRegex(#"n't\b"#), replacement: " not"),
        Replacer(regex: CompiledRegex(#"'re\b"#), replacement: " are"),
        Replacer(regex: CompiledRegex(#"'s\b"#), replacement: " is"),
        Replacer(regex: CompiledRegex(#"'d\b"#), replacement: " would"),
        Replacer(regex: CompiledRegex(#"'ll\b"#), replacement: " will"),
        Replacer(regex: CompiledRegex(#"'t\b"#), replacement: " not"),
        Replacer(regex: CompiledRegex(#"'ve\b"#), replacement: " have"),
        Replacer(regex: CompiledRegex(#"'m\b"#), replacement: " am"),
    ]
}

/// Lightweight wrapper around `NSRegularExpression` that hides the
/// `NSRange` plumbing and falls through unchanged if the pattern fails to
/// compile (which is a programmer error, not a runtime concern).
struct CompiledRegex: @unchecked Sendable {
    let nsRegex: NSRegularExpression

    init(_ pattern: String, options: NSRegularExpression.Options = []) {
        do {
            nsRegex = try NSRegularExpression(pattern: pattern, options: options)
        } catch {
            fatalError("Invalid regex pattern \(pattern): \(error)")
        }
    }

    func replace(_ text: String, with template: String) -> String {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return nsRegex.stringByReplacingMatches(
            in: text,
            options: [],
            range: range,
            withTemplate: template
        )
    }
}
