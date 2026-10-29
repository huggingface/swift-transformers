import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

// The pre-change implementations, kept verbatim as the reference the new ones must match.
private enum Legacy {
    static func value(_ str: NSString) -> [UInt16] { Array(str as String).flatMap { $0.utf16 } }
    static func value(_ str: String) -> [UInt16] { value(str as NSString) }
    static func nsString(_ value: [UInt16]) -> NSString { String(utf16CodeUnits: value, count: value.count) as NSString }
    static func string(_ value: [UInt16]) -> String { String(nsString(value)) }
}

private func utf16Units(_ ns: NSString) -> [UInt16] {
    (0..<ns.length).map { ns.character(at: $0) }
}

private let wellFormed: [String] = [
    "", "a", "hello", "\u{2581}hello", "\u{2581}", "\u{2581}\u{2581}the", "<0xF0>", "<start_of_turn>",
    "caf\u{E9}", "cafe\u{301}", // precomposed and decomposed e-acute must stay distinct
    "\u{1F408}", "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}", "\u{1F1EE}\u{1F1EA}", // emoji, ZWJ family, flag
    "\u{1112}\u{1161}\u{11AB}", "\u{D55C}", // Hangul jamo vs syllable
    "\u{212B}", "\u{C5}", "A\u{30A}", // Angstrom sign, A-ring, A + combining ring
    "\u{FB01}", "\u{0000}", "a\u{0000}b", "\r\n", "\u{FEFF}", "\u{10FFFF}",
    "\u{4E2D}\u{6587}", "\u{0645}\u{0631}\u{062D}\u{0628}\u{0627}",
]

// Lone surrogates can only come from NSString/UTF-16 sources (a `\ud800` escape in JSON).
private let loneSurrogates: [[UInt16]] = [
    [0xD800], [0xDC00], [0x61, 0xD83D], [0xDE08, 0x62], [0xD83D, 0x61], [0xDBFF, 0xDBFF, 0xDFFF],
]

// Darwin's NSString holds raw UTF-16, so it keeps a lone surrogate. swift-corelibs-foundation stores an NSString as a
// Swift String, so on Linux `String(utf16CodeUnits:count:)` and every NSString repair it to U+FFFD, and the previous
// implementation did the same.
private func expectedNSStringUnits(_ units: [UInt16]) -> [UInt16] {
    #if canImport(Darwin)
    return units
    #else
    return Array(String(decoding: units, as: UTF16.self).utf16)
    #endif
}

@Suite("BinaryDistinctString equivalence with the previous implementation")
struct BinaryDistinctStringTests {
    @Test("Well-formed strings keep identical UTF-16 values through every initializer", arguments: wellFormed)
    func wellFormedValues(_ text: String) {
        let expected = Legacy.value(text)
        #expect(BinaryDistinctString(text).value == expected)
        #expect(BinaryDistinctString(text as NSString).value == expected)
        #expect(expected == Array(text.utf16))
    }

    @Test("Well-formed strings keep identical string and nsString views", arguments: wellFormed)
    func wellFormedViews(_ text: String) {
        let bds = BinaryDistinctString(text)
        #expect(Array(bds.string.utf8) == Array(Legacy.string(bds.value).utf8))
        #expect(Array(bds.string.utf16) == Array(Legacy.string(bds.value).utf16))
        #expect(utf16Units(bds.nsString) == utf16Units(Legacy.nsString(bds.value)))
        #expect(bds.string == text)
        // Distinct binary forms stay distinct.
        #expect(BinaryDistinctString("cafe\u{301}") != BinaryDistinctString("caf\u{E9}"))
    }

    @Test("Vocabulary and merge keys built from a BPE tokenizer.json match the previous results")
    func vocabAndMerges() throws {
        let vocab: [(String, Int)] = [
            ("<unk>", 0), ("a", 1), ("b", 2), ("ab", 3), ("\u{2581}a", 4), ("\u{2581}ab", 5), ("caf\u{E9}", 6),
            ("\u{1F408}", 8), ("\u{1F468}\u{200D}\u{1F469}", 9), ("\u{C5}", 10),
        ]
        let merges = ["a b", "\u{2581} a", "\u{2581}a b", "c a", "A \u{30A}"]
        let vocabJSON = vocab.map { "\"\($0.0)\": \($0.1)" }.joined(separator: ",")
        let mergesJSON = merges.map { "\"\($0)\"" }.joined(separator: ",")
        let json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\(vocabJSON)},\"merges\":[\(mergesJSON)],\"unk_token\":\"<unk>\"}}"
        let data = try YYJSONParser.parseToConfig(Data(json.utf8))
        let config = try JSONDecoder().decode(Config.self, from: Data(#"{"unk_token":"<unk>"}"#.utf8))
        let tokenizer = try BPETokenizer(tokenizerConfig: config, tokenizerData: data, addedTokens: ["<extra>": 12])

        #expect(tokenizer.vocabCount == vocab.count + 1)
        for (token, id) in vocab {
            #expect(tokenizer.convertTokenToId(token) == id, "\(token)")
            #expect(tokenizer.convertIdToToken(id).map { Array($0.utf16) } == Array(token.utf16))
        }
        #expect(tokenizer.convertTokenToId("<extra>") == 12)

        #expect(tokenizer.bpeRanks.count == merges.count)
        #expect(tokenizer.bpeRanks[BytePair("a", "b")] == 0)
        #expect(tokenizer.bpeRanks[BytePair("\u{2581}", "a")] == 1)
        #expect(tokenizer.bpeRanks[BytePair("\u{2581}a", "b")] == 2)
        #expect(tokenizer.bpeRanks[BytePair("c", "a")] == 3)
        #expect(tokenizer.bpeRanks[BytePair("A", "\u{30A}")] == 4)
    }

    @Test("A lone surrogate keeps its UTF-16 value and its string view maps to U+FFFD, as before", arguments: loneSurrogates)
    func loneSurrogateValue(_ units: [UInt16]) {
        let ns = String(utf16CodeUnits: units, count: units.count) as NSString
        let legacyString = Legacy.string(units)
        let new = BinaryDistinctString(units)

        // The string view repairs to U+FFFD on both implementations.
        #expect(Array(new.string.utf8) == Array(legacyString.utf8))
        #expect(Array(new.string.unicodeScalars.map(\.value)) == Array(legacyString.unicodeScalars.map(\.value)))
        #expect(new.string.unicodeScalars.contains("\u{FFFD}"))
        #expect(Array(new.string.utf16) == Array(legacyString.utf16))
        // On Darwin the NSString keeps the lone surrogate itself, so a vocabulary key is not folded into U+FFFD.
        #expect(utf16Units(new.nsString) == expectedNSStringUnits(units))
        #expect(utf16Units(new.nsString) == utf16Units(Legacy.nsString(units)))

        // Building from an NSString that carries a lone surrogate gives the same value as before.
        #expect(BinaryDistinctString(ns).value == Legacy.value(ns))
    }

    @Test("A lone-surrogate vocabulary key stays distinct from the replacement character")
    func loneSurrogateInVocab() throws {
        // yyjson rejects a lone `\ud800` escape ("no low surrogate in string"), so a tokenizer.json
        // cannot carry one; a `Config` built in code can.
        let lone = BinaryDistinctString([0xD800])
        let vocab = Config([
            BinaryDistinctString("<unk>"): Config(0), BinaryDistinctString("a"): Config(1),
            lone: Config(2), BinaryDistinctString("b"): Config(3),
        ])
        let model = Config([
            BinaryDistinctString("type"): Config("BPE"), BinaryDistinctString("vocab"): vocab,
            BinaryDistinctString("merges"): Config([Config("a b")]),
        ])
        let data = Config([BinaryDistinctString("model"): model])
        let config = Config([BinaryDistinctString("unk_token"): Config("<unk>")])
        let tokenizer = try BPETokenizer(tokenizerConfig: config, tokenizerData: data, addedTokens: [:])
        #expect(tokenizer.vocabCount == 4)
        #expect(tokenizer.convertTokenToId("a") == 1)
        #expect(tokenizer.convertTokenToId("b") == 3)
        // Vocabulary keys retain their original code units on every platform.
        // A lookup for U+FFFD misses and falls back to <unk> (id 0).
        #expect(tokenizer.convertTokenToId("\u{FFFD}") == 0)
        // Decoding the token yields the repaired string.
        #expect(tokenizer.convertIdToToken(2).map { Array($0.utf16) } == [0xFFFD])
    }
}
