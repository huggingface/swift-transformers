import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

// "hugs" is in the vocabulary, but the merges stop at "hug" + "s".
private let bpeIgnoreMergesVocab: [String: Int] = [
    "h": 0, "u": 1, "g": 2, "s": 3, "ug": 4, "hug": 5, "hugs": 6,
]
private let bpeIgnoreMergesMerges: [[String]] = [["u", "g"], ["h", "ug"]]

private func makeBPETokenizer(ignoreMerges: Bool?) throws -> BPETokenizer {
    var model: [String: Any] = [
        "type": "BPE", "vocab": bpeIgnoreMergesVocab, "merges": bpeIgnoreMergesMerges,
    ]
    model["ignore_merges"] = ignoreMerges
    let tokenizerData = try JSONDecoder().decode(
        Config.self, from: JSONSerialization.data(withJSONObject: ["model": model])
    )
    let tokenizerConfig = try JSONDecoder().decode(
        Config.self, from: JSONSerialization.data(withJSONObject: [String: Any]())
    )
    return try BPETokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: [:])
}

@Suite("BPE ignore_merges")
struct BPEIgnoreMergesTests {
    @Test("A word in the vocabulary is one token when ignore_merges is true")
    func wholeWordInVocabulary() throws {
        let tokenizer = try makeBPETokenizer(ignoreMerges: true)
        let tokens = tokenizer.tokenize(text: "hugs")
        #expect(tokens == ["hugs"])
        #expect(tokens.map { tokenizer.convertTokenToId($0) } == [6])
    }

    @Test("Disabled or absent ignore_merges applies the merges", arguments: [false, nil] as [Bool?])
    func mergesApplied(ignoreMerges: Bool?) throws {
        let tokenizer = try makeBPETokenizer(ignoreMerges: ignoreMerges)
        #expect(tokenizer.tokenize(text: "hugs") == ["hug", "s"])
    }

    @Test("A word not in the vocabulary still goes through the merges", arguments: [true, false])
    func wordNotInVocabulary(ignoreMerges: Bool) throws {
        let tokenizer = try makeBPETokenizer(ignoreMerges: ignoreMerges)
        #expect(tokenizer.tokenize(text: "shug") == ["s", "hug"])
    }
}
