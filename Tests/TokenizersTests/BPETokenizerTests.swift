import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

@Suite("BPE tokenizer")
struct BPETokenizerTests {
    @Test("Merges distinguish canonically equivalent symbols")
    func canonicallyEquivalentMerges() throws {
        let vocab: [BinaryDistinctString: Config] = ["▁": 0, ";": 1, "\u{037E}": 2, "▁;": 3]
        let data = Config(["model": Config(["vocab": Config(vocab), "merges": [["▁", ";"]]])])
        let tokenizer = try BPETokenizer(tokenizerConfig: Config([String: Config]()), tokenizerData: data, addedTokens: [:])
        #expect(BytePair("▁", ";") != BytePair("▁", "\u{037E}"))
        #expect(tokenizer.bpe(token: "▁;") == ["▁;"])
        #expect(tokenizer.bpe(token: "▁\u{037E}") == ["▁", "\u{037E}"])
    }

    @Test("Merge ranks count only list and legacy entries, and a repeated pair keeps its last rank")
    func mergeRanks() throws {
        let merges: Config = [["a", "b"], "c d", 5, ["e", 1, "f"], ["▁", "a"], ["a", "b"]]
        let data = Config(["model": Config(["vocab": Config(["a": 0, "b": 1] as [BinaryDistinctString: Config]), "merges": merges])])
        let ranks = try BPETokenizer(tokenizerConfig: Config([String: Config]()), tokenizerData: data, addedTokens: [:]).bpeRanks
        #expect(ranks.count == 4)
        #expect(ranks[BytePair("a", "b")] == 4)
        #expect(ranks[BytePair("c", "d")] == 1)
        #expect(ranks[BytePair("e", "f")] == 2)
        #expect(ranks[BytePair("▁", "a")] == 3)
    }

    @Test("A pair hashes and compares the same whether its strings are native or bridged")
    func bridgedPairHash() {
        let native = BytePair("\u{2581}caf\u{E9}", "\u{1F408}x")
        let units = Array("\u{2581}caf\u{E9}".utf16)
        let bridged = BytePair(String(utf16CodeUnits: units, count: units.count) as NSString as String, "\u{1F408}x")
        #expect(native == bridged)
        #expect(native.hashValue == bridged.hashValue)
        #expect([native: 7][bridged] == 7)
    }
}
