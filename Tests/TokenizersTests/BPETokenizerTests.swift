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
}
