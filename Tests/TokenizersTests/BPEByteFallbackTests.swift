import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

private let bpeByteFallbackVocab: [String: Int] = [
    "<unk>": 0, "A": 1, "L": 2,
    "<0xF0>": 3, "<0x9F>": 4, "<0x90>": 5, "<0x88>": 6,
]

private func makeBPETokenizer(
    byteFallback: Bool?,
    vocab: [String: Int] = bpeByteFallbackVocab,
    unknownToken: String? = "<unk>"
) throws -> BPETokenizer {
    var model: [String: Any] = [
        "type": "BPE", "vocab": vocab, "merges": [String](), "fuse_unk": false,
    ]
    model["byte_fallback"] = byteFallback
    model["unk_token"] = unknownToken
    var config: [String: Any] = [:]
    config["unk_token"] = unknownToken
    let tokenizerData = try JSONDecoder().decode(
        Config.self, from: JSONSerialization.data(withJSONObject: ["model": model])
    )
    let tokenizerConfig = try JSONDecoder().decode(
        Config.self, from: JSONSerialization.data(withJSONObject: config)
    )
    return try BPETokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: [:])
}

@Suite("BPE byte fallback")
struct BPEByteFallbackTests {
    @Test("An unknown multibyte character produces one unknown token when byte fallback is disabled")
    func alphabetOnlyVocabulary() throws {
        // Reduced from the alphabet-only BPE configuration published at:
        // https://huggingface.co/kojima-lab/molcrawl-protein-sequence-proteingym-gpt2-xl/commit/cae49d07f7947108c1d6252778934abef3ba921a
        let tokenizer = try makeBPETokenizer(byteFallback: false, vocab: ["<unk>": 0, "A": 1, "L": 2])
        let tokens = tokenizer.tokenize(text: "A🐈L")
        #expect(tokens == ["A", "<unk>", "L"])
        #expect(tokens.map { tokenizer.convertTokenToId($0) } == [1, 0, 2])
    }

    @Test("Disabled or absent byte fallback ignores available byte tokens", arguments: [false, nil] as [Bool?])
    func disabledByteFallback(byteFallback: Bool?) throws {
        let tokenizer = try makeBPETokenizer(byteFallback: byteFallback)
        #expect(tokenizer.tokenize(text: "🐈") == ["<unk>"])
    }

    @Test("Enabled byte fallback preserves consecutive UTF-8 byte sequences")
    func enabledByteFallback() throws {
        let tokenizer = try makeBPETokenizer(byteFallback: true)
        let bytes = ["<0xF0>", "<0x9F>", "<0x90>", "<0x88>"]
        #expect(tokenizer.tokenize(text: "A🐈🐈L") == ["A"] + bytes + bytes + ["L"])
    }

    @Test("Incomplete byte vocabulary produces one unknown token")
    func incompleteByteVocabulary() throws {
        let tokenizer = try makeBPETokenizer(
            byteFallback: true, vocab: bpeByteFallbackVocab.filter { $0.key != "<0x88>" }
        )
        #expect(tokenizer.tokenize(text: "🐈") == ["<unk>"])
    }

    @Test("Unknown characters are omitted when no unknown token is configured", arguments: [false, true])
    func noUnknownToken(byteFallback: Bool) throws {
        let tokenizer = try makeBPETokenizer(
            byteFallback: byteFallback, vocab: ["A": 1, "L": 2], unknownToken: nil
        )
        #expect(tokenizer.tokenize(text: "A🐈L") == ["A", "L"])
    }
}
