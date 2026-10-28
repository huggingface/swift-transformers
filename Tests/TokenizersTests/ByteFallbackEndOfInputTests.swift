import Foundation
import Hub
import Testing

@testable import Tokenizers

@Suite("Byte fallback end-of-input")
struct ByteFallbackEndOfInputTests {
    @Test("Terminal UTF-8 bytes decode without a following ordinary token")
    func terminalEmoji() {
        let decoder = ByteFallbackDecoder(config: Config(["type": "ByteFallback"] as [NSString: Any]))
        #expect(decoder.decode(tokens: ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"]) == ["😀"])
    }

    @Test("Interior and terminal byte runs retain their order")
    func mixedRuns() {
        let decoder = ByteFallbackDecoder(config: Config(["type": "ByteFallback"] as [NSString: Any]))
        #expect(decoder.decode(tokens: ["a", "<0x62>", "c", "<0x20>"]) == ["a", "b", "c", " "])
        #expect(decoder.decode(tokens: []) == [])
        #expect(decoder.decode(tokens: ["ordinary"]) == ["ordinary"])
    }

    private func pipeline() -> Config {
        Config(
            [
                "type": "Sequence",
                "decoders": [
                    ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
                    ["type": "ByteFallback"],
                    ["type": "Fuse"],
                    ["type": "Strip", "content": " ", "start": 0, "stop": 1],
                ],
            ] as [NSString: Any])
    }

    @Test("A terminal byte-spelled space still passes through downstream Fuse and Strip")
    func downstreamCleanup() throws {
        let decoder = try #require(try DecoderFactory.fromConfig(config: pipeline()))
        #expect(decoder.decode(tokens: ["<0x20>"]) == [""])
        #expect(decoder.decode(tokens: ["a", "<0x20>"]) == ["a"])
        #expect(decoder.decode(tokens: ["▁"]) == [""])
    }

    @Test("PreTrainedTokenizer preserves terminal bytes, special-token filtering and configured cleanup")
    func configuredTokenizer() throws {
        let config = Config(
            [
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "eos_token": "</s>",
                "clean_up_tokenization_spaces": false,
            ] as [NSString: Any])
        let data = Config(
            [
                "added_tokens": [
                    ["id": 0, "content": "<unk>", "special": true],
                    ["id": 1, "content": "</s>", "special": true],
                ],
                "decoder": pipeline(),
                "model": [
                    "type": "BPE", "unk_token": "<unk>", "byte_fallback": true,
                    "vocab": [
                        "<unk>": 0, "</s>": 1, "a": 2, "▁": 3,
                        "<0x20>": 4, "<0xF0>": 5, "<0x9F>": 6, "<0x98>": 7, "<0x80>": 8,
                    ],
                    "merges": [String](),
                ],
            ] as [NSString: Any])
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: config, tokenizerData: data)
        #expect(tokenizer.decode(tokens: [5, 6, 7, 8]) == "😀")
        #expect(tokenizer.decode(tokens: [4]) == tokenizer.decode(tokens: [3]))
        #expect(tokenizer.decode(tokens: [2, 4]) == "a")
        #expect(tokenizer.decode(tokens: [1, 5, 6, 7, 8], skipSpecialTokens: true) == "😀")
        #expect(tokenizer.decode(tokens: [1, 5, 6, 7, 8], skipSpecialTokens: false) == "</s>😀")
    }
}
