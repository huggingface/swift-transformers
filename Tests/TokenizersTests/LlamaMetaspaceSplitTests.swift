import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

/// A non-legacy Llama tokenizer gets its Metaspace pre-tokenizer from LlamaPreTrainedTokenizer.
/// Expected ids from `tokenizers` 0.23.3 with a Metaspace pre-tokenizer (prepend_scheme first, split false) and the same BPE model.
@Suite("Non-legacy Llama Metaspace split")
struct LlamaMetaspaceSplitTests {
    private func makeTokenizer(legacy: Bool, preTokenizer: [String: Any]? = nil) throws -> Tokenizer {
        let model: [String: Any] = [
            "type": "BPE", "vocab": ["<unk>": 0, "▁": 1, "a": 2, "b": 3, "▁▁": 4, "▁a": 5, "▁b": 6],
            "merges": ["▁ ▁", "▁ a", "▁ b"], "unk_token": "<unk>",
        ]
        var data: [String: Any] = ["model": model]
        data["pre_tokenizer"] = preTokenizer
        let tokenizerData = try JSONDecoder().decode(Config.self, from: JSONSerialization.data(withJSONObject: data))
        let tokenizerConfig = try JSONDecoder().decode(
            Config.self,
            from: JSONSerialization.data(withJSONObject: ["tokenizer_class": "LlamaTokenizer", "unk_token": "<unk>", "legacy": legacy])
        )
        return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    @Test("Runs of spaces stay in one piece")
    func runsOfSpaces() throws {
        let tokenizer = try makeTokenizer(legacy: false)
        #expect(tokenizer.encode(text: "a  b", addSpecialTokens: false) == [5, 4, 3])
        #expect(tokenizer.encode(text: "a   b", addSpecialTokens: false) == [5, 4, 6])
        #expect(tokenizer.encode(text: "a b", addSpecialTokens: false) == [5, 6])
    }

    @Test("A Metaspace pre-tokenizer in tokenizer.json keeps its split setting")
    func tokenizerJSONSplitSetting() throws {
        let splitTrue = try makeTokenizer(
            legacy: false, preTokenizer: ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": true])
        #expect(splitTrue.encode(text: "a  b", addSpecialTokens: false) == [5, 1, 6])

        // Like the Rust pre-tokenizer, a missing split key means true.
        let splitMissing = try makeTokenizer(legacy: false, preTokenizer: ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "first"])
        #expect(splitMissing.encode(text: "a  b", addSpecialTokens: false) == [5, 1, 6])

        let splitFalse = try makeTokenizer(
            legacy: false, preTokenizer: ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false])
        #expect(splitFalse.encode(text: "a  b", addSpecialTokens: false) == [5, 4, 3])
    }
}
