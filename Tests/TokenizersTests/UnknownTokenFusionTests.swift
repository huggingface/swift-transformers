import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

/// Splits every character into its own pre-tokenized word.
private let splitEachCharacter: [String: Any] = [
    "type": "Split", "pattern": ["Regex": "."], "behavior": "Isolated", "invert": false,
]

private func makeTokenizer(model: [String: Any], preTokenizer: [String: Any]? = nil, tokenizerClass: String) throws -> Tokenizer {
    var data: [String: Any] = ["model": model]
    data["pre_tokenizer"] = preTokenizer
    let tokenizerData = try JSONDecoder().decode(Config.self, from: JSONSerialization.data(withJSONObject: data))
    // Like real tokenizer_config.json files: `unk_token` but no `fuse_unk`.
    let tokenizerConfig = try JSONDecoder().decode(
        Config.self,
        from: JSONSerialization.data(withJSONObject: ["tokenizer_class": tokenizerClass, "unk_token": "<unk>"])
    )
    return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
}

private func makeBPE(fuseUnknown: Bool, preTokenizer: [String: Any]? = nil) throws -> Tokenizer {
    try makeTokenizer(
        model: [
            "type": "BPE", "vocab": ["<unk>": 0, "A": 1, "L": 2], "merges": [String](),
            "unk_token": "<unk>", "fuse_unk": fuseUnknown,
        ],
        preTokenizer: preTokenizer,
        tokenizerClass: "LlamaTokenizer"
    )
}

private func makeUnigram(preTokenizer: [String: Any]? = nil) throws -> Tokenizer {
    try makeTokenizer(
        model: ["type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0.0], ["a", -1.0], ["b", -1.0]]],
        preTokenizer: preTokenizer,
        tokenizerClass: "T5Tokenizer"
    )
}

/// Expected ids from `tokenizers` 0.23.2.
@Suite("Unknown token fusion")
struct UnknownTokenFusionTests {
    @Test("BPE reads fuse_unk from the model in tokenizer.json", arguments: [(true, [1, 0, 2]), (false, [1, 0, 0, 2])])
    func bpeFuseUnknownFromModel(fuseUnknown: Bool, expected: [Int]) throws {
        let tokenizer = try makeBPE(fuseUnknown: fuseUnknown)
        #expect(tokenizer.encode(text: "A🐈🐈L") == expected)
    }

    @Test("BPE does not fuse unknown tokens across pre-tokenized words")
    func bpeFusionIsPerWord() throws {
        let tokenizer = try makeBPE(fuseUnknown: true, preTokenizer: splitEachCharacter)
        #expect(tokenizer.encode(text: "A🐈🐈L") == [1, 0, 0, 2])
    }

    @Test("Unigram fuses unknown tokens within a word")
    func unigramFusionWithinWord() throws {
        let tokenizer = try makeUnigram()
        #expect(tokenizer.encode(text: "a🐈🐈b") == [1, 0, 2])
    }

    @Test("Unigram does not fuse unknown tokens across pre-tokenized words")
    func unigramFusionIsPerWord() throws {
        let tokenizer = try makeUnigram(preTokenizer: splitEachCharacter)
        #expect(tokenizer.encode(text: "a🐈🐈b") == [1, 0, 0, 2])
    }
}
