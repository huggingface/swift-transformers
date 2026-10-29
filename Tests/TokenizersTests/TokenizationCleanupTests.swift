import Testing

@testable import Hub
@testable import Tokenizers

@Suite("Tokenization cleanup")
struct TokenizationCleanupTests {
    @Test("Cleanup defaults to false and honors explicit settings", arguments: [nil, false, true] as [Bool?])
    func cleanupSetting(_ cleanup: Bool?) throws {
        var config: [String: Config] = ["tokenizer_class": "BertTokenizer"]
        if let cleanup {
            config["clean_up_tokenization_spaces"] = .init(cleanup)
        }
        // Disable decoder cleanup to isolate the tokenizer-level setting.
        let data = Config([
            "model": ["type": "WordPiece", "vocab": ["hello": 0, ",": 1, "world": 2, ".": 3]],
            "decoder": ["type": "WordPiece", "prefix": "##", "cleanup": false],
        ])
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: Config(config), tokenizerData: data)

        let expected = cleanup == true ? "hello, world." : "hello , world ."
        #expect(tokenizer.decode(tokens: [0, 1, 2, 3]) == expected)
    }

    @Test("BERT decoder cleanup works without a top-level cleanup setting")
    func bertDecoderCleanup() throws {
        let config = Config(["tokenizer_class": "BertTokenizer"])
        let data = Config([
            "model": ["type": "WordPiece", "vocab": ["hello": 0, ",": 1, "world": 2, ".": 3]],
            "decoder": ["type": "WordPiece", "prefix": "##", "cleanup": true],
        ])
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: config, tokenizerData: data)

        #expect(tokenizer.decode(tokens: [0, 1, 2, 3]) == "hello, world.")
    }
}
