//
//  AddedTokensTests.swift
//
//
//  Created by Pedro Cuenca on 20240426.
//

import Foundation
import Hub
import Testing
import Tokenizers

@Suite("Added Tokens Tests")
struct AddedTokensTests {
    @Test("Phi model added tokens handling")
    func phiAddedTokens() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "microsoft/Phi-3-mini-128k-instruct")
        let inputIds = tokenizer("This is the <|end|>. My only friend, the <|end|>")
        #expect(inputIds == [910, 338, 278, 29871, 32007, 29889, 1619, 871, 5121, 29892, 278, 29871, 32007])

        let decoded = tokenizer.decode(tokens: inputIds)
        #expect(decoded == "This is the <|end|>. My only friend, the <|end|>")
    }

    @Test("Gemma model added tokens handling")
    func gemmaAddedTokens() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "pcuenq/gemma-tokenizer")
        let inputIds = tokenizer("This\n\nis\na\ntest.")
        #expect(inputIds == [2, 1596, 109, 502, 108, 235250, 108, 2195, 235265])

        let decoded = tokenizer.decode(tokens: inputIds)
        #expect(decoded == "<bos>This\n\nis\na\ntest.")
    }

    @Test("String splitting with capture groups")
    func splitWithCaptureGroups() {
        let addedTokensRegexp = #"(<\|end\|>)\s*|(<\|raw\|>)\s*"#
        let captureRegex = try! NSRegularExpression(pattern: addedTokensRegexp, options: [])

        #expect(
            "eating <|raw|> meat <|end|> That's all".split(by: captureRegex) ==
                ["eating ", "<|raw|>", "meat ", "<|end|>", "That's all"]
        )

        #expect(
            "<|raw|>".split(by: captureRegex) ==
                ["<|raw|>"]
        )

        #expect(
            "This string doesn't have those separators".split(by: captureRegex) ==
                ["This string doesn't have those separators"]
        )

        #expect(
            "start <|end|>".split(by: captureRegex) ==
                ["start ", "<|end|>"]
        )

        #expect(
            "start <|end|> ".split(by: captureRegex) ==
                ["start ", "<|end|>"]
        )

        #expect(
            "start <|end|>       ".split(by: captureRegex) ==
                ["start ", "<|end|>"]
        )

        #expect(
            "start <|end|>       for real".split(by: captureRegex) ==
                ["start ", "<|end|>", "for real"]
        )

        #expect(
            "<|raw|><|end|>".split(by: captureRegex) ==
                ["<|raw|>", "<|end|>"]
        )
    }
}
