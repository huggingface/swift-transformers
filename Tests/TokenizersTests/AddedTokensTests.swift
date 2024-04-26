//
//  AddedTokensTests.swift
//
//
//  Created by Pedro Cuenca on 20240426.
//

import XCTest
import Tokenizers
import Hub

class AddedTokensTests: XCTestCase {
    func testPhiAddedEnd() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "mlx-community/Phi-3-mini-128k-instruct-4bit")
        let inputIds = tokenizer("This is the <|end|>. My only friend, the <|end|>")
        XCTAssertEqual(inputIds, [1, 910, 338, 278, 29871, 32007, 29889, 1619, 871, 5121, 29892, 278, 29871, 32007])

        let decoded = tokenizer.decode(tokens: inputIds)
        XCTAssertEqual(decoded, "<s> This is the <|end|>. My only friend, the <|end|>")
    }

    func testSplitWithCaptureGroups() {
        let addedTokensRegexp = #"(<\|end\|>)\s*|(<\|raw\|>)\s*"#
        let captureRegex = try! NSRegularExpression(pattern: addedTokensRegexp, options: [])

        XCTAssertEqual(
            "eating <|raw|> meat <|end|> That's all".split(by: captureRegex),
            ["eating ", "<|raw|>", "meat ", "<|end|>", "That's all"]
        )

        XCTAssertEqual(
            "<|raw|>".split(by: captureRegex),
            ["<|raw|>"]
        )

        XCTAssertEqual(
            "This string doesn't have those separators".split(by: captureRegex),
            ["This string doesn't have those separators"]
        )

        XCTAssertEqual(
            "start <|end|>".split(by: captureRegex),
            ["start ", "<|end|>"]
        )

        XCTAssertEqual(
            "start <|end|> ".split(by: captureRegex),
            ["start ", "<|end|>"]
        )

        XCTAssertEqual(
            "start <|end|>       ".split(by: captureRegex),
            ["start ", "<|end|>"]
        )

        XCTAssertEqual(
            "start <|end|>       for real".split(by: captureRegex),
            ["start ", "<|end|>", "for real"]
        )

        XCTAssertEqual(
            "<|raw|><|end|>".split(by: captureRegex),
            ["<|raw|>", "<|end|>"]
        )

    }
}
