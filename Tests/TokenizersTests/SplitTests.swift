//
//  SplitTests.swift
//
//
//  Created by Pedro Cuenca on 20240120.
//

import Foundation
import Testing
import Tokenizers

@Suite("Split Behavior Tests")
struct SplitTests {
    @Test("String splitting with capture groups")
    func splitWithCaptureGroups() {
        let addedTokensRegexp = #"(<\|end\|>)\s*|(<\|raw\|>)\s*"#
        let captureRegex = try! NSRegularExpression(pattern: addedTokensRegexp, options: [])

        #expect(
            "eating <|raw|> meat <|end|> That's all".split(by: captureRegex) == ["eating ", "<|raw|>", "meat ", "<|end|>", "That's all"]
        )

        #expect(
            "<|raw|>".split(by: captureRegex) == ["<|raw|>"]
        )

        #expect(
            "This string doesn't have those separators".split(by: captureRegex) == ["This string doesn't have those separators"]
        )

        #expect(
            "start <|end|>".split(by: captureRegex) == ["start ", "<|end|>"]
        )

        #expect(
            "start <|end|> ".split(by: captureRegex) == ["start ", "<|end|>"]
        )

        #expect(
            "start <|end|>       ".split(by: captureRegex) == ["start ", "<|end|>"]
        )

        #expect(
            "start <|end|>       for real".split(by: captureRegex) == ["start ", "<|end|>", "for real"]
        )

        #expect(
            "<|raw|><|end|>".split(by: captureRegex) == ["<|raw|>", "<|end|>"]
        )
    }

    @Test("Split behavior merged with previous")
    func splitBehaviorMergedWithPrevious() {
        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == ["the-", "final-", "-", "countdown"]
        )

        #expect(
            "the-final--countdown-".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == ["the-", "final-", "-", "countdown-"]
        )

        #expect(
            "the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == ["the-", "final-", "-", "countdown-", "-"]
        )

        #expect(
            "-the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == ["-", "the-", "final-", "-", "countdown-", "-"]
        )

        #expect(
            "--the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == ["-", "-", "the-", "final-", "-", "countdown-", "-"]
        )
    }

    @Test("Split behavior merged with next")
    func splitBehaviorMergedWithNext() {
        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) == ["the", "-final", "-", "-countdown"]
        )

        #expect(
            "-the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) == ["-the", "-final", "-", "-countdown"]
        )

        #expect(
            "--the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) == ["-", "-the", "-final", "-", "-countdown"]
        )

        #expect(
            "--the-final--countdown-".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) == ["-", "-the", "-final", "-", "-countdown", "-"]
        )
    }

    @Test("Split behavior isolated and removed")
    func splitBehaviorOther() {
        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .isolated) == ["the", "-", "final", "-", "-", "countdown"]
        )

        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .removed) == ["the", "final", "countdown"]
        )
    }
}
