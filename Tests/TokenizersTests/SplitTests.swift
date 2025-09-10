//
//  SplitTests.swift
//
//
//  Created by Pedro Cuenca on 20240120.
//

import Foundation
import Testing
import Tokenizers

@Suite struct SplitTests {
    @Test func testSplitBehaviorMergedWithPrevious() {
        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) ==
            ["the-", "final-", "-", "countdown"]
        )

        #expect(
            "the-final--countdown-".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) ==
            ["the-", "final-", "-", "countdown-"]
        )

        #expect(
            "the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) ==
            ["the-", "final-", "-", "countdown-", "-"]
        )

        #expect(
            "-the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) ==
            ["-", "the-", "final-", "-", "countdown-", "-"]
        )

        #expect(
            "--the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) ==
            ["-", "-", "the-", "final-", "-", "countdown-", "-"]
        )
    }

    @Test func testSplitBehaviorMergedWithNext() {
        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) ==
            ["the", "-final", "-", "-countdown"]
        )

        #expect(
            "-the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) ==
            ["-the", "-final", "-", "-countdown"]
        )

        #expect(
            "--the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) ==
            ["-", "-the", "-final", "-", "-countdown"]
        )

        #expect(
            "--the-final--countdown-".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) ==
            ["-", "-the", "-final", "-", "-countdown", "-"]
        )
    }

    @Test func testSplitBehaviorOther() {
        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .isolated) ==
            ["the", "-", "final", "-", "-", "countdown"]
        )

        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .removed) ==
            ["the", "final", "countdown"]
        )
    }
}
