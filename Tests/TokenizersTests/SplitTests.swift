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
    @Test func splitBehaviorMergedWithPrevious() {
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

    @Test func splitBehaviorMergedWithNext() {
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

    @Test func splitBehaviorOther() {
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
