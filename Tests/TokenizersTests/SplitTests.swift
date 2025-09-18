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
    @Test("Split behavior merged with previous")
    func splitBehaviorMergedWithPrevious() {
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

    @Test("Split behavior merged with next")
    func splitBehaviorMergedWithNext() {
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

    @Test("Split behavior isolated and removed")
    func splitBehaviorOther() {
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
