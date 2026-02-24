//
//  PreTokenizerTests.swift
//
//  Created by Jan Krukowski on 23/11/2023.
//

import Foundation
import Hub
import Testing

@testable import Tokenizers

@Suite("Pre-Tokenizer Tests")
struct PreTokenizerTests {
    @Test("Whitespace pre-tokenizer splits text by whitespace")
    func whitespacePreTokenizer() {
        let preTokenizer = WhitespacePreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer.preTokenize(text: "Hey friend!") == ["Hey", "friend!"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", "friend!", "How", "are", "you?!?",
            ]
        )
        #expect(
            preTokenizer.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "Hey,", "friend,", "what's", "up?",
            ]
        )
    }

    @Test("Punctuation pre-tokenizer separates punctuation")
    func punctuationPreTokenizer() {
        let preTokenizer = PunctuationPreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer.preTokenize(text: "Hey friend!") == ["Hey friend", "!"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey friend", "!", "     How are you", "?!?",
            ]
        )
        #expect(
            preTokenizer.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "   Hey", ",", "    friend", ",", "    what", "'", "s up", "?", "  ",
            ]
        )
    }

    @Test("Byte-level pre-tokenizer with various configurations")
    func byteLevelPreTokenizer() {
        let preTokenizer1 = ByteLevelPreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!") == ["Hey", "Ġfriend", "!"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", "Ġfriend", "!", "ĠĠĠĠ", "ĠHow", "Ġare", "Ġyou", "?!?",
            ]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "ĠĠ", "ĠHey", ",", "ĠĠĠ", "Ġfriend", ",", "ĠĠĠ", "Ġwhat", "'s", "Ġup", "?", "ĠĠ",
            ]
        )

        let preTokenizer2 = ByteLevelPreTokenizer(config: Config(["addPrefixSpace": true]))

        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!") == ["ĠHey", "Ġfriend", "Ġ!"]
        )
        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!     How are you?!?") == [
                "ĠHey", "Ġfriend", "Ġ!", "ĠĠĠĠ", "ĠHow", "Ġare", "Ġyou", "Ġ?!?",
            ]
        )
        #expect(
            preTokenizer2.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "ĠĠ", "ĠHey", "Ġ,", "ĠĠĠ", "Ġfriend", "Ġ,", "ĠĠĠ", "Ġwhat", "Ġ's", "Ġup", "Ġ?",
                "ĠĠ",
            ]
        )

        let preTokenizer3 = ByteLevelPreTokenizer(config: Config(["useRegex": false]))

        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!") == ["HeyĠfriend!"]
        )
        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!     How are you?!?") == [
                "HeyĠfriend!ĠĠĠĠĠHowĠareĠyou?!?"
            ]
        )
        #expect(
            preTokenizer3.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "ĠĠĠHey,ĠĠĠĠfriend,ĠĠĠĠwhat'sĠup?ĠĠ"
            ]
        )
    }

    @Test("Digits pre-tokenizer handles numeric content")
    func digitsPreTokenizer() {
        let preTokenizer1 = DigitsPreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer1.preTokenize(text: "1 12 123! 1234abc") == [
                "1", " ", "12", " ", "123", "! ", "1234", "abc",
            ]
        )

        let preTokenizer2 = DigitsPreTokenizer(config: Config(["individualDigits": true]))

        #expect(
            preTokenizer2.preTokenize(text: "1 12 123! 1234abc") == [
                "1", " ", "1", "2", " ", "1", "2", "3", "! ", "1", "2", "3", "4", "abc",
            ]
        )
    }

    @Test("Split pre-tokenizer with string and regex patterns")
    func splitPreTokenizer() {
        let preTokenizer1 = SplitPreTokenizer(config: Config(["pattern": ["String": " "]]))
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!") == ["Hey", " ", "friend!"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", " ", "friend!", " ", " ", " ", " ", " ", "How", " ", "are", " ", "you?!?",
            ]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                " ", " ", " ", "Hey,", " ", " ", " ", " ", "friend,", " ", " ", " ", " ", "what's",
                " ", "up?", " ", " ",
            ]
        )

        let preTokenizer2 = SplitPreTokenizer(config: Config(["pattern": ["Regex": "\\s"]]))
        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!") == ["Hey", " ", "friend!"]
        )
        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", " ", "friend!", " ", " ", " ", " ", " ", "How", " ", "are", " ", "you?!?",
            ]
        )
        #expect(
            preTokenizer2.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                " ", " ", " ", "Hey,", " ", " ", " ", " ", "friend,", " ", " ", " ", " ", "what's",
                " ", "up?", " ", " ",
            ]
        )

        let preTokenizer3 = SplitPreTokenizer(
            config: Config([
                "pattern": [
                    "Regex":
                        "(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                ], "invert": true,
            ]))
        #expect(
            preTokenizer3.preTokenize(text: "Hello") == ["Hello"]
        )

        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!") == ["Hey", " friend", "!"]
        )
        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", " friend", "!", "    ", " How", " are", " you", "?!?",
            ]
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

    @Test("Split behavior other")
    func splitBehaviorOther() {
        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .isolated) == ["the", "-", "final", "-", "-", "countdown"]
        )

        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .removed) == ["the", "final", "countdown"]
        )
    }

    /// https://github.com/huggingface/tokenizers/pull/1357
    @Test("Metaspace pre-tokenizer with prefix space handling")
    func metaspacePreTokenizer() {
        // Prepend "always"
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "add_prefix_space": true,
                "replacement": "▁",
                "prepend_scheme": "always",
            ]))

        // TODO: different sections on <s>
        let text = "Hey my friend <s>how▁are you"
        let tokens =
            text
            .split(by: "<s>", includeSeparators: true)
            .flatMap { preTokenizer.preTokenize(text: $0) }

        #expect(
            tokens == ["▁Hey", "▁my", "▁friend", "▁", "▁<s>", "▁how", "▁are", "▁you"]
        )
    }

    @Test("Metaspace prepend_scheme 'always' without add_prefix_space (XLM-RoBERTa case)")
    func metaspacePrependSchemeAlwaysWithoutAddPrefixSpace() {
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "prepend_scheme": "always",
            ]))

        #expect(
            preTokenizer.preTokenize(text: "Hello") == ["▁Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world") == ["▁Hello", "▁world"]
        )
        // Already starts with replacement — no double prepend
        #expect(
            preTokenizer.preTokenize(text: "▁Hello") == ["▁Hello"]
        )
    }

    @Test("Metaspace prepend_scheme 'first' only prepends on first section")
    func metaspacePrependSchemeFirst() {
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "prepend_scheme": "first",
            ]))

        // First section (default options include .firstSection)
        #expect(
            preTokenizer.preTokenize(text: "Hello") == ["▁Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world") == ["▁Hello", "▁world"]
        )

        // Non-first section (empty options)
        #expect(
            preTokenizer.preTokenize(text: "Hello", options: []) == ["Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world", options: []) == ["Hello", "▁world"]
        )
    }

    @Test("Metaspace prepend_scheme 'never' never prepends")
    func metaspacePrependSchemeNever() {
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "prepend_scheme": "never",
            ]))

        #expect(
            preTokenizer.preTokenize(text: "Hello") == ["Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world") == ["Hello", "▁world"]
        )
    }

    @Test("Metaspace legacy add_prefix_space without prepend_scheme")
    func metaspaceLegacyAddPrefixSpace() {
        // add_prefix_space: true, no prepend_scheme → behaves like "always"
        let alwaysTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "add_prefix_space": true,
            ]))

        #expect(
            alwaysTokenizer.preTokenize(text: "Hello") == ["▁Hello"]
        )
        #expect(
            alwaysTokenizer.preTokenize(text: "Hello world") == ["▁Hello", "▁world"]
        )

        // add_prefix_space: false, no prepend_scheme → behaves like "never"
        let neverTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "add_prefix_space": false,
            ]))

        #expect(
            neverTokenizer.preTokenize(text: "Hello") == ["Hello"]
        )
        #expect(
            neverTokenizer.preTokenize(text: "Hello world") == ["Hello", "▁world"]
        )
    }

    @Test("Metaspace default config (no add_prefix_space, no prepend_scheme) defaults to always")
    func metaspaceDefaultConfig() {
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁"
            ]))

        #expect(
            preTokenizer.preTokenize(text: "Hello") == ["▁Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world") == ["▁Hello", "▁world"]
        )
    }

    @Test("Metaspace prepend_scheme supersedes add_prefix_space")
    func metaspacePrependSchemeSupersedesAddPrefixSpace() {
        // prepend_scheme: "always" wins even when add_prefix_space is false
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "add_prefix_space": false,
                "prepend_scheme": "always",
            ]))

        #expect(
            preTokenizer.preTokenize(text: "Hello") == ["▁Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world") == ["▁Hello", "▁world"]
        )
    }

    @Test("Metaspace prepend_scheme 'never' supersedes add_prefix_space true")
    func metaspacePrependSchemeNeverSupersedesAddPrefixSpace() {
        // prepend_scheme: "never" wins even when add_prefix_space is true
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "add_prefix_space": true,
                "prepend_scheme": "never",
            ]))

        #expect(
            preTokenizer.preTokenize(text: "Hello") == ["Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world") == ["Hello", "▁world"]
        )
    }

    @Test("Metaspace handles empty string input")
    func metaspaceEmptyString() {
        let always = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "prepend_scheme": "always",
            ]))
        #expect(
            always.preTokenize(text: "") == ["▁"]
        )

        let never = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "prepend_scheme": "never",
            ]))
        #expect(
            never.preTokenize(text: "") == []
        )
    }

    @Test("BERT pre-tokenizer performs basic splitting")
    func bertPreTokenizer() {
        let preTokenizer1 = BertPreTokenizer(config: Config([String: Config]()))
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!") == ["Hey", "friend", "!"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", "friend", "!", "How", "are", "you", "?", "!", "?",
            ]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend ,    what's up?  ") == [
                "Hey", ",", "friend", ",", "what", "\'", "s", "up", "?",
            ]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend ,  0 99  what's up?  ") == [
                "Hey", ",", "friend", ",", "0", "99", "what", "\'", "s", "up", "?",
            ]
        )
    }
}
