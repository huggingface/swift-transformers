//
//  PreTokenizerTests.swift
//
//  Created by Jan Krukowski on 23/11/2023.
//

import Hub
@testable import Tokenizers
import Testing

@Suite
struct PreTokenizerTests {
    @Test
    func testWhitespacePreTokenizer() {
        let preTokenizer = WhitespacePreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer.preTokenize(text: "Hey friend!") ==
            ["Hey", "friend!"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hey friend!     How are you?!?") ==
            ["Hey", "friend!", "How", "are", "you?!?"]
        )
        #expect(
            preTokenizer.preTokenize(text: "   Hey,    friend,    what's up?  ") ==
            ["Hey,", "friend,", "what's", "up?"]
        )
    }

    @Test
    func testPunctuationPreTokenizer() {
        let preTokenizer = PunctuationPreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer.preTokenize(text: "Hey friend!") ==
            ["Hey friend", "!"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hey friend!     How are you?!?") ==
            ["Hey friend", "!", "     How are you", "?!?"]
        )
        #expect(
            preTokenizer.preTokenize(text: "   Hey,    friend,    what's up?  ") ==
            ["   Hey", ",", "    friend", ",", "    what", "'", "s up", "?", "  "]
        )
    }

    @Test
    func testByteLevelPreTokenizer() {
        let preTokenizer1 = ByteLevelPreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!") ==
            ["Hey", "Ġfriend", "!"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?") ==
            ["Hey", "Ġfriend", "!", "ĠĠĠĠ", "ĠHow", "Ġare", "Ġyou", "?!?"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend,    what's up?  ") ==
            ["ĠĠ", "ĠHey", ",", "ĠĠĠ", "Ġfriend", ",", "ĠĠĠ", "Ġwhat", "'s", "Ġup", "?", "ĠĠ"]
        )

        let preTokenizer2 = ByteLevelPreTokenizer(config: Config(["addPrefixSpace": true]))

        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!") ==
            ["ĠHey", "Ġfriend", "Ġ!"]
        )
        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!     How are you?!?") ==
            ["ĠHey", "Ġfriend", "Ġ!", "ĠĠĠĠ", "ĠHow", "Ġare", "Ġyou", "Ġ?!?"]
        )
        #expect(
            preTokenizer2.preTokenize(text: "   Hey,    friend,    what's up?  ") ==
            ["ĠĠ", "ĠHey", "Ġ,", "ĠĠĠ", "Ġfriend", "Ġ,", "ĠĠĠ", "Ġwhat", "Ġ's", "Ġup", "Ġ?", "ĠĠ"]
        )

        let preTokenizer3 = ByteLevelPreTokenizer(config: Config(["useRegex": false]))

        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!") ==
            ["HeyĠfriend!"]
        )
        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!     How are you?!?") ==
            ["HeyĠfriend!ĠĠĠĠĠHowĠareĠyou?!?"]
        )
        #expect(
            preTokenizer3.preTokenize(text: "   Hey,    friend,    what's up?  ") ==
            ["ĠĠĠHey,ĠĠĠĠfriend,ĠĠĠĠwhat'sĠup?ĠĠ"]
        )
    }

    @Test
    func testDigitsPreTokenizer() {
        let preTokenizer1 = DigitsPreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer1.preTokenize(text: "1 12 123! 1234abc") ==
            ["1", " ", "12", " ", "123", "! ", "1234", "abc"]
        )

        let preTokenizer2 = DigitsPreTokenizer(config: Config(["individualDigits": true]))

        #expect(
            preTokenizer2.preTokenize(text: "1 12 123! 1234abc") ==
            ["1", " ", "1", "2", " ", "1", "2", "3", "! ", "1", "2", "3", "4", "abc"]
        )
    }

    @Test
    func testSplitPreTokenizer() {
        let preTokenizer1 = SplitPreTokenizer(config: Config(["pattern": ["String": " "]]))
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!") ==
            ["Hey", " ", "friend!"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?") ==
            ["Hey", " ", "friend!", " ", " ", " ", " ", " ", "How", " ", "are", " ", "you?!?"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend,    what's up?  ") ==
            [" ", " ", " ", "Hey,", " ", " ", " ", " ", "friend,", " ", " ", " ", " ", "what's", " ", "up?", " ", " "]
        )

        let preTokenizer2 = SplitPreTokenizer(config: Config(["pattern": ["Regex": "\\s"]]))
        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!") ==
            ["Hey", " ", "friend!"]
        )
        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!     How are you?!?") ==
            ["Hey", " ", "friend!", " ", " ", " ", " ", " ", "How", " ", "are", " ", "you?!?"]
        )
        #expect(
            preTokenizer2.preTokenize(text: "   Hey,    friend,    what's up?  ") ==
            [" ", " ", " ", "Hey,", " ", " ", " ", " ", "friend,", " ", " ", " ", " ", "what's", " ", "up?", " ", " "]
        )

        let preTokenizer3 = SplitPreTokenizer(config: Config(["pattern": ["Regex": "(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"], "invert": true]))
        #expect(
            preTokenizer3.preTokenize(text: "Hello") ==
            ["Hello"]
        )

        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!") ==
            ["Hey", " friend", "!"]
        )
        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!     How are you?!?") ==
            ["Hey", " friend", "!", "    ", " How", " are", " you", "?!?"]
        )
    }

    /// https://github.com/huggingface/tokenizers/pull/1357
    @Test
    func testMetaspacePreTokenizer() {
        // Prepend "always"
        let preTokenizer = MetaspacePreTokenizer(config: Config([
            "add_prefix_space": true,
            "replacement": "▁",
            "prepend_scheme": "always",
        ]))

        // TODO: different sections on <s>
        let text = "Hey my friend <s>how▁are you"
        let tokens = text
            .split(by: "<s>", includeSeparators: true)
            .flatMap { preTokenizer.preTokenize(text: $0) }

        #expect(
            tokens ==
            ["▁Hey", "▁my", "▁friend", "▁", "▁<s>", "▁how", "▁are", "▁you"]
        )
    }

    @Test
    func testBertPreTokenizer() {
        let preTokenizer1 = BertPreTokenizer(config: Config([String: Config]()))
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!") ==
            ["Hey", "friend", "!"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?") ==
            ["Hey", "friend", "!", "How", "are", "you", "?", "!", "?"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend ,    what's up?  ") ==
            ["Hey", ",", "friend", ",", "what", "\'", "s", "up", "?"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend ,  0 99  what's up?  ") ==
            ["Hey", ",", "friend", ",", "0", "99", "what", "\'", "s", "up", "?"]
        )
    }
}
