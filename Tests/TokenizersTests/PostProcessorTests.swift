@testable import Hub
@testable import Tokenizers
import XCTest

class PostProcessorTests: XCTestCase {
    func testRobertaProcessing() {
        let testCases: [(Config, [String], [String]?, [String])] = [
            // Should keep spaces; uneven spaces; ignore `addPrefixSpace`.
            (
                Config(["cls": .token(0, "[HEAD]"),
                        "sep": .token(0, "[END]"),
                        "trimOffset": .boolean(false),
                        "addPrefixSpace": .boolean(true)]),
                [" The", " sun", "sets ", "  in  ", "   the  ", "west"],
                nil,
                ["[HEAD]", " The", " sun", "sets ", "  in  ", "   the  ", "west", "[END]"]
            ),
            // Should leave only one space around each token.
            (
                Config(["cls": .token(0, "[START]"),
                        "sep": .token(0, "[BREAK]"),
                        "trimOffset": .boolean(true),
                        "addPrefixSpace": .boolean(true)]),
                [" The ", " sun", "sets ", "  in ", "  the    ", "west"],
                nil,
                ["[START]", " The ", " sun", "sets ", " in ", " the ", "west", "[BREAK]"]
            ),
            // Should ignore empty tokens pair.
            (
                Config(["cls": .token(0, "[START]"),
                        "sep": .token(0, "[BREAK]"),
                        "trimOffset": .boolean(true),
                        "addPrefixSpace": .boolean(true)]),
                [" The ", " sun", "sets ", "  in ", "  the    ", "west"],
                [],
                ["[START]", " The ", " sun", "sets ", " in ", " the ", "west", "[BREAK]"]
            ),
            // Should trim all whitespace.
            (
                Config(["cls": .token(0, "[CLS]"),
                        "sep": .token(0, "[SEP]"),
                        "trimOffset": .boolean(true),
                        "addPrefixSpace": .boolean(false)]),
                [" The ", " sun", "sets ", "  in ", "  the    ", "west"],
                nil,
                ["[CLS]", "The", "sun", "sets", "in", "the", "west", "[SEP]"]
            ),
            // Should add tokens.
            (
                Config(["cls": .token(0, "[CLS]"),
                        "sep": .token(0, "[SEP]"),
                        "trimOffset": .boolean(true),
                        "addPrefixSpace": .boolean(true)]),
                [" The ", " sun", "sets ", "  in ", "  the    ", "west"],
                [".", "The", " cat ", "   is ", " sitting  ", " on", "the ", "mat"],
                ["[CLS]", " The ", " sun", "sets ", " in ", " the ", "west", "[SEP]",
                 "[SEP]", ".", "The", " cat ", " is ", " sitting ", " on", "the ",
                 "mat", "[SEP]"]
            ),
            (
                Config(["cls": .token(0, "[CLS]"),
                        "sep": .token(0, "[SEP]"),
                        "trimOffset": .boolean(true),
                        "addPrefixSpace": .boolean(true)]),
                [" 你 ", " 好 ", ","],
                [" 凯  ", "  蒂  ", "!"],
                ["[CLS]", " 你 ", " 好 ", ",", "[SEP]", "[SEP]", " 凯 ", " 蒂 ", "!", "[SEP]"]
            ),
        ]

        for (config, tokens, tokensPair, expect) in testCases {
            let processor = RobertaProcessing(config: config)
            let output = processor.postProcess(tokens: tokens, tokensPair: tokensPair)
            XCTAssertEqual(output, expect)
        }
    }
}
