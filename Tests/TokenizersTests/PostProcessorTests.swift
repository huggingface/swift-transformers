import Foundation
import Testing

@testable import Hub
@testable import Tokenizers

@Suite("Post-processor functionality tests")
struct PostProcessorTests {
    @Suite("RoBERTa post-processing behavior")
    struct RoBERTaProcessingTests {
        @Test("Should keep spaces; uneven spaces; ignore addPrefixSpace")
        func keepsSpacesUnevenIgnoresAddPrefixSpace() {
            let config = Config([
                "cls": ["[HEAD]", 0 as UInt],
                "sep": ["[END]", 0 as UInt],
                "trimOffset": false,
                "addPrefixSpace": true,
            ])
            let tokens = [" The", " sun", "sets ", "  in  ", "   the  ", "west"]
            let expect = ["[HEAD]", " The", " sun", "sets ", "  in  ", "   the  ", "west", "[END]"]
            let processor = RobertaProcessing(config: config)
            let output = processor.postProcess(tokens: tokens, tokensPair: nil)
            #expect(output == expect)
        }

        @Test("Should leave only one space around each token")
        func normalizesSpacesAroundTokens() {
            let config = Config([
                "cls": ["[START]", 0 as UInt],
                "sep": ["[BREAK]", 0 as UInt],
                "trimOffset": true,
                "addPrefixSpace": true,
            ])
            let tokens = [" The ", " sun", "sets ", "  in ", "  the    ", "west"]
            let expect = ["[START]", " The ", " sun", "sets ", " in ", " the ", "west", "[BREAK]"]
            let processor = RobertaProcessing(config: config)
            let output = processor.postProcess(tokens: tokens, tokensPair: nil)
            #expect(output == expect)
        }

        @Test("Should ignore empty tokens pair")
        func ignoresEmptyTokensPair() {
            let config = Config([
                "cls": ["[START]", 0 as UInt],
                "sep": ["[BREAK]", 0 as UInt],
                "trimOffset": true,
                "addPrefixSpace": true,
            ])
            let tokens = [" The ", " sun", "sets ", "  in ", "  the    ", "west"]
            let tokensPair: [String] = []
            let expect = ["[START]", " The ", " sun", "sets ", " in ", " the ", "west", "[BREAK]"]
            let processor = RobertaProcessing(config: config)
            let output = processor.postProcess(tokens: tokens, tokensPair: tokensPair)
            #expect(output == expect)
        }

        @Test("Should trim all whitespace")
        func trimsAllWhitespace() {
            let config = Config([
                "cls": ["[CLS]", 0 as UInt],
                "sep": ["[SEP]", 0 as UInt],
                "trimOffset": true,
                "addPrefixSpace": false,
            ])
            let tokens = [" The ", " sun", "sets ", "  in ", "  the    ", "west"]
            let expect = ["[CLS]", "The", "sun", "sets", "in", "the", "west", "[SEP]"]
            let processor = RobertaProcessing(config: config)
            let output = processor.postProcess(tokens: tokens, tokensPair: nil)
            #expect(output == expect)
        }

        @Test("Should add tokens")
        func addsTokensEnglish() {
            let config = Config([
                "cls": ["[CLS]", 0 as UInt],
                "sep": ["[SEP]", 0 as UInt],
                "trimOffset": true,
                "addPrefixSpace": true,
            ])
            let tokens = [" The ", " sun", "sets ", "  in ", "  the    ", "west"]
            let tokensPair = [".", "The", " cat ", "   is ", " sitting  ", " on", "the ", "mat"]
            let expect = [
                "[CLS]", " The ", " sun", "sets ", " in ", " the ", "west", "[SEP]",
                "[SEP]", ".", "The", " cat ", " is ", " sitting ", " on", "the ",
                "mat", "[SEP]",
            ]
            let processor = RobertaProcessing(config: config)
            let output = processor.postProcess(tokens: tokens, tokensPair: tokensPair)
            #expect(output == expect)
        }

        @Test("Should add tokens (CJK)")
        func addsTokensCJK() {
            let config = Config([
                "cls": ["[CLS]", 0 as UInt],
                "sep": ["[SEP]", 0 as UInt],
                "trimOffset": true,
                "addPrefixSpace": true,
            ])
            let tokens = [" 你 ", " 好 ", ","]
            let tokensPair = [" 凯  ", "  蒂  ", "!"]
            let expect = ["[CLS]", " 你 ", " 好 ", ",", "[SEP]", "[SEP]", " 凯 ", " 蒂 ", "!", "[SEP]"]
            let processor = RobertaProcessing(config: config)
            let output = processor.postProcess(tokens: tokens, tokensPair: tokensPair)
            #expect(output == expect)
        }
    }
}
