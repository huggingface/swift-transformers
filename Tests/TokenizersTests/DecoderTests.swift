//
//  DecoderTests.swift
//
//  Created by Pedro Cuenca on 20231123.
//

import Foundation
import Hub
import Testing

@testable import Tokenizers

@Suite("Tokenizer Decoder Tests")
struct DecoderTests {
    /// https://github.com/huggingface/tokenizers/pull/1357
    @Test("Metaspace decoder with prefix space replacement")
    func metaspaceDecoder() {
        let decoder = MetaspaceDecoder(
            config: Config([
                "add_prefix_space": true,
                "replacement": "▁",
            ]))

        let tokens = ["▁Hey", "▁my", "▁friend", "▁", "▁<s>", "▁how", "▁are", "▁you"]
        let decoded = decoder.decode(tokens: tokens)

        #expect(
            decoded == ["Hey", " my", " friend", " ", " <s>", " how", " are", " you"]
        )
    }

    /// Regression coverage for #329: newer tokenizer.json files written by
    /// transformers ≥ 5 (e.g. `T5Tokenizer.save_pretrained`) drop the legacy
    /// `add_prefix_space` field and only set `prepend_scheme`. The decoder
    /// must derive the strip-leading-space behavior from `prepend_scheme`.
    @Test("Metaspace decoder honors prepend_scheme = always (T5 fine-tune)")
    func metaspaceDecoderPrependSchemeAlways() {
        let decoder = MetaspaceDecoder(
            config: Config([
                "prepend_scheme": "always",
                "replacement": "▁",
                "split": true,
            ]))

        let tokens = ["▁How", "▁are", "▁you", "?"]
        let decoded = decoder.decode(tokens: tokens)

        #expect(decoded == ["How", " are", " you", "?"])
        #expect(decoded.joined() == "How are you?")
    }

    @Test("Metaspace decoder honors prepend_scheme = never")
    func metaspaceDecoderPrependSchemeNever() {
        let decoder = MetaspaceDecoder(
            config: Config([
                "prepend_scheme": "never",
                "replacement": "▁",
            ]))

        // With "never", no leading space was prepended at encode time, so the
        // decoder must leave the leading replacement alone.
        let tokens = ["▁How", "▁are", "▁you", "?"]
        let decoded = decoder.decode(tokens: tokens)

        #expect(decoded == [" How", " are", " you", "?"])
    }

    @Test("Metaspace decoder honors prepend_scheme = first")
    func metaspaceDecoderPrependSchemeFirst() {
        let decoder = MetaspaceDecoder(
            config: Config([
                "prepend_scheme": "first",
                "replacement": "▁",
            ]))

        let tokens = ["▁How", "▁are", "▁you", "?"]
        let decoded = decoder.decode(tokens: tokens)

        // "first" prepended a leading space to the first piece only, so the
        // decoder strips it from the first token.
        #expect(decoded == ["How", " are", " you", "?"])
    }

    @Test("Metaspace decoder prepend_scheme supersedes add_prefix_space")
    func metaspaceDecoderPrependSchemeSupersedesAddPrefixSpace() {
        // When both keys are present, `prepend_scheme` wins per tokenizers PR #1357.
        let decoder = MetaspaceDecoder(
            config: Config([
                "prepend_scheme": "never",
                "add_prefix_space": true,
                "replacement": "▁",
            ]))

        let tokens = ["▁How", "▁are"]
        let decoded = decoder.decode(tokens: tokens)

        #expect(decoded == [" How", " are"])
    }

    @Test("WordPiece decoder with prefix and cleanup")
    func wordPieceDecoder() {
        let config = Config(["prefix": "##", "cleanup": true])
        let decoder = WordPieceDecoder(config: config)

        let testCases: [([String], String)] = [
            (["##inter", "##national", "##ization"], "##internationalization"),
            (["##auto", "##mat", "##ic", "transmission"], "##automatic transmission"),
            (["who", "do", "##n't", "does", "n't", "can't"], "who don't doesn't can't"),
            (["##un", "##believ", "##able", "##fa", "##ntastic"], "##unbelievablefantastic"),
            (
                ["this", "is", "un", "##believ", "##able", "fa", "##ntastic"],
                "this is unbelievable fantastic"
            ),
            (["The", "##quick", "##brown", "fox"], "Thequickbrown fox"),
        ]

        for (tokens, expected) in testCases {
            let output = decoder.decode(tokens: tokens)
            #expect(output.joined() == expected)
        }
    }
}
