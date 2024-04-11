//
//  DecoderTests.swift
//
//  Created by Pedro Cuenca on 20231123.
//

import Hub
import XCTest

@testable import Tokenizers

class DecoderTests: XCTestCase {
    // https://github.com/huggingface/tokenizers/pull/1357
    func testMetaspaceDecoder() {
        let decoder = MetaspaceDecoder(
            config: Config([
                "add_prefix_space": true,
                "replacement": "▁",
            ])
        )

        let tokens = ["▁Hey", "▁my", "▁friend", "▁", "▁<s>", "▁how", "▁are", "▁you"]
        let decoded = decoder.decode(tokens: tokens)

        XCTAssertEqual(
            decoded,
            ["Hey", " my", " friend", " ", " <s>", " how", " are", " you"]
        )
    }
}
