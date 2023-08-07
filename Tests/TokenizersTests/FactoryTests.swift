//
//  FactoryTests.swift
//  
//
//  Created by Pedro Cuenca on 4/8/23.
//

import XCTest
import Tokenizers

class FactoryTests: XCTestCase {
    func testFromPretrained() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "pcuenq/Llama-2-7b-chat-coreml")
        let inputIds = tokenizer("Today she took a train to the West")
        XCTAssertEqual(inputIds, [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122])
    }
}
