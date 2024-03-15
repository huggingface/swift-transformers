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
        let tokenizer = try await AutoTokenizer.from(pretrained: "coreml-projects/Llama-2-7b-chat-coreml")
        let inputIds = tokenizer("Today she took a train to the West")
        XCTAssertEqual(inputIds, [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122])
    }
    
    func testWhisper() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "openai/whisper-large-v2")
        let inputIds = tokenizer("Today she took a train to the West")
        XCTAssertEqual(inputIds, [50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257])
    }
}
