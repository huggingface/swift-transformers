//
//  CoreMLGPT2Tests.swift
//  CoreMLGPT2Tests
//
//  Created by Julien Chaumond on 18/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import XCTest
@testable import Tokenizers

struct EncodingSampleDataset: Decodable {
    let text: String
    let encoded_text: [String]
    let bpe_tokens: [String]
    let token_ids: [Int]
}

struct EncodingSample {
    static let dataset: EncodingSampleDataset = {
        let url = Bundle.module.url(forResource: "gpt2_encoded_tokens", withExtension: "json")!
        let json = try! Data(contentsOf: url)
        let decoder = JSONDecoder()
        let dataset = try! decoder.decode(EncodingSampleDataset.self, from: json)
        return dataset
    }()
}



class GPT2TokenizerTests: XCTestCase {
    func testByteEncode() {
        let dataset = EncodingSample.dataset
        
        let tokenizer = GPT2Tokenizer()
        XCTAssertEqual(
            tokenizer.byteEncode(text: dataset.text),
            dataset.encoded_text
        )
    }
    
    func testTokenize() {
        let dataset = EncodingSample.dataset
        
        let tokenizer = GPT2Tokenizer()
        XCTAssertEqual(
            tokenizer.tokenize(text: dataset.text),
            dataset.bpe_tokens
        )
    }
    
    func testEncode() {
        let dataset = EncodingSample.dataset
        
        let tokenizer = GPT2Tokenizer()
        XCTAssertEqual(
            tokenizer.encode(text: dataset.text),
            dataset.token_ids
        )
    }
    
    func testDecode() {
        let dataset = EncodingSample.dataset
        
        let tokenizer = GPT2Tokenizer()
        print(
            tokenizer.decode(tokens: dataset.token_ids)
        )
        XCTAssertEqual(
            tokenizer.decode(tokens: dataset.token_ids),
            dataset.text
        )
    }
}
