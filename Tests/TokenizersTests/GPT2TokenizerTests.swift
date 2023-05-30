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
    lazy var tokenizer: GPT2Tokenizer = {
        let vocab = {
            let url = Bundle.module.url(forResource: "gpt2-vocab", withExtension: "json")!
            let json = try! Data(contentsOf: url)
            let decoder = JSONDecoder()
            let vocab = try! decoder.decode([String: Int].self, from: json)
            return vocab
        }()

        let merges = {
            let url = Bundle.module.url(forResource: "gpt2-merges", withExtension: "txt")!
            let bpeMergesTxt = try! String(contentsOf: url)
            let arr = bpeMergesTxt.split(separator: "\n").map { String($0) }
            return Array(arr[1...])
        }()
        
        return GPT2Tokenizer(vocab: vocab, merges: merges)
    }()
    
    func testByteEncode() {
        let dataset = EncodingSample.dataset
        
        XCTAssertEqual(
            tokenizer.byteEncode(text: dataset.text),
            dataset.encoded_text
        )
    }
    
    func testTokenize() {
        let dataset = EncodingSample.dataset
        
        XCTAssertEqual(
            tokenizer.tokenize(text: dataset.text),
            dataset.bpe_tokens
        )
    }
    
    func testEncode() {
        let dataset = EncodingSample.dataset
        
        XCTAssertEqual(
            tokenizer.encode(text: dataset.text),
            dataset.token_ids
        )
    }
    
    func testDecode() {
        let dataset = EncodingSample.dataset
        
        print(
            tokenizer.decode(tokens: dataset.token_ids)
        )
        XCTAssertEqual(
            tokenizer.decode(tokens: dataset.token_ids),
            dataset.text
        )
    }
}
