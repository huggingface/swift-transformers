//
//  BPETokenizerTests.swift
//
//  Created by Pedro Cuenca on July 2023.
//  Based on GPT2TokenizerTests by Julien Chaumond.
//  Copyright © 2023 Hugging Face. All rights reserved.
//

import XCTest
@testable import Hub
@testable import Tokenizers

struct BPEEncodingSampleDataset: Decodable {
    let text: String
    let bpe_tokens: [String]
    let token_ids: [Int]
}

class BPETokenizerTests {
    // Resources in test bundle.
    // TODO: download from the Hub.
    let tokenizerConfigFilename: String
    let tokenizerDataFilename: String
    let encodedSamplesFilename: String
    
    init(tokenizerConfigFilename: String, tokenizerDataFilename: String, encodedSamplesFilename: String) {
        self.tokenizerConfigFilename = tokenizerConfigFilename
        self.tokenizerDataFilename = tokenizerDataFilename
        self.encodedSamplesFilename = encodedSamplesFilename
    }
    
    lazy var dataset: BPEEncodingSampleDataset = {
        let url = Bundle.module.url(forResource: encodedSamplesFilename, withExtension: "json")!
        let json = try! Data(contentsOf: url)
        let decoder = JSONDecoder()
        let dataset = try! decoder.decode(BPEEncodingSampleDataset.self, from: json)
        return dataset
    }()
    
    lazy var tokenizer: Tokenizer = {
        let tokenizerConfig = {
            let url = Bundle.module.url(forResource: tokenizerConfigFilename, withExtension: "json")!
            let data = try! Data(contentsOf: url)
            let parsed = try! JSONSerialization.jsonObject(with: data, options: [])
            let dictionary = parsed as! [String: Any]
            return Config(dictionary)
        }()
        
        let tokenizerData = {
            let url = Bundle.module.url(forResource: tokenizerDataFilename, withExtension: "json")!
            let data = try! Data(contentsOf: url)
            let parsed = try! JSONSerialization.jsonObject(with: data, options: [])
            let dictionary = parsed as! [String: Any]
            return Config(dictionary)
        }()
        
        return try! TokenizerFactory.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }()
        
    func testTokenize() {
        XCTAssertEqual(
            tokenizer.tokenize(text: dataset.text),
            dataset.bpe_tokens
        )
    }
    
    func testEncode() {
        let ids = tokenizer.encode(text: dataset.text)
        
        XCTAssertEqual(
            tokenizer.encode(text: dataset.text),
            dataset.token_ids
        )
    }
    
    func testDecode() {
        print(
            tokenizer.decode(tokens: dataset.token_ids)
        )
        XCTAssertEqual(
            tokenizer.decode(tokens: dataset.token_ids),
            dataset.text
        )
    }
}
