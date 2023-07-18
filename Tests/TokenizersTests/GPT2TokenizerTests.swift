//
//  GPT2TokenizerTests.swift
//  CoreMLGPT2Tests
//
//  Created by Julien Chaumond on 18/07/2019.
//  Adapted by Pedro Cuenca on July 2023.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import XCTest

class GPT2TokenizerTests: XCTestCase {
    lazy var bpeTests = BPETokenizerTests(
        // From https://huggingface.co/Xenova/distilgpt2, as `distilgpt2` doesn't have a tokenizer.json
        tokenizerConfigFilename: "gpt2_tokenizer_config",
        tokenizerDataFilename: "gpt2_tokenizer",
        encodedSamplesFilename: "gpt2_encoded_tokens"
    )
        
    func testTokenize() {
        bpeTests.testTokenize()
    }
    
    func testEncode() {
        bpeTests.testEncode()
    }
    
    func testDecode() {
        bpeTests.testDecode()
    }
}
