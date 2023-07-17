//
//  LlamaTokenizerTests.swift
//
//  Created by Pedro Cuenca on July 2023.
//  Based on GPT2TokenizerTests by Julien Chaumond.
//  Copyright © 2023 Hugging Face. All rights reserved.
//

import XCTest

class LlamaTokenizerTests: XCTestCase {
    lazy var bpeTests = BPETokenizerTests(
        // From `hf-internal-testing/llama-tokenizer`
        tokenizerConfigFilename: "llama_tokenizer_config",
        tokenizerDataFilename: "llama_tokenizer",
        encodedSamplesFilename: "llama_encoded"
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
