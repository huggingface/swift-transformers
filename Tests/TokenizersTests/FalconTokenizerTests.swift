//
//  LlamaTokenizerTests.swift
//
//  Created by Pedro Cuenca on July 2023.
//  Based on GPT2TokenizerTests by Julien Chaumond.
//  Copyright © 2023 Hugging Face. All rights reserved.
//

import XCTest

class FalconTokenizerTests: XCTestCase {
    lazy var bpeTests = BPETokenizerTests(
        tokenizerConfigFilename: "falcon_tokenizer_config",
        tokenizerDataFilename: "falcon_tokenizer",
        encodedSamplesFilename: "falcon_encoded"
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
