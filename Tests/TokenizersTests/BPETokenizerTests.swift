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
@testable import Models

class GPT2TokenizerTests: BPETokenizerTests {
    override class var hubModelName: String? { "distilgpt2" }
    override class var encodedSamplesFilename: String? { "gpt2_encoded_tokens" }
}

class FalconTokenizerTests: BPETokenizerTests {
    override class var hubModelName: String? { "tiiuae/falcon-7b-instruct" }
    override class var encodedSamplesFilename: String? { "falcon_encoded" }
}

class LlamaTokenizerTests: BPETokenizerTests {
    // meta-llama/Llama-2-7b-chat requires approval, and hf-internal-testing/llama-tokenizer does not have a config.json
    override class var hubModelName: String? { "pcuenq/Llama-2-7b-chat-coreml" }
    override class var encodedSamplesFilename: String? { "llama_encoded" }
}

struct BPEEncodingSampleDataset: Decodable {
    let text: String
    let bpe_tokens: [String]
    let token_ids: [Int]
    let decoded_text: String
}

class BPETokenizerTester {
    let encodedSamplesFilename: String
    
    private var configuration: LanguageModelConfigurationFromHub? = nil
    private var _tokenizer: Tokenizer? = nil
    
    init(hubModelName: String, encodedSamplesFilename: String) {
        configuration = LanguageModelConfigurationFromHub(modelName: hubModelName)
        self.encodedSamplesFilename = encodedSamplesFilename
    }
    
    lazy var dataset: BPEEncodingSampleDataset = {
        let url = Bundle.module.url(forResource: encodedSamplesFilename, withExtension: "json")!
        let json = try! Data(contentsOf: url)
        let decoder = JSONDecoder()
        let dataset = try! decoder.decode(BPEEncodingSampleDataset.self, from: json)
        return dataset
    }()
    
    
    var tokenizer: Tokenizer {
        get async {
            guard _tokenizer == nil else { return _tokenizer! }
            do {
                guard let tokenizerConfig = try await configuration!.tokenizerConfig else { throw "Cannot retrieve Tokenizer configuration" }
                let tokenizerData = try await configuration!.tokenizerData
                _tokenizer = try TokenizerFactory.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
            } catch {
                XCTFail("Cannot load tokenizer: \(error)")
            }
            return _tokenizer!
        }
    }
        
    func testTokenize() async {
        let tokenized = await tokenizer.tokenize(text: dataset.text)
        XCTAssertEqual(
            tokenized,
            dataset.bpe_tokens
        )
    }
    
    func testEncode() async {
        let encoded = await tokenizer.encode(text: dataset.text)
        XCTAssertEqual(
            encoded,
            dataset.token_ids
        )
    }
    
    func testDecode() async {
        let decoded = await tokenizer.decode(tokens: dataset.token_ids)
        XCTAssertEqual(
            decoded,
            dataset.decoded_text
        )
    }
}

class BPETokenizerTests: XCTestCase {
    // Parallel testing in Xcode (when enabled) uses different processes, so this shouldn't be a problem
    static var _bpeTester: BPETokenizerTester? = nil
    
    class var hubModelName: String? { nil }
    class var encodedSamplesFilename: String? { nil }
    
    override class func setUp() {
        if let hubModelName = hubModelName, let encodedSamplesFilename = encodedSamplesFilename {
            _bpeTester = BPETokenizerTester(
                hubModelName: hubModelName,
                encodedSamplesFilename: encodedSamplesFilename
            )
        }
    }
        
    func testTokenize() async {
        if let bpeTester = Self._bpeTester {
            await bpeTester.testTokenize()
        }
    }
    
    func testEncode() async {
        if let bpeTester = Self._bpeTester {
            await bpeTester.testEncode()
        }
    }
    
    func testDecode() async {
        if let bpeTester = Self._bpeTester {
            await bpeTester.testDecode()
        }
    }
}
