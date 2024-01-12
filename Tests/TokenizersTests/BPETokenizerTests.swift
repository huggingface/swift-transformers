//
//  BPETokenizerTests.swift
//
//  Created by Pedro Cuenca on July 2023.
//  Based on GPT2TokenizerTests by Julien Chaumond.
//  Copyright © 2023 Hugging Face. All rights reserved.
//

import XCTest
import Hub
@testable import Tokenizers
@testable import Models

class GPT2TokenizerTests: BPETokenizerTests {
    override class var hubModelName: String? { "distilgpt2" }
    override class var encodedSamplesFilename: String? { "gpt2_encoded_tokens" }
    override class var unknownTokenId: Int? { 50256 }
}

class FalconTokenizerTests: BPETokenizerTests {
    override class var hubModelName: String? { "tiiuae/falcon-7b" }
    override class var encodedSamplesFilename: String? { "falcon_encoded" }
    override class var unknownTokenId: Int? { nil }
}

class LlamaTokenizerTests: BPETokenizerTests {
    // meta-llama/Llama-2-7b-chat requires approval, and hf-internal-testing/llama-tokenizer does not have a config.json
    override class var hubModelName: String? { "coreml-projects/Llama-2-7b-chat-coreml" }
    override class var encodedSamplesFilename: String? { "llama_encoded" }
    override class var unknownTokenId: Int? { 0 }
}

class WhisperLargeTokenizerTests: BPETokenizerTests {
    override class var hubModelName: String? { "openai/whisper-large-v2" }
    override class var encodedSamplesFilename: String? { "whisper_large_v2_encoded" }
    override class var unknownTokenId: Int? { 50257 }
}

class WhisperTinyTokenizerTests: BPETokenizerTests {
    override class var hubModelName: String? { "openai/whisper-tiny.en" }
    override class var encodedSamplesFilename: String? { "whisper_tiny_en_encoded" }
    override class var unknownTokenId: Int? { 50256 }
}

struct BPEEncodingSampleDataset: Decodable {
    let text: String
    let bpe_tokens: [String]
    let token_ids: [Int]
    let decoded_text: String
}


typealias EdgeCasesDataset = [String : [EdgeCase]]

struct EdgeCase: Decodable {
    let input: String
    let encoded: EncodedData
    let decoded_with_special: String
    let decoded_without_special: String
}

struct EncodedData: Decodable {
    let input_ids: [Int]
    let token_type_ids: [Int]?
    let attention_mask: [Int]
}


class BPETokenizerTester {
    let encodedSamplesFilename: String
    let unknownTokenId: Int?
    
    private var configuration: LanguageModelConfigurationFromHub? = nil
    private var edgeCases: [EdgeCase]? = nil
    private var _tokenizer: Tokenizer? = nil
    
    init(hubModelName: String, encodedSamplesFilename: String, unknownTokenId: Int?) {
        configuration = LanguageModelConfigurationFromHub(modelName: hubModelName)
        self.encodedSamplesFilename = encodedSamplesFilename
        self.unknownTokenId = unknownTokenId
        
        // Read the edge cases dataset
        edgeCases = {
            let url = Bundle.module.url(forResource: "tokenizer_tests", withExtension: "json")!
            let json = try! Data(contentsOf: url)
            let decoder = JSONDecoder()
            let cases = try! decoder.decode(EdgeCasesDataset.self, from: json)
            // Return the ones for this model
            return cases[hubModelName]
        }()
    }
    
    lazy var dataset: BPEEncodingSampleDataset = {
        let url = Bundle.module.url(forResource: encodedSamplesFilename, withExtension: "json")!
        let json = try! Data(contentsOf: url)
        let decoder = JSONDecoder()
        let dataset = try! decoder.decode(BPEEncodingSampleDataset.self, from: json)
        return dataset
    }()
    
    
    var tokenizer: Tokenizer? {
        get async {
            guard _tokenizer == nil else { return _tokenizer! }
            do {
                guard let tokenizerConfig = try await configuration!.tokenizerConfig else { throw "Cannot retrieve Tokenizer configuration" }
                let tokenizerData = try await configuration!.tokenizerData
                _tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
            } catch {
                XCTFail("Cannot load tokenizer: \(error)")
            }
            return _tokenizer
        }
    }
        
    func testTokenize() async {
        let tokenized = await tokenizer?.tokenize(text: dataset.text)
        XCTAssertEqual(
            tokenized,
            dataset.bpe_tokens
        )
    }
    
    func testEncode() async {
        let encoded = await tokenizer?.encode(text: dataset.text)
        XCTAssertEqual(
            encoded,
            dataset.token_ids
        )
    }
    
    func testDecode() async {
        let decoded = await tokenizer?.decode(tokens: dataset.token_ids)
        XCTAssertEqual(
            decoded,
            dataset.decoded_text
        )
    }
    
    /// Test encode and decode for a few edge cases
    func testEdgeCases() async {
        guard let edgeCases = edgeCases else { return }
        guard let tokenizer = await tokenizer else { return }
        for edgeCase in edgeCases {
            print("Testing \(edgeCase.input)")
            XCTAssertEqual(
                tokenizer.encode(text: edgeCase.input),
                edgeCase.encoded.input_ids
            )
            XCTAssertEqual(
                tokenizer.decode(tokens: edgeCase.encoded.input_ids),
                edgeCase.decoded_with_special
            )
        }
    }
    
    func testUnknownToken() async {
        guard let tokenizer = await tokenizer else { return }
        XCTAssertEqual(tokenizer.unknownTokenId, unknownTokenId)
        XCTAssertEqual(
            tokenizer.unknownTokenId,
            tokenizer.convertTokenToId("_this_token_does_not_exist_")
        )
        if let unknownTokenId = tokenizer.unknownTokenId {
            XCTAssertEqual(
                tokenizer.unknownToken,
                tokenizer.convertIdToToken(unknownTokenId)
            )
        } else {
            XCTAssertNil(tokenizer.unknownTokenId)
        }
    }
}

class BPETokenizerTests: XCTestCase {
    // Parallel testing in Xcode (when enabled) uses different processes, so this shouldn't be a problem
    static var _bpeTester: BPETokenizerTester? = nil
    
    class var hubModelName: String? { nil }
    class var encodedSamplesFilename: String? { nil }
    
    // Known id retrieved from Python, to verify it was parsed correctly
    class var unknownTokenId: Int? { nil }
    
    override class func setUp() {
        if let hubModelName = hubModelName, let encodedSamplesFilename = encodedSamplesFilename {
            _bpeTester = BPETokenizerTester(
                hubModelName: hubModelName,
                encodedSamplesFilename: encodedSamplesFilename,
                unknownTokenId: unknownTokenId
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
    
    func testEdgeCases() async {
        if let bpeTester = Self._bpeTester {
            await bpeTester.testEdgeCases()
        }
    }
    
    func testUnknownToken() async {
        if let bpeTester = Self._bpeTester {
            await bpeTester.testUnknownToken()
        }
    }
}
