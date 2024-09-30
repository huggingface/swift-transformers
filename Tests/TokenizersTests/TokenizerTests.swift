//
//  TokenizerTests.swift
//
//  Created by Pedro Cuenca on July 2023.
//  Based on GPT2TokenizerTests by Julien Chaumond.
//  Copyright © 2023 Hugging Face. All rights reserved.
//

import XCTest
import Hub
@testable import Tokenizers
@testable import Models

class GPT2TokenizerTests: TokenizerTests {
    override class var hubModelName: String? { "distilgpt2" }
    override class var encodedSamplesFilename: String? { "gpt2_encoded_tokens" }
    override class var unknownTokenId: Int? { 50256 }
}

class FalconTokenizerTests: TokenizerTests {
    override class var hubModelName: String? { "tiiuae/falcon-7b" }
    override class var encodedSamplesFilename: String? { "falcon_encoded" }
    override class var unknownTokenId: Int? { nil }
}

class LlamaTokenizerTests: TokenizerTests {
    // meta-llama/Llama-2-7b-chat requires approval, and hf-internal-testing/llama-tokenizer does not have a config.json
    override class var hubModelName: String? { "coreml-projects/Llama-2-7b-chat-coreml" }
    override class var encodedSamplesFilename: String? { "llama_encoded" }
    override class var unknownTokenId: Int? { 0 }

    func testHexaEncode() async {
        if let tester = Self._tester {
            let tokenized = await tester.tokenizer?.tokenize(text: "\n")
            XCTAssertEqual(tokenized, ["▁", "<0x0A>"])
        }
    }
}

class Llama32TokenizerTests: TokenizerTests {
    override class var hubModelName: String? { "pcuenq/Llama-3.2-1B-Instruct-tokenizer" }
    override class var encodedSamplesFilename: String? { "llama_3.2_encoded" }
}

class WhisperLargeTokenizerTests: TokenizerTests {
    override class var hubModelName: String? { "openai/whisper-large-v2" }
    override class var encodedSamplesFilename: String? { "whisper_large_v2_encoded" }
    override class var unknownTokenId: Int? { 50257 }
}

class WhisperTinyTokenizerTests: TokenizerTests {
    override class var hubModelName: String? { "openai/whisper-tiny.en" }
    override class var encodedSamplesFilename: String? { "whisper_tiny_en_encoded" }
    override class var unknownTokenId: Int? { 50256 }
}

class T5TokenizerTests: TokenizerTests {
    override class var hubModelName: String? { "t5-base" }
    override class var encodedSamplesFilename: String? { "t5_base_encoded" }
    override class var unknownTokenId: Int? { 2 }
}

class GemmaTokenizerTests: TokenizerTests {
    override class var hubModelName: String? { "pcuenq/gemma-tokenizer" }
    override class var encodedSamplesFilename: String? { "gemma_encoded" }
    override class var unknownTokenId: Int? { 3 }

    func testUnicodeEdgeCase() async {
        guard let tester = Self._tester else {
            XCTFail()
            return
        }

        // These are two different characters
        let cases = ["à" /* 0x61 0x300 */, "à" /* 0xe0 */]
        let expected = [217138, 1305]

        // These are different characters
        for (s, expected) in zip(cases, expected) {
            let encoded = await tester.tokenizer?.encode(text: " " + s)
            XCTAssertEqual(encoded, [2, expected])
        }
    }
}

class GemmaUnicodeTests: XCTestCase {
    func testGemmaVocab() async throws {
        guard let tokenizer = try await AutoTokenizer.from(pretrained: "pcuenq/gemma-tokenizer") as? PreTrainedTokenizer else {
            XCTFail()
            return
        }

        // FIXME: This should be 256_000, I believe
        XCTAssertEqual((tokenizer.model as? BPETokenizer)?.vocabCount, 255994)
    }
}


struct EncodedTokenizerSamplesDataset: Decodable {
    let text: String
    // Bad naming, not just for bpe.
    // We are going to replace this testing method anyway.
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


class TokenizerTester {
    let encodedSamplesFilename: String
    let unknownTokenId: Int?
    
    private var configuration: LanguageModelConfigurationFromHub? = nil
    private var edgeCases: [EdgeCase]? = nil
    private var _tokenizer: Tokenizer? = nil
    
    init(hubModelName: String, encodedSamplesFilename: String, unknownTokenId: Int?, hubApi: HubApi) {
        configuration = LanguageModelConfigurationFromHub(modelName: hubModelName, hubApi: hubApi)
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
    
    lazy var dataset: EncodedTokenizerSamplesDataset = {
        let url = Bundle.module.url(forResource: encodedSamplesFilename, withExtension: "json")!
        let json = try! Data(contentsOf: url)
        let decoder = JSONDecoder()
        let dataset = try! decoder.decode(EncodedTokenizerSamplesDataset.self, from: json)
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
    
    var tokenizerModel: TokenizingModel? {
        get async {
            // The model is not usually accessible; maybe it should
            guard let tokenizer = await tokenizer else { return nil }
            return (tokenizer as! PreTrainedTokenizer).model
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
        guard let edgeCases = edgeCases else {
            print("Edge cases test ignored")
            return
        }
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
        guard let model = await tokenizerModel else { return }
        XCTAssertEqual(model.unknownTokenId, unknownTokenId)
        XCTAssertEqual(
            model.unknownTokenId,
            model.convertTokenToId("_this_token_does_not_exist_")
        )
        if let unknownTokenId = model.unknownTokenId {
            XCTAssertEqual(
                model.unknownToken,
                model.convertIdToToken(unknownTokenId)
            )
        } else {
            XCTAssertNil(model.unknownTokenId)
        }
    }
}

class TokenizerTests: XCTestCase {
    // Parallel testing in Xcode (when enabled) uses different processes, so this shouldn't be a problem
    static var _tester: TokenizerTester? = nil
    
    class var hubModelName: String? { nil }
    class var encodedSamplesFilename: String? { nil }
    
    // Known id retrieved from Python, to verify it was parsed correctly
    class var unknownTokenId: Int? { nil }

    static var downloadDestination: URL = {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appending(component: "huggingface-tests")
    }()

    class var hubApi: HubApi { HubApi(downloadBase: downloadDestination) }

    override class func setUp() {
        if let hubModelName = hubModelName, let encodedSamplesFilename = encodedSamplesFilename {
            _tester = TokenizerTester(
                hubModelName: hubModelName,
                encodedSamplesFilename: encodedSamplesFilename,
                unknownTokenId: unknownTokenId,
                hubApi: hubApi
            )
        }
    }

    override class func tearDown() {
        do {
            try FileManager.default.removeItem(at: downloadDestination)
        } catch {
            print("Can't remove test download destination \(downloadDestination), error: \(error)")
        }
    }

    func testTokenize() async {
        if let tester = Self._tester {
            await tester.testTokenize()
        }
    }
    
    func testEncode() async {
        if let tester = Self._tester {
            await tester.testEncode()
        }
    }
    
    func testDecode() async {
        if let tester = Self._tester {
            await tester.testDecode()
        }
    }
    
    func testEdgeCases() async {
        if let tester = Self._tester {
            await tester.testEdgeCases()
        }
    }
    
    func testUnknownToken() async {
        if let tester = Self._tester {
            await tester.testUnknownToken()
        }
    }
}
