//
//  TokenizerTests.swift
//
//  Created by Pedro Cuenca on July 2023.
//  Based on GPT2TokenizerTests by Julien Chaumond.
//  Copyright © 2023 Hugging Face. All rights reserved.
//

import Foundation
import Hub
@testable import Models
import Testing
@testable import Tokenizers

// MARK: - Tokenizer Test Configuration

struct TokenizerConfig {
    let hubModelName: String
    let encodedSamplesFilename: String
    let unknownTokenId: Int?

    static let gpt2 = TokenizerConfig(
        hubModelName: "distilgpt2",
        encodedSamplesFilename: "gpt2_encoded_tokens",
        unknownTokenId: 50256
    )

    static let falcon = TokenizerConfig(
        hubModelName: "tiiuae/falcon-7b",
        encodedSamplesFilename: "falcon_encoded",
        unknownTokenId: nil
    )

    static let llama = TokenizerConfig(
        hubModelName: "coreml-projects/Llama-2-7b-chat-coreml",
        encodedSamplesFilename: "llama_encoded",
        unknownTokenId: 0
    )

    static let llama32 = TokenizerConfig(
        hubModelName: "pcuenq/Llama-3.2-1B-Instruct-tokenizer",
        encodedSamplesFilename: "llama_3.2_encoded",
        unknownTokenId: nil
    )

    static let whisperLarge = TokenizerConfig(
        hubModelName: "openai/whisper-large-v2",
        encodedSamplesFilename: "whisper_large_v2_encoded",
        unknownTokenId: 50257
    )

    static let whisperTiny = TokenizerConfig(
        hubModelName: "openai/whisper-tiny.en",
        encodedSamplesFilename: "whisper_tiny_en_encoded",
        unknownTokenId: 50256
    )

    static let t5 = TokenizerConfig(
        hubModelName: "t5-base",
        encodedSamplesFilename: "t5_base_encoded",
        unknownTokenId: 2
    )

    static let bertCased = TokenizerConfig(
        hubModelName: "distilbert/distilbert-base-multilingual-cased",
        encodedSamplesFilename: "distilbert_cased_encoded",
        unknownTokenId: 100
    )

    static let bertUncased = TokenizerConfig(
        hubModelName: "google-bert/bert-base-uncased",
        encodedSamplesFilename: "bert_uncased_encoded",
        unknownTokenId: 100
    )

    static let gemma = TokenizerConfig(
        hubModelName: "pcuenq/gemma-tokenizer",
        encodedSamplesFilename: "gemma_encoded",
        unknownTokenId: 3
    )
}

// MARK: - Test Data Structures

struct EncodedTokenizerSamplesDataset: Decodable {
    let text: String
    let bpe_tokens: [String]
    let token_ids: [Int]
    let decoded_text: String
}

typealias EdgeCasesDataset = [String: [EdgeCase]]

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

// MARK: - Tokenizer Test Fixture

actor TokenizerTestFixture {
    let config: TokenizerConfig
    private let hubApi: HubApi
    private let configuration: LanguageModelConfigurationFromHub
    private let edgeCases: [EdgeCase]?

    private var _tokenizer: Tokenizer?
    private var _dataset: EncodedTokenizerSamplesDataset?

    init(config: TokenizerConfig) {
        self.config = config

        let downloadDestination: URL = {
            let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            return base.appending(component: "huggingface-tests")
        }()

        hubApi = HubApi(downloadBase: downloadDestination)
        configuration = LanguageModelConfigurationFromHub(
            modelName: config.hubModelName,
            hubApi: hubApi
        )

        // Read the edge cases dataset
        edgeCases = {
            guard let url = Bundle.module.url(forResource: "tokenizer_tests", withExtension: "json") else {
                return nil
            }
            guard let json = try? Data(contentsOf: url) else { return nil }
            let decoder = JSONDecoder()
            guard let cases = try? decoder.decode(EdgeCasesDataset.self, from: json) else { return nil }
            return cases[config.hubModelName]
        }()
    }

    var tokenizer: Tokenizer? {
        get async throws {
            if let _tokenizer {
                return _tokenizer
            }

            guard let tokenizerConfig = try await configuration.tokenizerConfig else {
                throw TokenizerError.tokenizerConfigNotFound
            }
            let tokenizerData = try await configuration.tokenizerData
            _tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
            return _tokenizer
        }
    }

    func getUnknownTokenInfo() async throws -> (unknownTokenId: Int?, unknownToken: String?) {
        guard let tokenizer = try await tokenizer else {
            return (unknownTokenId: nil, unknownToken: nil)
        }
        let model = (tokenizer as! PreTrainedTokenizer).model
        return (unknownTokenId: model.unknownTokenId, unknownToken: model.unknownToken)
    }

    var dataset: EncodedTokenizerSamplesDataset {
        get async throws {
            if let _dataset {
                return _dataset
            }

            guard let url = Bundle.module.url(forResource: config.encodedSamplesFilename, withExtension: "json") else {
                throw TokenizerError.tokenizerConfigNotFound
            }
            let json = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            let dataset = try decoder.decode(EncodedTokenizerSamplesDataset.self, from: json)
            _dataset = dataset
            return dataset
        }
    }

    func getEdgeCases() async -> [EdgeCase]? {
        edgeCases
    }
}

// MARK: - Parameterized Tokenizer Tests

@Suite("Tokenizer Tests", .serialized)
struct TokenizerTests {
    @Test("Tokenize", arguments: [
        TokenizerConfig.gpt2,
        TokenizerConfig.falcon,
        TokenizerConfig.llama,
        TokenizerConfig.llama32,
        TokenizerConfig.whisperLarge,
        TokenizerConfig.whisperTiny,
        TokenizerConfig.t5,
        TokenizerConfig.bertCased,
        TokenizerConfig.bertUncased,
        TokenizerConfig.gemma,
    ])
    func testTokenize(config: TokenizerConfig) async throws {
        let fixture = TokenizerTestFixture(config: config)

        do {
            let tokenizer = try await fixture.tokenizer
            let dataset = try await fixture.dataset

            let tokenized = tokenizer?.tokenize(text: dataset.text)
            #expect(tokenized == dataset.bpe_tokens)
        } catch {
            // Skip test if we can't download the tokenizer (network issues, etc.)
            Issue.record("Could not load tokenizer for \(config.hubModelName): \(error)")
            return
        }
    }

    @Test("Encode", arguments: [
        TokenizerConfig.gpt2,
        TokenizerConfig.falcon,
        TokenizerConfig.llama,
        TokenizerConfig.llama32,
        TokenizerConfig.whisperLarge,
        TokenizerConfig.whisperTiny,
        TokenizerConfig.t5,
        TokenizerConfig.bertCased,
        TokenizerConfig.bertUncased,
        TokenizerConfig.gemma,
    ])
    func testEncode(config: TokenizerConfig) async throws {
        let fixture = TokenizerTestFixture(config: config)

        do {
            let tokenizer = try await fixture.tokenizer
            let dataset = try await fixture.dataset

            let encoded = tokenizer?.encode(text: dataset.text)
            #expect(encoded == dataset.token_ids)
        } catch {
            Issue.record("Could not load tokenizer for \(config.hubModelName): \(error)")
            return
        }
    }

    @Test("Decode", arguments: [
        TokenizerConfig.gpt2,
        TokenizerConfig.falcon,
        TokenizerConfig.llama,
        TokenizerConfig.llama32,
        TokenizerConfig.whisperLarge,
        TokenizerConfig.whisperTiny,
        TokenizerConfig.t5,
        TokenizerConfig.bertCased,
        TokenizerConfig.bertUncased,
        TokenizerConfig.gemma,
    ])
    func testDecode(config: TokenizerConfig) async throws {
        let fixture = TokenizerTestFixture(config: config)

        do {
            let tokenizer = try await fixture.tokenizer
            let dataset = try await fixture.dataset

            let decoded = tokenizer?.decode(tokens: dataset.token_ids)
            #expect(decoded == dataset.decoded_text)
        } catch {
            Issue.record("Could not load tokenizer for \(config.hubModelName): \(error)")
            return
        }
    }

    @Test("Edge Cases", arguments: [
        TokenizerConfig.gpt2,
        TokenizerConfig.falcon,
        TokenizerConfig.llama,
        TokenizerConfig.llama32,
        TokenizerConfig.whisperLarge,
        TokenizerConfig.whisperTiny,
        TokenizerConfig.t5,
        TokenizerConfig.bertCased,
        TokenizerConfig.bertUncased,
        TokenizerConfig.gemma,
    ])
    func testEdgeCases(config: TokenizerConfig) async throws {
        let fixture = TokenizerTestFixture(config: config)

        do {
            guard let edgeCases = await fixture.getEdgeCases() else {
                return // Skip if no edge cases for this tokenizer
            }

            let tokenizer = try await fixture.tokenizer
            guard let tokenizer else { return }

            for edgeCase in edgeCases {
                #expect(
                    tokenizer.encode(text: edgeCase.input) == edgeCase.encoded.input_ids,
                    "Failed encoding edge case: \(edgeCase.input)"
                )
                #expect(
                    tokenizer.decode(tokens: edgeCase.encoded.input_ids) == edgeCase.decoded_with_special,
                    "Failed decoding with special tokens: \(edgeCase.input)"
                )
                #expect(
                    tokenizer.decode(tokens: edgeCase.encoded.input_ids, skipSpecialTokens: true) == edgeCase.decoded_without_special,
                    "Failed decoding without special tokens: \(edgeCase.input)"
                )
            }
        } catch {
            Issue.record("Could not load tokenizer for \(config.hubModelName): \(error)")
            return
        }
    }

    @Test("Unknown Token", arguments: [
        TokenizerConfig.gpt2,
        TokenizerConfig.falcon,
        TokenizerConfig.llama,
        TokenizerConfig.llama32,
        TokenizerConfig.whisperLarge,
        TokenizerConfig.whisperTiny,
        TokenizerConfig.t5,
        TokenizerConfig.bertCased,
        TokenizerConfig.bertUncased,
        TokenizerConfig.gemma,
    ])
    func testUnknownToken(config: TokenizerConfig) async throws {
        let fixture = TokenizerTestFixture(config: config)

        do {
            let tokenInfo = try await fixture.getUnknownTokenInfo()

            guard let tokenizer = try await fixture.tokenizer else { return }
            let model = (tokenizer as! PreTrainedTokenizer).model

            #expect(tokenInfo.unknownTokenId == config.unknownTokenId)
            #expect(
                model.unknownTokenId == model.convertTokenToId("_this_token_does_not_exist_")
            )

            if let unknownTokenId = model.unknownTokenId {
                #expect(
                    model.unknownToken == model.convertIdToToken(unknownTokenId)
                )
            } else {
                #expect(model.unknownTokenId == nil)
            }
        } catch {
            Issue.record("Could not load tokenizer for \(config.hubModelName): \(error)")
            return
        }
    }
}

// MARK: - Specialized Tests

@Suite("Llama Specific Tests", .serialized)
struct LlamaSpecificTests {
    @Test
    func hexaEncode() async throws {
        let fixture = TokenizerTestFixture(config: .llama)
        let tokenizer = try await fixture.tokenizer

        let tokenized = tokenizer?.tokenize(text: "\n")
        #expect(tokenized == ["▁", "<0x0A>"])
    }
}

@Suite("Gemma Specific Tests", .serialized)
struct GemmaSpecificTests {
    @Test
    func unicodeEdgeCase() async throws {
        let fixture = TokenizerTestFixture(config: .gemma)
        let tokenizer = try await fixture.tokenizer

        // These are two different characters
        let cases = ["à" /* 0x61 0x300 */, "à" /* 0xe0 */ ]
        let expected = [217138, 1305]

        // These are different characters
        for (s, expected) in zip(cases, expected) {
            let encoded = tokenizer?.encode(text: " " + s)
            #expect(encoded == [2, expected])
        }
    }

    @Test
    func gemmaVocab() async throws {
        guard let tokenizer = try await AutoTokenizer.from(pretrained: "pcuenq/gemma-tokenizer") as? PreTrainedTokenizer else {
            Issue.record("Failed to load tokenizer")
            return
        }

        // FIXME: This should be 256_000, I believe
        #expect((tokenizer.model as? BPETokenizer)?.vocabCount == 255994)
    }
}

// MARK: - Other Specialized Tests

@Suite("Phi Tests")
struct PhiTests {
    @Test
    func phi4() async throws {
        guard let tokenizer = try await AutoTokenizer.from(pretrained: "microsoft/phi-4") as? PreTrainedTokenizer else {
            Issue.record("Failed to load tokenizer")
            return
        }

        #expect(tokenizer.encode(text: "hello") == [15339])
        #expect(tokenizer.encode(text: "hello world") == [15339, 1917])
        #expect(tokenizer.encode(text: "<|im_start|>user<|im_sep|>Who are you?<|im_end|><|im_start|>assistant<|im_sep|>") == [100264, 882, 100266, 15546, 527, 499, 30, 100265, 100264, 78191, 100266])
    }
}

@Suite("Llama Post-Processor Override Tests")
struct LlamaPostProcessorOverrideTests {
    @Test
    func deepSeek() async throws {
        guard let tokenizer = try await AutoTokenizer.from(pretrained: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B") as? PreTrainedTokenizer else {
            Issue.record("Failed to load tokenizer")
            return
        }
        #expect(tokenizer.encode(text: "Who are you?") == [151646, 15191, 525, 498, 30])
    }

    @Test
    func testLlama() async throws {
        guard let tokenizer = try await AutoTokenizer.from(pretrained: "coreml-projects/Llama-2-7b-chat-coreml") as? PreTrainedTokenizer else {
            Issue.record("Failed to load tokenizer")
            return
        }
        #expect(tokenizer.encode(text: "Who are you?") == [1, 11644, 526, 366, 29973])
    }
}

@Suite("Local Pretrained Tests")
struct LocalFromPretrainedTests {
    @Test
    func localTokenizerFromPretrained() async throws {
        let downloadDestination: URL = {
            let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            return base.appending(component: "hf-local-pretrained-tests-downloads")
        }()

        let hubApi = HubApi(downloadBase: downloadDestination)
        let downloadedTo = try await hubApi.snapshot(from: "pcuenq/gemma-tokenizer")

        let tokenizer = try await AutoTokenizer.from(modelFolder: downloadedTo) as? PreTrainedTokenizer
        #expect(tokenizer != nil)

        try FileManager.default.removeItem(at: downloadDestination)
    }
}

@Suite("BERT Diacritics Tests")
struct BertDiacriticsTests {
    @Test
    func testBertCased() async throws {
        guard let tokenizer = try await AutoTokenizer.from(pretrained: "distilbert/distilbert-base-multilingual-cased") as? PreTrainedTokenizer else {
            Issue.record()
            return
        }

        #expect(tokenizer.encode(text: "mąka") == [101, 181, 102075, 10113, 102])
        #expect(tokenizer.tokenize(text: "Car") == ["Car"])
    }

    @Test
    func bertCasedResaved() async throws {
        guard let tokenizer = try await AutoTokenizer.from(pretrained: "pcuenq/distilbert-base-multilingual-cased-tokenizer") as? PreTrainedTokenizer else {
            Issue.record()
            return
        }

        #expect(tokenizer.encode(text: "mąka") == [101, 181, 102075, 10113, 102])
    }

    @Test
    func testBertUncased() async throws {
        guard let tokenizer = try await AutoTokenizer.from(pretrained: "google-bert/bert-base-uncased") as? PreTrainedTokenizer else {
            Issue.record()
            return
        }

        #expect(tokenizer.tokenize(text: "mąka") == ["ma", "##ka"])
        #expect(tokenizer.encode(text: "mąka") == [101, 5003, 2912, 102])
        #expect(tokenizer.tokenize(text: "département") == ["depart", "##ement"])
        #expect(tokenizer.encode(text: "département") == [101, 18280, 13665, 102])
        #expect(tokenizer.tokenize(text: "Car") == ["car"])

        #expect(tokenizer.tokenize(text: "€4") == ["€", "##4"])
        #expect(tokenizer.tokenize(text: "test $1 R2 #3 €4 £5 ¥6 ₣7 ₹8 ₱9 test") == ["test", "$", "1", "r", "##2", "#", "3", "€", "##4", "£5", "¥", "##6", "[UNK]", "₹", "##8", "₱", "##9", "test"])
    }
}

@Suite("BERT Spaces Tests")
struct BertSpacesTests {
    @Test
    func encodeDecode() async throws {
        guard let tokenizer = try await AutoTokenizer.from(pretrained: "google-bert/bert-base-uncased") as? PreTrainedTokenizer else {
            Issue.record()
            return
        }

        let text = "l'eure"
        let tokenized = tokenizer.tokenize(text: text)
        #expect(tokenized == ["l", "'", "eu", "##re"])
        let encoded = tokenizer.encode(text: text)
        #expect(encoded == [101, 1048, 1005, 7327, 2890, 102])
        let decoded = tokenizer.decode(tokens: encoded, skipSpecialTokens: true)
        // Note: this matches the behaviour of the Python "slow" tokenizer, but the fast one produces "l ' eure"
        #expect(decoded == "l'eure")
    }
}

@Suite("RoBERTa Tests")
struct RobertaTests {
    @Test
    func testEncodeDecode() async throws {
        guard let tokenizer = try await AutoTokenizer.from(pretrained: "FacebookAI/roberta-base") as? PreTrainedTokenizer else {
            Issue.record()
            return
        }

        #expect(tokenizer.tokenize(text: "l'eure") == ["l", "'", "e", "ure"])
        #expect(tokenizer.encode(text: "l'eure") == [0, 462, 108, 242, 2407, 2])
        #expect(tokenizer.decode(tokens: tokenizer.encode(text: "l'eure"), skipSpecialTokens: true) == "l'eure")

        #expect(tokenizer.tokenize(text: "mąka") == ["m", "Ä", "ħ", "ka"])
        #expect(tokenizer.encode(text: "mąka") == [0, 119, 649, 5782, 2348, 2])

        #expect(tokenizer.tokenize(text: "département") == ["d", "Ã©", "part", "ement"])
        #expect(tokenizer.encode(text: "département") == [0, 417, 1140, 7755, 6285, 2])

        #expect(tokenizer.tokenize(text: "Who are you?") == ["Who", "Ġare", "Ġyou", "?"])
        #expect(tokenizer.encode(text: "Who are you?") == [0, 12375, 32, 47, 116, 2])

        #expect(tokenizer.tokenize(text: " Who are you? ") == ["ĠWho", "Ġare", "Ġyou", "?", "Ġ"])
        #expect(tokenizer.encode(text: " Who are you? ") == [0, 3394, 32, 47, 116, 1437, 2])

        #expect(tokenizer.tokenize(text: "<s>Who are you?</s>") == ["<s>", "Who", "Ġare", "Ġyou", "?", "</s>"])
        #expect(tokenizer.encode(text: "<s>Who are you?</s>") == [0, 0, 12375, 32, 47, 116, 2, 2])
    }
}
