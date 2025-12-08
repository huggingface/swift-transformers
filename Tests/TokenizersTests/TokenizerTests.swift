//
//  TokenizerTests.swift
//
//  Created by Pedro Cuenca on July 2023.
//  Based on GPT2TokenizerTests by Julien Chaumond.
//  Copyright © 2023 Hugging Face. All rights reserved.
//

import Foundation
import Testing

@testable import Hub
@testable import Models
@testable import Tokenizers

private let downloadDestination: URL = {
    let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    return base.appending(component: "huggingface-tests")
}()

private let hubApiForTests = HubApi(downloadBase: downloadDestination)

private enum TestError: Error { case unsupportedTokenizer }

private struct Dataset: Decodable {
    let text: String
    // Bad naming, not just for bpe.
    // We are going to replace this testing method anyway.
    let bpe_tokens: [String]
    let token_ids: [Int]
    let decoded_text: String
}

private func loadDataset(filename: String) throws -> Dataset {
    let url = Bundle.module.url(forResource: filename, withExtension: "json")!
    let json = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    return try decoder.decode(Dataset.self, from: json)
}

private struct EdgeCase: Decodable {
    let input: String

    struct EncodedData: Decodable {
        let input_ids: [Int]
        let token_type_ids: [Int]?
        let attention_mask: [Int]
    }

    let encoded: EncodedData
    let decoded_with_special: String
    let decoded_without_special: String
}

private func loadEdgeCases(for hubModelName: String) throws -> [EdgeCase]? {
    let url = Bundle.module.url(forResource: "tokenizer_tests", withExtension: "json")!
    let json = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    let cases = try decoder.decode([String: [EdgeCase]].self, from: json)
    return cases[hubModelName]
}

private func makeTokenizer(hubModelName: String, hubApi: HubApi) async throws -> PreTrainedTokenizer {
    let config = LanguageModelConfigurationFromHub(modelName: hubModelName, hubApi: hubApi)
    guard let tokenizerConfig = try await config.tokenizerConfig else {
        throw TokenizerError.tokenizerConfigNotFound
    }
    let tokenizerData = try await config.tokenizerData
    let tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    guard let pretrained = tokenizer as? PreTrainedTokenizer else {
        throw TestError.unsupportedTokenizer
    }
    return pretrained
}

// MARK: -

struct ModelSpec: Sendable, CustomStringConvertible {
    let hubModelName: String
    let encodedSamplesFilename: String
    let unknownTokenId: Int?

    var description: String {
        hubModelName
    }

    init(_ hubModelName: String, _ encodedSamplesFilename: String, _ unknownTokenId: Int? = nil) {
        self.hubModelName = hubModelName
        self.encodedSamplesFilename = encodedSamplesFilename
        self.unknownTokenId = unknownTokenId
    }
}

// MARK: -

@Suite("Tokenizer Tests")
struct TokenizerTests {
    @Test(arguments: [
        ModelSpec("coreml-projects/Llama-2-7b-chat-coreml", "llama_encoded", 0),
        ModelSpec("distilbert/distilbert-base-multilingual-cased", "distilbert_cased_encoded", 100),
        ModelSpec("distilgpt2", "gpt2_encoded_tokens", 50256),
        ModelSpec("openai/whisper-large-v2", "whisper_large_v2_encoded", 50257),
        ModelSpec("openai/whisper-tiny.en", "whisper_tiny_en_encoded", 50256),
        ModelSpec("pcuenq/Llama-3.2-1B-Instruct-tokenizer", "llama_3.2_encoded"),
        ModelSpec("t5-base", "t5_base_encoded", 2),
        ModelSpec("tiiuae/falcon-7b", "falcon_encoded"),
    ])
    func tokenizer(spec: ModelSpec) async throws {
        let tokenizer = try await makeTokenizer(hubModelName: spec.hubModelName, hubApi: hubApiForTests)
        let dataset = try loadDataset(filename: spec.encodedSamplesFilename)

        #expect(tokenizer.tokenize(text: dataset.text) == dataset.bpe_tokens)
        #expect(tokenizer.encode(text: dataset.text) == dataset.token_ids)
        #expect(tokenizer.decode(tokens: dataset.token_ids) == dataset.decoded_text)

        // Edge cases (if available)
        if let edgeCases = try? loadEdgeCases(for: spec.hubModelName) {
            for edgeCase in edgeCases {
                #expect(tokenizer.encode(text: edgeCase.input) == edgeCase.encoded.input_ids)
                #expect(tokenizer.decode(tokens: edgeCase.encoded.input_ids) == edgeCase.decoded_with_special)
                #expect(tokenizer.decode(tokens: edgeCase.encoded.input_ids, skipSpecialTokens: true) == edgeCase.decoded_without_special)
            }
        }

        // Unknown token checks
        let model = tokenizer.model
        #expect(model.unknownTokenId == spec.unknownTokenId)
        #expect(model.unknownTokenId == model.convertTokenToId("_this_token_does_not_exist_"))
        if let unknownTokenId = model.unknownTokenId {
            #expect(model.unknownToken == model.convertIdToToken(unknownTokenId))
        } else {
            #expect(model.unknownTokenId == nil)
        }
    }

    @Test
    func gemmaUnicode() async throws {
        let tokenizerOpt = try await AutoTokenizer.from(pretrained: "pcuenq/gemma-tokenizer") as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        // These are two different characters
        let cases = [
            "à", // 0x61 0x300
            "à", // 0xe0
        ]
        let expected = [217138, 1305]
        for (s, expected) in zip(cases, expected) {
            let encoded = tokenizer.encode(text: " " + s)
            #expect(encoded == [2, expected])
        }

        // Keys that start with BOM sequence
        // https://github.com/huggingface/swift-transformers/issues/88
        // https://github.com/ml-explore/mlx-swift-examples/issues/50#issuecomment-2046592213
        #expect(tokenizer.convertIdToToken(122661) == "\u{feff}#")
        #expect(tokenizer.convertIdToToken(235345) == "#")

        // Verifies all expected entries are parsed
        #expect((tokenizer.model as? BPETokenizer)?.vocabCount == 256_000)

        // Test added tokens
        let inputIds = tokenizer("This\n\nis\na\ntest.")
        #expect(inputIds == [2, 1596, 109, 502, 108, 235250, 108, 2195, 235265])
        let decoded = tokenizer.decode(tokens: inputIds)
        #expect(decoded == "<bos>This\n\nis\na\ntest.")
    }

    @Test
    func phi4() async throws {
        let tokenizerOpt = try await AutoTokenizer.from(pretrained: "microsoft/phi-4") as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(tokenizer.encode(text: "hello") == [15339])
        #expect(tokenizer.encode(text: "hello world") == [15339, 1917])
        #expect(tokenizer.encode(text: "<|im_start|>user<|im_sep|>Who are you?<|im_end|><|im_start|>assistant<|im_sep|>") == [100264, 882, 100266, 15546, 527, 499, 30, 100265, 100264, 78191, 100266])
    }

    @Test
    func tokenizerFromLocalFolder() async throws {
        let bundle = Bundle.module
        guard
            let tokenizerConfigURL = bundle.url(
                forResource: "tokenizer_config",
                withExtension: "json"
            ),
            bundle.url(
                forResource: "tokenizer",
                withExtension: "json"
            ) != nil
        else {
            Issue.record("Missing offline tokenizer fixtures")
            return
        }

        let configuration = LanguageModelConfigurationFromHub(modelFolder: tokenizerConfigURL.deletingLastPathComponent())

        let tokenizerConfigOpt = try await configuration.tokenizerConfig
        #expect(tokenizerConfigOpt != nil)
        let tokenizerConfig = tokenizerConfigOpt!
        let tokenizerData = try await configuration.tokenizerData

        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData
        )

        let encoded = tokenizer.encode(text: "offline path")
        #expect(!encoded.isEmpty)
    }

    /// https://github.com/huggingface/swift-transformers/issues/96
    @Test
    func legacyLlamaBehaviour() async throws {
        let tokenizerOpt = try await AutoTokenizer.from(pretrained: "mlx-community/Phi-3-mini-4k-instruct-4bit-no-q-embed") as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        let inputIds = tokenizer(" Hi")
        #expect(inputIds == [1, 29871, 6324])
    }

    /// https://github.com/huggingface/swift-transformers/issues/99
    @Test
    func robertaXLMTokenizer() async throws {
        let tokenizerOpt = try await AutoTokenizer.from(pretrained: "intfloat/multilingual-e5-small") as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        let ids = tokenizer.encode(text: "query: how much protein should a female eat")
        let expected = [0, 41, 1294, 12, 3642, 5045, 21308, 5608, 10, 117776, 73203, 2]
        #expect(ids == expected)
    }

    @Test
    func nllbTokenizer() async throws {
        do {
            _ = try await AutoTokenizer.from(pretrained: "Xenova/nllb-200-distilled-600M")
            Issue.record("Expected AutoTokenizer.from to throw for strict mode")
        } catch {
            // Expected to throw in normal (strict) mode
        }

        // no strict mode proceeds
        let tokenizerOpt = try await AutoTokenizer.from(pretrained: "Xenova/nllb-200-distilled-600M", strict: false) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        let ids = tokenizer.encode(text: "Why did the chicken cross the road?")
        let expected = [256047, 24185, 4077, 349, 1001, 22690, 83580, 349, 82801, 248130, 2]
        #expect(ids == expected)
    }

    /// Deepseek needs a post-processor override to add a bos token as in the reference implementation
    @Test
    func deepSeekPostProcessor() async throws {
        let tokenizerOpt = try await AutoTokenizer.from(pretrained: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B") as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!
        #expect(tokenizer.encode(text: "Who are you?") == [151646, 15191, 525, 498, 30])
    }

    /// Some Llama tokenizers already use a bos-prepending Template post-processor
    @Test
    func llamaPostProcessor() async throws {
        let tokenizerOpt = try await AutoTokenizer.from(pretrained: "coreml-projects/Llama-2-7b-chat-coreml") as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!
        #expect(tokenizer.encode(text: "Who are you?") == [1, 11644, 526, 366, 29973])
    }

    @Test
    func localTokenizerFromPretrained() async throws {
        let downloadDestination: URL = {
            let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            return base.appending(component: "hf-local-pretrained-tests-downloads")
        }()

        let hubApi = HubApi(downloadBase: downloadDestination)
        let downloadedTo = try await hubApi.snapshot(from: "pcuenq/gemma-tokenizer")

        let tokenizerOpt = try await AutoTokenizer.from(modelFolder: downloadedTo) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)

        try FileManager.default.removeItem(at: downloadDestination)
    }

    @Test
    func bertCased() async throws {
        let tokenizerOpt = try await AutoTokenizer.from(pretrained: "distilbert/distilbert-base-multilingual-cased") as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(tokenizer.encode(text: "mąka") == [101, 181, 102075, 10113, 102])
        #expect(tokenizer.tokenize(text: "Car") == ["Car"])
    }

    @Test
    func bertCasedResaved() async throws {
        let tokenizerOpt = try await AutoTokenizer.from(pretrained: "pcuenq/distilbert-base-multilingual-cased-tokenizer") as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(tokenizer.encode(text: "mąka") == [101, 181, 102075, 10113, 102])
    }

    @Test
    func bertUncased() async throws {
        let tokenizerOpt = try await AutoTokenizer.from(pretrained: "google-bert/bert-base-uncased") as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(tokenizer.tokenize(text: "mąka") == ["ma", "##ka"])
        #expect(tokenizer.encode(text: "mąka") == [101, 5003, 2912, 102])
        #expect(tokenizer.tokenize(text: "département") == ["depart", "##ement"])
        #expect(tokenizer.encode(text: "département") == [101, 18280, 13665, 102])
        #expect(tokenizer.tokenize(text: "Car") == ["car"])

        #expect(tokenizer.tokenize(text: "€4") == ["€", "##4"])
        #expect(tokenizer.tokenize(text: "test $1 R2 #3 €4 £5 ¥6 ₣7 ₹8 ₱9 test") == ["test", "$", "1", "r", "##2", "#", "3", "€", "##4", "£5", "¥", "##6", "[UNK]", "₹", "##8", "₱", "##9", "test"])

        let text = "l'eure"
        let tokenized = tokenizer.tokenize(text: text)
        #expect(tokenized == ["l", "'", "eu", "##re"])
        let encoded = tokenizer.encode(text: text)
        #expect(encoded == [101, 1048, 1005, 7327, 2890, 102])
        let decoded = tokenizer.decode(tokens: encoded, skipSpecialTokens: true)
        // Note: this matches the behaviour of the Python "slow" tokenizer, but the fast one produces "l ' eure"
        #expect(decoded == "l'eure")

        // Reading added_tokens from tokenizer.json
        #expect(tokenizer.convertTokenToId("[PAD]") == 0)
        #expect(tokenizer.convertTokenToId("[UNK]") == 100)
        #expect(tokenizer.convertTokenToId("[CLS]") == 101)
        #expect(tokenizer.convertTokenToId("[SEP]") == 102)
        #expect(tokenizer.convertTokenToId("[MASK]") == 103)
    }

    @Test
    func robertaEncodeDecode() async throws {
        let tokenizerOpt = try await AutoTokenizer.from(pretrained: "FacebookAI/roberta-base") as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

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

    @Test
    func tokenizerBackend() async throws {
        let tokenizerOpt = try await AutoTokenizer.from(pretrained: "mlx-community/Ministral-3-3B-Instruct-2512-4bit") as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(tokenizer.encode(text: "She took a train to the West") == [6284, 5244, 1261, 10018, 1317, 1278, 5046])
    }
}
