//
//  FactoryTests.swift
//
//
//  Created by Pedro Cuenca on 4/8/23.
//

import Foundation
import Hub
import Testing

@testable import Tokenizers

private func makeHubApi() -> (api: HubApi, downloadDestination: URL) {
    let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    let destination = base.appending(component: "huggingface-tests-\(UUID().uuidString)")
    return (HubApi(downloadBase: destination), destination)
}

@Suite("Factory")
struct FactoryTests {
    @Test
    func fromPretrained() async throws {
        let (hubApi, downloadDestination) = makeHubApi()
        defer { try? FileManager.default.removeItem(at: downloadDestination) }

        let tokenizer = try await AutoTokenizer.from(pretrained: "coreml-projects/Llama-2-7b-chat-coreml", hubApi: hubApi)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122])
    }

    @Test
    func whisper() async throws {
        let (hubApi, downloadDestination) = makeHubApi()
        defer { try? FileManager.default.removeItem(at: downloadDestination) }

        let tokenizer = try await AutoTokenizer.from(pretrained: "openai/whisper-large-v2", hubApi: hubApi)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257])
    }

    @Test
    func fromModelFolder() async throws {
        let (hubApi, downloadDestination) = makeHubApi()
        defer { try? FileManager.default.removeItem(at: downloadDestination) }

        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        let repo = Hub.Repo(id: "coreml-projects/Llama-2-7b-chat-coreml")
        let localModelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)

        let tokenizer = try await AutoTokenizer.from(modelFolder: localModelFolder, hubApi: hubApi)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122])
    }

    @Test
    func whisperFromModelFolder() async throws {
        let (hubApi, downloadDestination) = makeHubApi()
        defer { try? FileManager.default.removeItem(at: downloadDestination) }

        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        let repo = Hub.Repo(id: "openai/whisper-large-v2")
        let localModelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)

        let tokenizer = try await AutoTokenizer.from(modelFolder: localModelFolder, hubApi: hubApi)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257])
    }

    @Test
    func modelsWithIncorrectHubTokenizerClassUseTokenizersBackend() async throws {
        for modelType in IncorrectHubTokenizerClassModel.allCases {
            let modelFolder = try makeModelFolder(modelType: modelType.rawValue)
            defer { try? FileManager.default.removeItem(at: modelFolder) }

            let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)

            #expect(tokenizer is PreTrainedTokenizer)
            #expect(!(tokenizer is LlamaPreTrainedTokenizer))
            #expect(tokenizer.encode(text: "<bos>", addSpecialTokens: false) == [0])
        }
    }

    @Test
    func repairsDeepseekOCRByteLevelTokenizerPipeline() async throws {
        let modelFolder = try makeModelFolder(
            modelType: "deepseekocr",
            malformedByteLevelPipeline: true
        )
        defer { try? FileManager.default.removeItem(at: modelFolder) }

        let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
        let tokens = tokenizer.tokenize(text: "document parsing. ")

        #expect(tokens.allSatisfy { tokenizer.convertTokenToId($0) != nil })
        #expect(
            tokenizer.encode(text: "document parsing. ", addSpecialTokens: false)
                == tokens.compactMap(tokenizer.convertTokenToId)
        )
    }
}

private enum IncorrectHubTokenizerClassModel: String, CaseIterable {
    case deepSeekOCR = "deepseek_ocr"
    case deepSeekOCR2 = "deepseek_ocr2"
}

private func makeModelFolder(
    modelType: String,
    malformedByteLevelPipeline: Bool = false
) throws -> URL {
    let modelFolder = FileManager.default.temporaryDirectory.appendingPathComponent(
        "swift-transformers-tokenizer-test-\(UUID().uuidString)",
        isDirectory: true
    )
    try FileManager.default.createDirectory(
        at: modelFolder,
        withIntermediateDirectories: true
    )

    try Data(#"{"model_type":"\#(modelType)"}"#.utf8).write(
        to: modelFolder.appendingPathComponent("config.json")
    )
    try Data(
        #"""
        {
          "tokenizer_class": "LlamaTokenizerFast",
          "bos_token": "<bos>",
          "eos_token": "<eos>",
          "unk_token": "<unk>",
          "pad_token": "<pad>"
        }
        """#.utf8
    ).write(to: modelFolder.appendingPathComponent("tokenizer_config.json"))

    guard
        let tokenizerURL = Bundle.module.url(
            forResource: "tokenizer",
            withExtension: "json"
        )
    else {
        throw CocoaError(.fileNoSuchFile)
    }
    let destinationURL = modelFolder.appendingPathComponent("tokenizer.json")
    if malformedByteLevelPipeline {
        let data = try Data(contentsOf: tokenizerURL)
        var tokenizerData = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        tokenizerData["pre_tokenizer"] = [
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
            "split": false,
        ]
        tokenizerData["decoder"] = [
            "type": "Sequence",
            "decoders": [],
        ]
        try JSONSerialization.data(withJSONObject: tokenizerData).write(to: destinationURL)
    } else {
        try FileManager.default.copyItem(at: tokenizerURL, to: destinationURL)
    }

    return modelFolder
}
