//
//  FactoryTests.swift
//
//
//  Created by Pedro Cuenca on 4/8/23.
//

import Foundation
import Hub
import Testing
import Tokenizers

@Suite
final class TestWithCustomHubDownloadLocation {
    let downloadDestination: URL = {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appending(component: "huggingface-tests")
    }()

    init() throws { }

    deinit {
        do {
            try FileManager.default.removeItem(at: downloadDestination)
        } catch {
            print("Can't remove test download destination \(downloadDestination), error: \(error)")
        }
    }

    var hubApi: HubApi {
        HubApi(downloadBase: downloadDestination)
    }
}

@Suite struct FactoryTests {
    let testWithCustomHubDownloadLocation = try! TestWithCustomHubDownloadLocation()

    var hubApi: HubApi {
        testWithCustomHubDownloadLocation.hubApi
    }

    @Test func fromPretrained() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "coreml-projects/Llama-2-7b-chat-coreml", hubApi: hubApi)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122])
    }

    @Test func whisper() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "openai/whisper-large-v2", hubApi: hubApi)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257])
    }

    @Test func fromModelFolder() async throws {
        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        let repo = Hub.Repo(id: "coreml-projects/Llama-2-7b-chat-coreml")
        let localModelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)

        let tokenizer = try await AutoTokenizer.from(modelFolder: localModelFolder, hubApi: hubApi)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122])
    }

    @Test func whisperFromModelFolder() async throws {
        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        let repo = Hub.Repo(id: "openai/whisper-large-v2")
        let localModelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)

        let tokenizer = try await AutoTokenizer.from(modelFolder: localModelFolder, hubApi: hubApi)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257])
    }
}
