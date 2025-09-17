//
//  HubApiIntegrationTests.swift
//
//  Networked tests gated by HF_TOKEN=1
//

import Foundation
import Testing

@testable import Hub

@Suite("Hub API (integration)",
       .disabled(if: ProcessInfo.processInfo.environment["HF_TOKEN"] == "", "Set HF_TOKEN to run network tests"))
struct HubApiIntegrationTests {
    @Test(
        "config download (snake_case)",
        .timeLimit(.minutes(2))
    )
    func configDownloadSnakeCase() async throws {
        let downloadDestination = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: downloadDestination, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: downloadDestination) }

        let hubApi = HubApi(downloadBase: downloadDestination)
        let configLoader = LanguageModelConfigurationFromHub(modelName: "t5-base", hubApi: hubApi)
        let config = try await configLoader.modelConfig

        #expect(config["eos_token_id"].integer() == 1)
        #expect(config["model_type"].string() == "t5")
        let architectures: [String]? = config["architectures"].get()
        #expect(architectures == ["T5ForConditionalGeneration"])
        #expect(!config["task_specific_params"].isNull())
        #expect(config["task_specific_params"]["summarization"]["max_length"].integer() == 200)
    }

    @Test(
        "config download (camelCase)",
        .timeLimit(.minutes(2))
    )
    func configDownloadCamelCase() async throws {
        let downloadDestination = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: downloadDestination, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: downloadDestination) }

        let hubApi = HubApi(downloadBase: downloadDestination)
        let configLoader = LanguageModelConfigurationFromHub(modelName: "t5-base", hubApi: hubApi)
        let config = try await configLoader.modelConfig

        #expect(config["eosTokenId"].integer() == 1)
        #expect(config["modelType"].string() == "t5")
        #expect(config["taskSpecificParams"]["summarization"]["maxLength"].integer() == 200)
    }
}

@Suite("Snapshot download (integration)",
       .disabled(if: ProcessInfo.processInfo.environment["HF_TOKEN"] == "", "Set HF_TOKEN to run network tests"))
struct SnapshotDownloadIntegrationTests {
    let repo = "coreml-projects/Llama-2-7b-chat-coreml"
    let lfsRepo = "pcuenq/smol-lfs"

    private func getRelativeFiles(base: URL, url: URL, repo: String) -> [String] {
        var filenames: [String] = []
        let prefix = base.appending(path: "models/\(repo)").path.appending("/")
        if let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles],
            errorHandler: nil
        ) {
            for case let fileURL as URL in enumerator {
                do {
                    let resourceValues = try fileURL.resourceValues(forKeys: [.isRegularFileKey])
                    if resourceValues.isRegularFile == true {
                        filenames.append(String(fileURL.path.suffix(from: prefix.endIndex)))
                    }
                } catch {
                    // ignore for tests
                }
            }
        }
        return filenames
    }

    @Test("snapshot downloads JSON files", .timeLimit(.minutes(3)))
    func snapshotDownloadJson() async throws {
        let downloadDestination = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: downloadDestination, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: downloadDestination) }

        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 6)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(base: downloadDestination, url: downloadDestination, repo: repo)
        #expect(Set(downloadedFilenames) == Set([
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "llama-2-7b-chat.mlpackage/Manifest.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
        ]))
    }
}
