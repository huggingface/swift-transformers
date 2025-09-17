//
//  HubApiIntegrationTests.swift
//
//  Networked tests gated by HUB_NETWORK_TESTS=1
//

import Foundation
import Testing

@testable import Hub

@Suite("Hub API (integration)",
       .disabled(if: ProcessInfo.processInfo.environment["HUB_NETWORK_TESTS"] != "1", "Set HUB_NETWORK_TESTS=1 to run network tests"))
struct HubApiIntegrationTests {
    @Test(
        "config download (snake_case)",
        .timeLimit(.minutes(2)),
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
        .timeLimit(.minutes(2)),
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
