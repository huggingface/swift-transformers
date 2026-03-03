//
//  HubTests.swift
//
//  Created by Pedro Cuenca on 18/05/2023.
//

import Foundation
import Testing

@testable import Hub

@Suite
class HubTests {
    let downloadDestination: URL = {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appending(component: "huggingface-tests-\(UUID().uuidString)")
    }()
    deinit {
        do {
            try FileManager.default.removeItem(at: downloadDestination)
        } catch {
            print("Can't remove test download destination \(downloadDestination), error: \(error)")
        }
    }

    var hubApi: HubApi { HubApi(downloadBase: downloadDestination) }

    @Test
    func testConfigDownload() async {
        do {
            let configLoader = LanguageModelConfigurationFromHub(modelName: "t5-base", hubApi: hubApi)
            guard let config = try await configLoader.modelConfig else {
                Issue.record("Test repo is expected to have a config.json file")
                return
            }

            // Test leaf value (Int)
            guard let eos = config["eos_token_id"].integer() else {
                Issue.record("nil leaf value (Int)")
                return
            }
            #expect(eos == 1)

            // Test leaf value (String)
            guard let modelType = config["model_type"].string() else {
                Issue.record("nil leaf value (String)")
                return
            }
            #expect(modelType == "t5")

            // Test leaf value (Array)
            guard let architectures: [String] = config["architectures"].get() else {
                Issue.record("nil array")
                return
            }
            #expect(architectures == ["T5ForConditionalGeneration"])

            // Test nested wrapper
            guard !config["task_specific_params"].isNull() else {
                Issue.record("nil nested wrapper")
                return
            }

            guard let summarizationMaxLength = config["task_specific_params"]["summarization"]["max_length"].integer() else {
                Issue.record("cannot traverse nested containers")
                return
            }
            #expect(summarizationMaxLength == 200)
        } catch {
            Issue.record("Cannot download test configuration from the Hub: \(error)")
        }
    }

    @Test
    func testConfigCamelCase() async {
        do {
            let configLoader = LanguageModelConfigurationFromHub(modelName: "t5-base", hubApi: hubApi)
            guard let config = try await configLoader.modelConfig else {
                Issue.record("Test repo is expected to have a config.json file")
                return
            }

            // Test leaf value (Int)
            guard let eos = config["eosTokenId"].integer() else {
                Issue.record("nil leaf value (Int)")
                return
            }
            #expect(eos == 1)

            // Test leaf value (String)
            guard let modelType = config["modelType"].string() else {
                Issue.record("nil leaf value (String)")
                return
            }
            #expect(modelType == "t5")

            guard let summarizationMaxLength = config["taskSpecificParams"]["summarization"]["maxLength"].integer() else {
                Issue.record("cannot traverse nested containers")
                return
            }
            #expect(summarizationMaxLength == 200)
        } catch {
            Issue.record("Cannot download test configuration from the Hub: \(error)")
        }
    }

    @Test
    func testNemotronInfinity() async throws {
        do {
            let configLoader = LanguageModelConfigurationFromHub(modelName: "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit", hubApi: hubApi)
            guard let config = try await configLoader.modelConfig else {
                Issue.record("Test repo is expected to have a config.json file")
                return
            }

            guard let timeStepLimit = config["timeStepLimit"].array() else {
                Issue.record("timeStepLimit could not be read")
                return
            }
            #expect(timeStepLimit[1].floating() == Float.infinity)
        } catch {
            Issue.record("Cannot download test configuration from the Hub: \(error)")
        }
    }
}
