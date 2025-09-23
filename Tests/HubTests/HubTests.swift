//
//  HubTests.swift
//
//  Created by Pedro Cuenca on 18/05/2023.
//

import XCTest

@testable import Hub

class HubTests: XCTestCase {
    let downloadDestination: URL = {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appending(component: "huggingface-tests")
    }()

    override func setUp() {}

    override func tearDown() {
        do {
            try FileManager.default.removeItem(at: downloadDestination)
        } catch {
            print("Can't remove test download destination \(downloadDestination), error: \(error)")
        }
    }

    var hubApi: HubApi { HubApi(downloadBase: downloadDestination) }

    func testConfigDownload() async {
        do {
            let configLoader = LanguageModelConfigurationFromHub(modelName: "t5-base", hubApi: hubApi)
            let config = try await configLoader.modelConfig

            // Test leaf value (Int)
            guard let eos = config["eos_token_id"].integer() else {
                XCTFail("nil leaf value (Int)")
                return
            }
            XCTAssertEqual(eos, 1)

            // Test leaf value (String)
            guard let modelType = config["model_type"].string() else {
                XCTFail("nil leaf value (String)")
                return
            }
            XCTAssertEqual(modelType, "t5")

            // Test leaf value (Array)
            guard let architectures: [String] = config["architectures"].get() else {
                XCTFail("nil array")
                return
            }
            XCTAssertEqual(architectures, ["T5ForConditionalGeneration"])

            // Test nested wrapper
            guard !config["task_specific_params"].isNull() else {
                XCTFail("nil nested wrapper")
                return
            }

            guard let summarizationMaxLength = config["task_specific_params"]["summarization"]["max_length"].integer() else {
                XCTFail("cannot traverse nested containers")
                return
            }
            XCTAssertEqual(summarizationMaxLength, 200)
        } catch {
            XCTFail("Cannot download test configuration from the Hub: \(error)")
        }
    }

    func testConfigCamelCase() async {
        do {
            let configLoader = LanguageModelConfigurationFromHub(modelName: "t5-base", hubApi: hubApi)
            let config = try await configLoader.modelConfig

            // Test leaf value (Int)
            guard let eos = config["eosTokenId"].integer() else {
                XCTFail("nil leaf value (Int)")
                return
            }
            XCTAssertEqual(eos, 1)

            // Test leaf value (String)
            guard let modelType = config["modelType"].string() else {
                XCTFail("nil leaf value (String)")
                return
            }
            XCTAssertEqual(modelType, "t5")

            guard let summarizationMaxLength = config["taskSpecificParams"]["summarization"]["maxLength"].integer() else {
                XCTFail("cannot traverse nested containers")
                return
            }
            XCTAssertEqual(summarizationMaxLength, 200)
        } catch {
            XCTFail("Cannot download test configuration from the Hub: \(error)")
        }
    }
}
