//
//  HubTests.swift
//
//  Created by Pedro Cuenca on 18/05/2023.
//

@testable import Hub
import Testing
import Foundation

@Suite
class HubTests {
    let downloadDestination: URL = {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appending(component: "huggingface-tests")
    }()

    init() { }

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
            let config = try await configLoader.modelConfig

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
            let config = try await configLoader.modelConfig

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
    func testConfigUnicode() {
        // These are two different characters
        let json = "{\"vocab\": {\"à\": 1, \"à\": 2}}"
        let data = json.data(using: .utf8)
        let dict = try! JSONSerialization.jsonObject(with: data!, options: []) as! [NSString: Any]
        let config = Config(dict)

        let vocab = config["vocab"].dictionary(or: [:])

        #expect(vocab.count == 2)
    }

    @Test
    func testConfigTokenValue() throws {
        let config1 = Config(["cls": ["str" as String, 100 as UInt] as [Any]])
        let tokenValue1 = config1.cls?.token()
        #expect(tokenValue1?.0 == 100)
        #expect(tokenValue1?.1 == "str")

        let data = #"{"cls": ["str", 100]}"#.data(using: .utf8)!
        let dict = try JSONSerialization.jsonObject(with: data, options: []) as! [NSString: Any]
        let config2 = Config(dict)
        let tokenValue2 = config2.cls?.token()
        #expect(tokenValue2?.0 == 100)
        #expect(tokenValue2?.1 == "str")
    }
}
