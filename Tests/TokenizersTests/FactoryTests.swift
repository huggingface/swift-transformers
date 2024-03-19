//
//  FactoryTests.swift
//  
//
//  Created by Pedro Cuenca on 4/8/23.
//

import XCTest
import Tokenizers
import Hub

class TestWithCustomHubDownloadLocation: XCTestCase {
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

    var hubApi: HubApi {
        return HubApi(downloadBase: downloadDestination)
    }
}

class FactoryTests: TestWithCustomHubDownloadLocation {
    func testFromPretrained() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "coreml-projects/Llama-2-7b-chat-coreml", hubApi: hubApi)
        let inputIds = tokenizer("Today she took a train to the West")
        XCTAssertEqual(inputIds, [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122])
    }
    
    func testWhisper() async throws {
        let tokenizer = try await AutoTokenizer.from(pretrained: "openai/whisper-large-v2", hubApi: hubApi)
        let inputIds = tokenizer("Today she took a train to the West")
        XCTAssertEqual(inputIds, [50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257])
    }
}
