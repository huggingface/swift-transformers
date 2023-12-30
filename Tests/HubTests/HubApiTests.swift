//
//  HubApiTests.swift
//
//  Created by Pedro Cuenca on 20233012.
//

import XCTest
@testable import Hub


class HubApiTests: XCTestCase {

    override func setUp() {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    // MARK: use a specific revision for these tests
    
    func testFilenameRetrieval() async {
        do {
            let filenames = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml")
            XCTAssertEqual(filenames.count, 13)
        } catch {
            XCTFail("\(error)")
        }
    }
    
    func testFilenameRetrievalWithGlob() async {
        do {
            try await {
                let filenames = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml", matching: "*.json")
                XCTAssertEqual(
                    Set(filenames),
                    Set([
                        "config.json", "tokenizer.json", "tokenizer_config.json",
                        "llama-2-7b-chat.mlpackage/Manifest.json",
                        "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
                        "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
                    ])
                )
            }()

            // Glob patterns are case sensitive
            try await {
                let filenames = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml", matching: "*.JSON")
                XCTAssertEqual(
                    filenames,
                    []
                )
            }()
        } catch {
            XCTFail("\(error)")
        }
    }

    func testFilenameRetrievalFromDirectories() async {
        do {
            // Contents of all directories matching a pattern
            let filenames = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml", matching: "*.mlpackage/*")
            XCTAssertEqual(
                Set(filenames),
                Set([
                    "llama-2-7b-chat.mlpackage/Manifest.json",
                    "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
                    "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
                    "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                    "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/weights/weight.bin",

                ])
            )
        } catch {
            XCTFail("\(error)")
        }
    }

}
