//
//  HubApiTests.swift
//
//  Created by Pedro Cuenca on 20231230.
//

@testable import Hub
import XCTest

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

    func testFilenameRetrievalWithMultiplePatterns() async {
        do {
            let patterns = ["config.json", "tokenizer.json", "tokenizer_*.json"]
            let filenames = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml", matching: patterns)
            XCTAssertEqual(
                Set(filenames),
                Set(["config.json", "tokenizer.json", "tokenizer_config.json"])
            )
        } catch {
            XCTFail("\(error)")
        }
    }
    
    func testGetFileMetadata() async throws {
        do {
            let url = URL(string: "https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/config.json")
            let metadata = try await Hub.getFileMetadata(fileURL: url!)
            
            XCTAssertNotNil(metadata.commitHash)
            XCTAssertNotNil(metadata.etag)
            XCTAssertEqual(metadata.location, url?.absoluteString)
            XCTAssertEqual(metadata.size, 163)
        } catch {
            XCTFail("\(error)")
        }
    }
    
    func testGetFileMetadataBlobPath() async throws {
        do {
            let url = URL(string: "https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/blob/main/config.json")
            let metadata = try await Hub.getFileMetadata(fileURL: url!)
            
            XCTAssertEqual(metadata.commitHash, nil)
            XCTAssertTrue(metadata.etag != nil && metadata.etag!.hasPrefix("1081d-"))
            XCTAssertEqual(metadata.location, url?.absoluteString)
            XCTAssertEqual(metadata.size, 67613)
        } catch {
            XCTFail("\(error)")
        }
    }
    
    func testGetFileMetadataWithRevision() async throws {
        do {
            let revision = "f2c752cfc5c0ab6f4bdec59acea69eefbee381c2"
            let url = URL(string: "https://huggingface.co/julien-c/dummy-unknown/resolve/\(revision)/config.json")
            let metadata = try await Hub.getFileMetadata(fileURL: url!)
            
            XCTAssertEqual(metadata.commitHash, revision)
            XCTAssertNotNil(metadata.etag)
            XCTAssertGreaterThan(metadata.etag!.count, 0)
            XCTAssertEqual(metadata.location, url?.absoluteString)
            XCTAssertEqual(metadata.size, 851)
        } catch {
            XCTFail("\(error)")
        }
    }

    func testGetFileMetadataWithBlobSearch() async throws {
        let repo = "coreml-projects/Llama-2-7b-chat-coreml"
        let metadataFromBlob = try await Hub.getFileMetadata(from: repo, matching: "*.json").sorted { $0.location < $1.location }
        let files = try await Hub.getFilenames(from: repo, matching: "*.json").sorted()
        for (metadata, file) in zip(metadataFromBlob, files) {
            XCTAssertNotNil(metadata.commitHash)
            XCTAssertNotNil(metadata.etag)
            XCTAssertGreaterThan(metadata.etag!.count, 0)
            XCTAssertTrue(metadata.location.contains(file))
            XCTAssertGreaterThan(metadata.size!, 0)
        }
    }
}

class SnapshotDownloadTests: XCTestCase {
    let repo = "coreml-projects/Llama-2-7b-chat-coreml"
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

    func getRelativeFiles(url: URL) -> [String] {
        var filenames: [String] = []
        let prefix = downloadDestination.appending(path: "models/\(repo)").path.appending("/")

        if let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles], errorHandler: nil) {
            for case let fileURL as URL in enumerator {
                do {
                    let resourceValues = try fileURL.resourceValues(forKeys: [.isRegularFileKey])
                    if resourceValues.isRegularFile == true {
                        filenames.append(String(fileURL.path.suffix(from: prefix.endIndex)))
                    }
                } catch {
                    print("Error reading file resources: \(error)")
                }
            }
        }
        return filenames
    }

    func testDownload() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }
        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 6)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination)
        XCTAssertEqual(
            Set(downloadedFilenames),
            Set([
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "llama-2-7b-chat.mlpackage/Manifest.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
            ])
        )
    }

    /// Background sessions get rate limited by the OS, see discussion here: https://github.com/huggingface/swift-transformers/issues/61
    /// Test only one file at a time
    func testDownloadInBackground() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination, useBackgroundSession: true)
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }
        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 1)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination)
        XCTAssertEqual(
            Set(downloadedFilenames),
            Set([
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
            ])
        )
    }

    func testCustomEndpointDownload() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination, endpoint: "https://hf-mirror.com")
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }
        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 6)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination)
        XCTAssertEqual(
            Set(downloadedFilenames),
            Set([
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "llama-2-7b-chat.mlpackage/Manifest.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
            ])
        )
    }
}
