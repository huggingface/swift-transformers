//
//  DownloaderTests.swift
//  swift-transformers
//
//  Created by Arda Atahan Ibis on 1/28/25.
//

import Combine
import XCTest

@testable import Hub
import XCTest

/// Errors that can occur during the download process
enum DownloadError: LocalizedError {
    case invalidDownloadLocation
    case unexpectedError

    var errorDescription: String? {
        switch self {
        case .invalidDownloadLocation:
            String(localized: "The download location is invalid or inaccessible.", comment: "Error when download destination is invalid")
        case .unexpectedError:
            String(localized: "An unexpected error occurred during the download process.", comment: "Generic download error message")
        }
    }
}

private extension Downloader {
    func interruptDownload() async {
        await session.get()?.invalidateAndCancel()
    }
}

final class DownloaderTests: XCTestCase {
    var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() {
        if let tempDir, FileManager.default.fileExists(atPath: tempDir.path) {
            try? FileManager.default.removeItem(at: tempDir)
        }
        super.tearDown()
    }

    /// This test downloads a known config file, verifies the download completes, checks the content matches expected value
    func testSuccessfulDownload() async throws {
        // Create a test file
        let url = URL(string: "https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/config.json")!
        let etag = try await Hub.getFileMetadata(fileURL: url).etag!
        let destination = tempDir.appendingPathComponent("config.json")
        let fileContent = """
        {
          "architectures": [
            "LlamaForCausalLM"
          ],
          "bos_token_id": 1,
          "eos_token_id": 2,
          "model_type": "llama",
          "pad_token_id": 0,
          "vocab_size": 32000
        }

        """

        let cacheDir = tempDir.appendingPathComponent("cache")
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let incompleteDestination = cacheDir.appendingPathComponent("config.json.\(etag).incomplete")
        FileManager.default.createFile(atPath: incompleteDestination.path, contents: nil, attributes: nil)

        let downloader = Downloader(to: destination, incompleteDestination: incompleteDestination)
        let sub = await downloader.download(from: url)

        listen: for await state in sub {
            switch state {
            case .notStarted:
                continue
            case .downloading:
                continue
            case let .failed(error):
                throw error
            case .completed:
                break listen
            }
        }

        // Verify download completed successfully
        XCTAssertTrue(FileManager.default.fileExists(atPath: destination.path))
        XCTAssertEqual(try String(contentsOf: destination, encoding: .utf8), fileContent)
    }

    /// This test attempts to download with incorrect expected file, verifies the download fails, ensures no partial file is left behind
    func testDownloadFailsWithIncorrectSize() async throws {
        let url = URL(string: "https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/config.json")!
        let etag = try await Hub.getFileMetadata(fileURL: url).etag!
        let destination = tempDir.appendingPathComponent("config.json")

        let cacheDir = tempDir.appendingPathComponent("cache")
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let incompleteDestination = cacheDir.appendingPathComponent("config.json.\(etag).incomplete")
        FileManager.default.createFile(atPath: incompleteDestination.path, contents: nil, attributes: nil)

        let downloader = Downloader(to: destination, incompleteDestination: incompleteDestination)
        // Download with incorrect expected size
        let sub = await downloader.download(from: url, expectedSize: 999999) // Incorrect size
        listen: for await state in sub {
            switch state {
            case .notStarted:
                continue
            case .downloading:
                continue
            case .failed:
                break listen
            case .completed:
                XCTFail("Download should have failed due to size mismatch")
                break listen
            }
        }

        // Verify no file was created at destination
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
    }

    /// This test downloads an LFS file, interrupts the download at 50% and 75% progress,
    /// verifies the download can resume and complete successfully, checks the final file exists and has content
    func testSuccessfulInterruptedDownload() async throws {
        let url = URL(string: "https://huggingface.co/coreml-projects/sam-2-studio/resolve/main/SAM%202%20Studio%201.1.zip")!
        let etag = try await Hub.getFileMetadata(fileURL: url).etag!
        let destination = tempDir.appendingPathComponent("SAM%202%20Studio%201.1.zip")

        // Create parent directory if it doesn't exist
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let cacheDir = tempDir.appendingPathComponent("cache")
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let incompleteDestination = cacheDir.appendingPathComponent("config.json.\(etag).incomplete")
        FileManager.default.createFile(atPath: incompleteDestination.path, contents: nil, attributes: nil)

        let downloader = Downloader(to: destination, incompleteDestination: incompleteDestination)
        let sub = await downloader.download(from: url, expectedSize: 73_194_001) // Correct size for verification

        // First interruption point at 50%
        var threshold = 0.5

        do {
            // Monitor download progress and interrupt at thresholds to test if
            // download continues from where it left off
            listen: for await state in sub {
                switch state {
                case .notStarted:
                    continue
                case let .downloading(progress):
                    if threshold != 1.0, progress >= threshold {
                        // Move to next threshold and interrupt
                        threshold = threshold == 0.5 ? 0.75 : 1.0
                        await downloader.interruptDownload()
                    }
                case let .failed(error):
                    throw error
                case .completed:
                    break listen
                }
            }

            // Verify the file exists and is complete
            if FileManager.default.fileExists(atPath: destination.path) {
                let attributes = try FileManager.default.attributesOfItem(atPath: destination.path)
                let finalSize = attributes[.size] as! Int64
                XCTAssertGreaterThan(finalSize, 0, "File should not be empty")
            } else {
                XCTFail("File was not created at destination")
            }
        } catch {
            throw error
        }
    }
}
