//
//  DownloaderTests.swift
//  swift-transformers
//
//  Created by Arda Atahan Ibis on 1/28/25.
//

import Foundation
import Testing

@testable import Hub

@Suite("Downloader (unit)")
struct DownloaderUnitTests {
    @Test("moveDownloadedFile respects percentEncoded flag and preserves existing file")
    func moveDownloadedFilePercentEncodedFlag() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let appSupport = tempDir.appendingPathComponent("Application Support")
        let destination = appSupport.appendingPathComponent("config.json")
        let source1 = tempDir.appendingPathComponent("v1.incomplete")
        let source2 = tempDir.appendingPathComponent("v2.incomplete")

        try FileManager.default.createDirectory(at: appSupport, withIntermediateDirectories: true)
        try "existing".write(to: destination, atomically: true, encoding: .utf8)
        try "v1".write(to: source1, atomically: true, encoding: .utf8)
        try "v2".write(to: source2, atomically: true, encoding: .utf8)

        do {
            try FileManager.default.moveDownloadedFile(from: source1, to: destination, percentEncoded: true)
            #expect(Bool(false), "Expected throw for percent-encoded path collision")
        } catch let error as NSError {
            #expect(error.code == 516)
        }
        #expect(try (String(contentsOf: destination)) == "existing")

        #expect(try { try FileManager.default.moveDownloadedFile(from: source2, to: destination, percentEncoded: false); return true }())
        #expect(try (String(contentsOf: destination)) == "v2")
    }

    @Test("incompleteFileSize returns size for existing partial file")
    func incompleteFileSize_read() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let partial = tempDir.appendingPathComponent("file.part")
        try Data(repeating: 0x41, count: 1234).write(to: partial)
        #expect(Downloader.incompleteFileSize(at: partial) == 1234)
    }
}

@Suite("Downloader (integration)",
       .disabled(if: ProcessInfo.processInfo.environment["HF_TOKEN"] == "", "Set HF_TOKEN to run network tests"))
struct DownloaderIntegrationTests {
    @Test("successful download and content matches", .timeLimit(.minutes(2)))
    func successfulDownload() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let url = URL(string: "https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/config.json")!
        let etag = try await Hub.getFileMetadata(fileURL: url).etag!
        let destination = tempDir.appendingPathComponent("config.json")
        let expected = """
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
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let incompleteDestination = cacheDir.appendingPathComponent("config.json.\(etag).incomplete")
        FileManager.default.createFile(atPath: incompleteDestination.path, contents: nil, attributes: nil)

        let downloader = Downloader(to: destination, incompleteDestination: incompleteDestination)
        let sub = await downloader.download(from: url)
        listen: for await state in sub {
            switch state {
            case .notStarted: continue
            case .downloading: continue
            case let .failed(error): throw error
            case .completed: break listen
            }
        }

        #expect(FileManager.default.fileExists(atPath: destination.path))
        #expect(try (String(contentsOf: destination, encoding: .utf8)) == expected)
    }

    @Test("download fails with incorrect expected size", .timeLimit(.minutes(2)))
    func downloadFailsWithIncorrectSize() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let url = URL(string: "https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/config.json")!
        let etag = try await Hub.getFileMetadata(fileURL: url).etag!
        let destination = tempDir.appendingPathComponent("config.json")

        let cacheDir = tempDir.appendingPathComponent("cache")
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let incompleteDestination = cacheDir.appendingPathComponent("config.json.\(etag).incomplete")
        FileManager.default.createFile(atPath: incompleteDestination.path, contents: nil, attributes: nil)

        let downloader = Downloader(to: destination, incompleteDestination: incompleteDestination)
        let sub = await downloader.download(from: url, expectedSize: 999_999)

        var failed = false
        listen: for await state in sub {
            switch state {
            case .notStarted, .downloading: continue
            case .failed:
                failed = true
                break listen
            case .completed:
                break listen
            }
        }

        #expect(failed)
        #expect(!FileManager.default.fileExists(atPath: destination.path))
    }

    @Test("interrupted LFS download resumes and completes", .timeLimit(.minutes(5)))
    func successfulInterruptedDownload() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let url = URL(string: "https://huggingface.co/coreml-projects/sam-2-studio/resolve/main/SAM%202%20Studio%201.1.zip")!
        let etag = try await Hub.getFileMetadata(fileURL: url).etag!
        let destination = tempDir.appendingPathComponent("SAM%202%20Studio%201.1.zip")

        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

        let cacheDir = tempDir.appendingPathComponent("cache")
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let incompleteDestination = cacheDir.appendingPathComponent("config.json.\(etag).incomplete")
        FileManager.default.createFile(atPath: incompleteDestination.path, contents: nil, attributes: nil)

        let downloader = Downloader(to: destination, incompleteDestination: incompleteDestination)
        let sub = await downloader.download(from: url, expectedSize: 73_194_001)

        var threshold = 0.5
        listen: for await state in sub {
            switch state {
            case .notStarted: continue
            case let .downloading(progress, _):
                if threshold != 1.0, progress >= threshold {
                    threshold = threshold == 0.5 ? 0.75 : 1.0
                    await downloader.session.get()?.invalidateAndCancel()
                }
            case let .failed(error): throw error
            case .completed: break listen
            }
        }

        #expect(FileManager.default.fileExists(atPath: destination.path))
        let attributes = try FileManager.default.attributesOfItem(atPath: destination.path)
        let finalSize = attributes[.size] as! Int64
        #expect(finalSize > 0)
    }
}
