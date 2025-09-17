//
//  HubApiTests.swift
//
//  Created by Pedro Cuenca on 20231230.
//

import Foundation
import Testing

@testable import Hub

@Suite(
    "Hub API (filenames and metadata)",
    .disabled(if: ProcessInfo.processInfo.environment["HF_TOKEN"] == "", "Set HF_TOKEN to run network tests")
)
struct HubApiFilenamesAndMetadataTests {
    // TODO: use a specific revision for these tests

    @Test("filename retrieval")
    func filenameRetrieval() async throws {
        let filenames = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml")
        #expect(filenames.count == 13)
    }

    @Test("filename retrieval with glob and case sensitivity")
    func filenameRetrievalWithGlob() async throws {
        let filenamesJson = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml", matching: "*.json")
        #expect(Set(filenamesJson) == Set([
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "llama-2-7b-chat.mlpackage/Manifest.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
        ]))

        let filenamesUpper = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml", matching: "*.JSON")
        #expect(filenamesUpper == [])
    }

    @Test("filename retrieval from directories")
    func filenameRetrievalFromDirectories() async throws {
        let filenames = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml", matching: "*.mlpackage/*")
        #expect(Set(filenames) == Set([
            "llama-2-7b-chat.mlpackage/Manifest.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        ]))
    }

    @Test("filename retrieval with multiple patterns")
    func filenameRetrievalWithMultiplePatterns() async throws {
        let patterns = ["config.json", "tokenizer.json", "tokenizer_*.json"]
        let filenames = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml", matching: patterns)
        #expect(Set(filenames) == Set(["config.json", "tokenizer.json", "tokenizer_config.json"]))
    }

    @Test("get file metadata (basic)")
    func getFileMetadata_basic() async throws {
        let url = URL(string: "https://huggingface.co/enterprise-explorers/Llama-2-7b-chat-coreml/resolve/main/config.json")!
        let metadata = try await Hub.getFileMetadata(fileURL: url)
        #expect(metadata.commitHash != nil)
        #expect(metadata.etag != nil)
        #expect(URL(string: metadata.location)?.path == "/api/resolve-cache/models\(url.path.replacingOccurrences(of: "resolve/main", with: metadata.commitHash!))")
        #expect(metadata.size == 163)
    }

    @Test("get Xet metadata")
    func getXetMetadata() async throws {
        let url = URL(string: "https://huggingface.co/FL33TW00D-HF/xet-test/resolve/main/tokenizer.json")!
        let metadata = try await Hub.getFileMetadata(fileURL: url)
        #expect(metadata.xetFileData != nil)
        #expect(metadata.xetFileData?.fileHash == "6aec39639a0a2d1ca966356b8c2b8426a484f80ff80731f44fa8482040713bdf")
    }

    @Test("get file metadata blob path")
    func getFileMetadata_blobPath() async throws {
        let url = URL(string: "https://huggingface.co/enterprise-explorers/Llama-2-7b-chat-coreml/resolve/main/config.json")!
        let metadata = try await Hub.getFileMetadata(fileURL: url)
        #expect(metadata.commitHash != nil)
        #expect(metadata.etag != nil && metadata.etag!.hasPrefix("d6ceb9"))
        #expect(URL(string: metadata.location)?.path == "/api/resolve-cache/models\(url.path.replacingOccurrences(of: "resolve/main", with: metadata.commitHash!))")
        #expect(metadata.size == 163)
    }

    @Test("get file metadata with revision")
    func getFileMetadata_withRevision() async throws {
        let revision = "f2c752cfc5c0ab6f4bdec59acea69eefbee381c2"
        let url = URL(string: "https://huggingface.co/julien-c/dummy-unknown/resolve/\(revision)/config.json")!
        let metadata = try await Hub.getFileMetadata(fileURL: url)
        #expect(metadata.commitHash == revision)
        #expect(metadata.etag != nil)
        #expect((metadata.etag?.count ?? 0) > 0)
        #expect(URL(string: metadata.location)?.path == "/api/resolve-cache/models\(url.path.replacingOccurrences(of: "resolve/\(revision)", with: metadata.commitHash!))")
        #expect(metadata.size == 851)
    }

    @Test("get file metadata from repo via blob search")
    func getFileMetadata_fromRepo() async throws {
        let repo = "coreml-projects/Llama-2-7b-chat-coreml"
        let metadataFromBlob = try await Hub.getFileMetadata(from: repo, matching: "*.json")
        for metadata in metadataFromBlob {
            #expect(metadata.commitHash != nil)
            #expect(metadata.etag != nil)
            #expect((metadata.etag?.count ?? 0) > 0)
            #expect((metadata.size ?? 0) > 0)
        }
    }
}

@Suite(
    "Snapshot download",
    .disabled(if: ProcessInfo.processInfo.environment["HF_TOKEN"] == "", "Set HF_TOKEN to run network tests")
)
struct SnapshotDownloadTests {
    let repo = "coreml-projects/Llama-2-7b-chat-coreml"
    let lfsRepo = "pcuenq/smol-lfs"
    let downloadDestination: URL = {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appending(component: "huggingface-tests")
    }()

    func getRelativeFiles(url: URL, repo: String) -> [String] {
        var filenames: [String] = []
        let prefix = downloadDestination.appending(path: "models/\(repo)").path.appending("/")

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
                    print("Error reading file resources: \(error)")
                }
            }
        }
        return filenames
    }

    @Test("download JSON files")
    func download() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        // Add debug prints
        print("Download destination before: \(downloadDestination.path)")

        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        // Add more debug prints
        print("Downloaded to: \(downloadedTo.path)")

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        print("Downloaded filenames: \(downloadedFilenames)")
        print("Prefix used in getRelativeFiles: \(downloadDestination.appending(path: "models/\(repo)").path)")

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 6)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        #expect(Set(downloadedFilenames) == Set([
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "llama-2-7b-chat.mlpackage/Manifest.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
        ]))
    }

    /// Background sessions get rate limited by the OS, see discussion here: https://github.com/huggingface/swift-transformers/issues/61
    /// Test only one file at a time
    @Test("download in background session")
    func downloadInBackground() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination, useBackgroundSession: true)
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }
        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 1)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        #expect(Set(downloadedFilenames) == Set([
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
        ]))
    }

    @Test("download from custom endpoint")
    func customEndpointDownload() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination, endpoint: "https://hf-mirror.com")
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }
        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 6)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        #expect(Set(downloadedFilenames) == Set([
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "llama-2-7b-chat.mlpackage/Manifest.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
        ]))
    }

    @Test("download file metadata for multiple files")
    func downloadFileMetadata() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }
        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 6)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        #expect(Set(downloadedFilenames) == Set([
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "llama-2-7b-chat.mlpackage/Manifest.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
        ]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")
        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        #expect(Set(downloadedMetadataFilenames) == Set([
            ".cache/huggingface/download/config.json.metadata",
            ".cache/huggingface/download/tokenizer.json.metadata",
            ".cache/huggingface/download/tokenizer_config.json.metadata",
            ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Manifest.json.metadata",
            ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json.metadata",
            ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json.metadata",
        ]))
    }

    @Test("skip redownload when metadata exists")
    func downloadFileMetadataExists() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 6)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        #expect(Set(downloadedFilenames) == Set([
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "llama-2-7b-chat.mlpackage/Manifest.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
        ]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let configPath = downloadedTo.appending(path: "config.json")
        var attributes = try FileManager.default.attributesOfItem(atPath: configPath.path)
        let originalTimestamp = attributes[.modificationDate] as! Date

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        #expect(Set(downloadedMetadataFilenames) == Set([
            ".cache/huggingface/download/config.json.metadata",
            ".cache/huggingface/download/tokenizer.json.metadata",
            ".cache/huggingface/download/tokenizer_config.json.metadata",
            ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Manifest.json.metadata",
            ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json.metadata",
            ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json.metadata",
        ]))

        _ = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: configPath.path)
        let secondDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will not be downloaded again thus last modified date will remain unchanged
        #expect(originalTimestamp == secondDownloadTimestamp)
    }

    @Test("metadata remains same when file unchanged")
    func downloadFileMetadataSame() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "tokenizer.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 1)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        #expect(Set(downloadedFilenames) == Set(["tokenizer.json"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let metadataPath = metadataDestination.appending(path: "tokenizer.json.metadata")

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        #expect(Set(downloadedMetadataFilenames) == Set([
            ".cache/huggingface/download/tokenizer.json.metadata",
        ]))

        let originalMetadata = try String(contentsOf: metadataPath, encoding: .utf8)

        _ = try await hubApi.snapshot(from: repo, matching: "tokenizer.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        let secondDownloadMetadata = try String(contentsOf: metadataPath, encoding: .utf8)

        // File hasn't changed so commit hash and etag will be identical
        let originalArr = originalMetadata.components(separatedBy: .newlines)
        let secondDownloadArr = secondDownloadMetadata.components(separatedBy: .newlines)

        #expect(originalArr[0] == secondDownloadArr[0])
        #expect(originalArr[1] == secondDownloadArr[1])
    }

    @Test("redownload on corrupted metadata for non-LFS")
    func downloadFileMetadataCorrupted() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 6)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        #expect(Set(downloadedFilenames) == Set([
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "llama-2-7b-chat.mlpackage/Manifest.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
        ]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let configPath = downloadedTo.appending(path: "config.json")
        var attributes = try FileManager.default.attributesOfItem(atPath: configPath.path)
        let originalTimestamp = attributes[.modificationDate] as! Date

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        #expect(Set(downloadedMetadataFilenames) == Set([
            ".cache/huggingface/download/config.json.metadata",
            ".cache/huggingface/download/tokenizer.json.metadata",
            ".cache/huggingface/download/tokenizer_config.json.metadata",
            ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Manifest.json.metadata",
            ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json.metadata",
            ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json.metadata",
        ]))

        // Corrupt config.json.metadata
        print("Testing corrupted file.")
        try "a".write(to: metadataDestination.appendingPathComponent("config.json.metadata"), atomically: true, encoding: .utf8)

        _ = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: configPath.path)
        let secondDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will be downloaded again thus last modified date will change
        #expect(originalTimestamp != secondDownloadTimestamp)

        // Corrupt config.metadata again
        print("Testing corrupted timestamp.")
        try "a\nb\nc\n".write(to: metadataDestination.appendingPathComponent("config.json.metadata"), atomically: true, encoding: .utf8)

        _ = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: configPath.path)
        let thirdDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will be downloaded again thus last modified date will change
        #expect(originalTimestamp != thirdDownloadTimestamp)
    }

    @Test("update metadata without redownload for LFS when etag matches")
    func downloadLargeFileMetadataCorrupted() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.mlmodel") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 1)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        #expect(Set(downloadedFilenames) == Set(["llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let modelPath = downloadedTo.appending(path: "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel")
        var attributes = try FileManager.default.attributesOfItem(atPath: modelPath.path)
        let originalTimestamp = attributes[.modificationDate] as! Date

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        #expect(Set(downloadedMetadataFilenames) == Set([
            ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel.metadata",
        ]))

        // Corrupt model.metadata etag
        print("Testing corrupted etag.")
        let corruptedMetadataString = "a\nfc329090bfbb2570382c9af997cffd5f4b78b39b8aeca62076db69534e020108\n0\n"
        let metadataFile = metadataDestination.appendingPathComponent("llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel.metadata")
        try corruptedMetadataString.write(to: metadataFile, atomically: true, encoding: .utf8)

        _ = try await hubApi.snapshot(from: repo, matching: "*.mlmodel") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: modelPath.path)
        let thirdDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will not be downloaded again because this is an LFS file.
        // While downloading LFS files, we first check if local file ETag is the same as remote ETag.
        // If that's the case we just update the metadata and keep the local file.
        #expect(originalTimestamp == thirdDownloadTimestamp)

        let metadataString = try String(contentsOfFile: metadataFile.path)

        // Updated metadata file needs to have the correct commit hash, etag and timestamp.
        // This is being updated because the local etag (SHA256 checksum) matches the remote etag
        #expect(metadataString != corruptedMetadataString)
    }

    @Test("download large LFS file and metadata")
    func downloadLargeFile() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.mlmodel") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 1)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        #expect(Set(downloadedFilenames) == Set(["llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")
        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        #expect(Set(downloadedMetadataFilenames) == Set([".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel.metadata"]))

        let metadataFile = metadataDestination.appendingPathComponent("llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel.metadata")
        let metadataString = try String(contentsOfFile: metadataFile.path)

        let expected = "eaf97358a37d03fd48e5a87d15aff2e8423c1afb\nfc329090bfbb2570382c9af997cffd5f4b78b39b8aeca62076db69534e020107"
        #expect(metadataString.contains(expected))
    }

    @Test("download small LFS file and metadata")
    func downloadSmolLargeFile() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 1)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(lfsRepo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: lfsRepo)
        #expect(Set(downloadedFilenames) == Set(["x.bin"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")
        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: lfsRepo)
        #expect(Set(downloadedMetadataFilenames) == Set([".cache/huggingface/download/x.bin.metadata"]))

        let metadataFile = metadataDestination.appendingPathComponent("x.bin.metadata")
        let metadataString = try String(contentsOfFile: metadataFile.path)

        let expected = "77b984598d90af6143d73d5a2d6214b23eba7e27\n98ea6e4f216f2fb4b69fff9b3a44842c38686ca685f3f55dc48c5d3fb1107be4"
        #expect(metadataString.contains(expected))
    }

    @Test("validate regex patterns for commit and sha256")
    func regexValidation() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 1)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(lfsRepo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: lfsRepo)
        #expect(Set(downloadedFilenames) == Set(["x.bin"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")
        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: lfsRepo)
        #expect(Set(downloadedMetadataFilenames) == Set([".cache/huggingface/download/x.bin.metadata"]))

        let metadataFile = metadataDestination.appendingPathComponent("x.bin.metadata")
        let metadataString = try String(contentsOfFile: metadataFile.path)
        let metadataArr = metadataString.components(separatedBy: .newlines)

        let commitHash = metadataArr[0]
        let etag = metadataArr[1]

        #expect(hubApi.isValidHash(hash: commitHash, pattern: hubApi.commitHashPattern))
        #expect(hubApi.isValidHash(hash: etag, pattern: hubApi.sha256Pattern))

        #expect(!hubApi.isValidHash(hash: "\(commitHash)a", pattern: hubApi.commitHashPattern))
        #expect(!hubApi.isValidHash(hash: "\(etag)a", pattern: hubApi.sha256Pattern))
    }

    @Test("recreate missing metadata for LFS file without redownload")
    func lFSFileNoMetadata() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 1)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(lfsRepo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: lfsRepo)
        #expect(Set(downloadedFilenames) == Set(["x.bin"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let filePath = downloadedTo.appending(path: "x.bin")
        var attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
        let originalTimestamp = attributes[.modificationDate] as! Date

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: lfsRepo)
        #expect(Set(downloadedMetadataFilenames) == Set([".cache/huggingface/download/x.bin.metadata"]))

        let metadataFile = metadataDestination.appendingPathComponent("x.bin.metadata")
        try FileManager.default.removeItem(atPath: metadataFile.path)

        _ = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
        let secondDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will not be downloaded again thus last modified date will remain unchanged
        #expect(originalTimestamp == secondDownloadTimestamp)
        #expect(FileManager.default.fileExists(atPath: metadataDestination.path))

        let metadataString = try String(contentsOfFile: metadataFile.path)
        let expected = "77b984598d90af6143d73d5a2d6214b23eba7e27\n98ea6e4f216f2fb4b69fff9b3a44842c38686ca685f3f55dc48c5d3fb1107be4"

        #expect(metadataString.contains(expected))
    }

    @Test("repair corrupted LFS metadata without redownload")
    func lFSFileCorruptedMetadata() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 1)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(lfsRepo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: lfsRepo)
        #expect(Set(downloadedFilenames) == Set(["x.bin"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let filePath = downloadedTo.appending(path: "x.bin")
        var attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
        let originalTimestamp = attributes[.modificationDate] as! Date

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: lfsRepo)
        #expect(Set(downloadedMetadataFilenames) == Set([".cache/huggingface/download/x.bin.metadata"]))

        let metadataFile = metadataDestination.appendingPathComponent("x.bin.metadata")
        try "a".write(to: metadataFile, atomically: true, encoding: .utf8)

        _ = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
        let secondDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will not be downloaded again thus last modified date will remain unchanged
        #expect(originalTimestamp == secondDownloadTimestamp)
        #expect(FileManager.default.fileExists(atPath: metadataDestination.path))

        let metadataString = try String(contentsOfFile: metadataFile.path)
        let expected = "77b984598d90af6143d73d5a2d6214b23eba7e27\n98ea6e4f216f2fb4b69fff9b3a44842c38686ca685f3f55dc48c5d3fb1107be4"

        #expect(metadataString.contains(expected))
    }

    @Test("redownload non-LFS when metadata missing")
    func nonLFSFileRedownload() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "config.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 1)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        #expect(Set(downloadedFilenames) == Set(["config.json"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let filePath = downloadedTo.appending(path: "config.json")
        var attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
        let originalTimestamp = attributes[.modificationDate] as! Date

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        #expect(Set(downloadedMetadataFilenames) == Set([".cache/huggingface/download/config.json.metadata"]))

        let metadataFile = metadataDestination.appendingPathComponent("config.json.metadata")
        try FileManager.default.removeItem(atPath: metadataFile.path)

        _ = try await hubApi.snapshot(from: repo, matching: "config.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
        let secondDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will be downloaded again thus last modified date will change
        #expect(originalTimestamp != secondDownloadTimestamp)
        #expect(FileManager.default.fileExists(atPath: metadataDestination.path))

        let metadataString = try String(contentsOfFile: metadataFile.path)
        let expected = "eaf97358a37d03fd48e5a87d15aff2e8423c1afb\nd6ceb92ce9e3c83ab146dc8e92a93517ac1cc66f"

        #expect(metadataString.contains(expected))
    }

    @Test("offline mode returns destination when files present")
    func offlineModeReturnsDestination() async throws {
        var hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        var downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")

            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 6)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        hubApi = HubApi(downloadBase: downloadDestination, useOfflineMode: true)

        downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 6)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))
    }

    @Test("offline mode throws without local repo")
    func offlineModeThrowsError() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination, useOfflineMode: true)

        do {
            try await hubApi.snapshot(from: repo, matching: "*.json")
            Issue.record("Expected an error to be thrown")
        } catch let error as HubApi.EnvironmentError {
            switch error {
            case let .offlineModeError(message):
                #expect(message == "Repository not available locally")
            default:
                Issue.record("Wrong error type: \(error)")
            }
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("offline mode errors when LFS metadata missing")
    func offlineModeWithoutMetadata() async throws {
        var hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "*") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")

            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 2)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(lfsRepo)"))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let metadataFile = metadataDestination.appendingPathComponent("x.bin.metadata")
        try FileManager.default.removeItem(atPath: metadataFile.path)

        hubApi = HubApi(downloadBase: downloadDestination, useOfflineMode: true)

        do {
            try await hubApi.snapshot(from: lfsRepo, matching: "*")
            Issue.record("Expected an error to be thrown")
        } catch let error as HubApi.EnvironmentError {
            switch error {
            case let .offlineModeError(message):
                #expect(message == "Metadata not available for x.bin")
            default:
                Issue.record("Wrong error type: \(error)")
            }
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("offline mode detects corrupted LFS metadata")
    func offlineModeWithCorruptedLFSMetadata() async throws {
        var hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "*") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")

            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 2)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(lfsRepo)"))

        let metadataDestination = downloadedTo.appendingPathComponent(".cache/huggingface/download").appendingPathComponent("x.bin.metadata")

        try "77b984598d90af6143d73d5a2d6214b23eba7e27\n98ea6e4f216f2ab4b69fff9b3a44842c38686ca685f3f55dc48c5d3fb1107be4\n0\n".write(
            to: metadataDestination,
            atomically: true,
            encoding: .utf8
        )

        hubApi = HubApi(downloadBase: downloadDestination, useOfflineMode: true)

        do {
            try await hubApi.snapshot(from: lfsRepo, matching: "*")
            Issue.record("Expected an error to be thrown")
        } catch let error as HubApi.EnvironmentError {
            switch error {
            case let .fileIntegrityError(message):
                #expect(message == "Hash mismatch for x.bin")
            default:
                Issue.record("Wrong error type: \(error)")
            }
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("offline mode errors when no files available")
    func offlineModeWithNoFiles() async throws {
        var hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")

            lastProgress = progress
        }

        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 1)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(lfsRepo)"))

        let fileDestination = downloadedTo.appendingPathComponent("x.bin")
        try FileManager.default.removeItem(at: fileDestination)

        hubApi = HubApi(downloadBase: downloadDestination, useOfflineMode: true)

        do {
            try await hubApi.snapshot(from: lfsRepo, matching: "x.bin")
            Issue.record("Expected an error to be thrown")
        } catch let error as HubApi.EnvironmentError {
            switch error {
            case let .offlineModeError(message):
                #expect(message == "No files available locally for this repository")
            default:
                Issue.record("Wrong error type: \(error)")
            }
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("resume download from empty .incomplete file")
    func resumeDownloadFromEmptyIncomplete() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        var downloadedTo = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
            "Library/Caches/huggingface-tests/models/coreml-projects/Llama-2-7b-chat-coreml"
        )

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let url = URL(string: "https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/config.json")!
        let etag = try await Hub.getFileMetadata(fileURL: url).etag!

        try FileManager.default.createDirectory(at: metadataDestination, withIntermediateDirectories: true, attributes: nil)
        try "".write(to: metadataDestination.appendingPathComponent("config.json.\(etag).incomplete"), atomically: true, encoding: .utf8)
        downloadedTo = try await hubApi.snapshot(from: repo, matching: "config.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }
        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 1)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        let fileContents = try String(contentsOfFile: downloadedTo.appendingPathComponent("config.json").path)

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
        #expect(fileContents.contains(expected))
    }

    @Test("resume download from non-empty .incomplete file")
    func resumeDownloadFromNonEmptyIncomplete() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        var downloadedTo = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/huggingface-tests/models/coreml-projects/Llama-2-7b-chat-coreml")

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let url = URL(string: "https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/config.json")!
        let etag = try await Hub.getFileMetadata(fileURL: url).etag!

        try FileManager.default.createDirectory(at: metadataDestination, withIntermediateDirectories: true, attributes: nil)
        try "X".write(to: metadataDestination.appendingPathComponent("config.json.\(etag).incomplete"), atomically: true, encoding: .utf8)
        downloadedTo = try await hubApi.snapshot(from: repo, matching: "config.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }
        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 1)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        let fileContents = try String(contentsOfFile: downloadedTo.appendingPathComponent("config.json").path)

        let expected = """
        X
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
        #expect(fileContents.contains(expected))
    }

    @Test("real download interruption and resumption")
    func realDownloadInterruptionAndResumption() async throws {
        // Use the DepthPro model weights file
        let targetFile = "SAM 2 Studio 1.1.zip"
        let repo = "coreml-projects/sam-2-studio"
        let hubApi = HubApi(downloadBase: downloadDestination)

        // Start and cancel shortly after to simulate interruption
        let downloadTask = Task {
            try await hubApi.snapshot(from: repo, matching: targetFile) { progress in
                print("Progress reached 1 \(progress.fractionCompleted * 100)%")
            }
        }
        try await Task.sleep(nanoseconds: 5_000_000_000)
        downloadTask.cancel()
        try await Task.sleep(nanoseconds: 1_000_000_000)

        // Resume download with a new task
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: targetFile) { progress in
            print("Progress reached 2 \(progress.fractionCompleted * 100)%")
        }

        let filePath = downloadedTo.appendingPathComponent(targetFile)
        #expect(
            FileManager.default.fileExists(atPath: filePath.path),
            "Downloaded file should exist at \(filePath.path)"
        )
    }

    @Test("real download with speed")
    func realDownloadWithSpeed() async throws {
        // Use the DepthPro model weights file
        let targetFile = "SAM 2 Studio 1.1.zip"
        let repo = "coreml-projects/sam-2-studio"
        let hubApi = HubApi(downloadBase: downloadDestination)

        var lastSpeed: Double? = nil

        // Add debug prints
        print("Download destination before: \(downloadDestination.path)")

        let downloadedTo = try await hubApi.snapshot(from: repo, matching: targetFile) { progress, speed in
            if let speed {
                print("Current speed: \(speed)")
            }

            lastSpeed = speed
        }

        // Add more debug prints
        print("Downloaded to: \(downloadedTo.path)")

        #expect(lastSpeed != nil)

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        print("Downloaded filenames: \(downloadedFilenames)")
        print("Prefix used in getRelativeFiles: \(downloadDestination.appending(path: "models/\(repo)").path)")

        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))

        let filePath = downloadedTo.appendingPathComponent(targetFile)
        #expect(
            FileManager.default.fileExists(atPath: filePath.path),
            "Downloaded file should exist at \(filePath.path)"
        )
    }

    @Test("download with revision")
    func downloadWithRevision() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let commitHash = "eaf97358a37d03fd48e5a87d15aff2e8423c1afb"
        let downloadedTo = try await hubApi.snapshot(from: repo, revision: commitHash, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        #expect(lastProgress?.fractionCompleted == 1)
        #expect(lastProgress?.completedUnitCount == 6)
        #expect(downloadedTo == downloadDestination.appending(path: "models/\(repo)"))
        #expect(Set(downloadedFilenames) == Set([
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "llama-2-7b-chat.mlpackage/Manifest.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
            "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
        ]))

        do {
            let revision = "nonexistent-revision"
            try await hubApi.snapshot(from: repo, revision: revision, matching: "*.json")
            Issue.record("Expected an error to be thrown")
        } catch let error as Hub.HubClientError {
            switch error {
            case .fileNotFound:
                break // Error type is correct
            default:
                Issue.record("Wrong error type: \(error)")
            }
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}
