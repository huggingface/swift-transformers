//
//  HubApi.swift
//
//
//  Created by Pedro Cuenca on 20231230.
//

import Foundation
import CryptoKit
import os

public struct HubApi {
    var downloadBase: URL
    var hfToken: String?
    var endpoint: String
    var useBackgroundSession: Bool

    public typealias RepoType = Hub.RepoType
    public typealias Repo = Hub.Repo
    
    public init(downloadBase: URL? = nil, hfToken: String? = nil, endpoint: String = "https://huggingface.co", useBackgroundSession: Bool = false) {
        self.hfToken = hfToken ?? Self.hfTokenFromEnv()
        if let downloadBase {
            self.downloadBase = downloadBase
        } else {
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            self.downloadBase = documents.appending(component: "huggingface")
        }
        self.endpoint = endpoint
        self.useBackgroundSession = useBackgroundSession
    }
    
    public static let shared = HubApi()
    
    private static let logger = Logger()
}

private extension HubApi {
    static func hfTokenFromEnv() -> String? {
        let possibleTokens = [
            { ProcessInfo.processInfo.environment["HF_TOKEN"] },
            { ProcessInfo.processInfo.environment["HUGGING_FACE_HUB_TOKEN"] },
            {
                ProcessInfo.processInfo.environment["HF_TOKEN_PATH"].flatMap {
                    try? String(
                        contentsOf: URL(filePath: NSString(string: $0).expandingTildeInPath),
                        encoding: .utf8
                    )
                }
            },
            {
                ProcessInfo.processInfo.environment["HF_HOME"].flatMap {
                    try? String(
                        contentsOf: URL(filePath: NSString(string: $0).expandingTildeInPath).appending(path: "token"),
                        encoding: .utf8
                    )
                }
            },
            { try? String(contentsOf: .homeDirectory.appendingPathComponent(".cache/huggingface/token"), encoding: .utf8) },
            { try? String(contentsOf: .homeDirectory.appendingPathComponent(".huggingface/token"), encoding: .utf8) }
        ]
        return possibleTokens
            .lazy
            .compactMap({ $0() })
            .filter({ !$0.isEmpty })
            .first
    }
}

/// File retrieval
public extension HubApi {
    /// Model data for parsed filenames
    struct Sibling: Codable {
        let rfilename: String
    }
    
    struct SiblingsResponse: Codable {
        let siblings: [Sibling]
    }
        
    /// Throws error if the response code is not 20X
    func httpGet(for url: URL) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url)
        if let hfToken = hfToken {
            request.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let response = response as? HTTPURLResponse else { throw Hub.HubClientError.unexpectedError }
        
        switch response.statusCode {
        case 200..<300: break
        case 400..<500: throw Hub.HubClientError.authorizationRequired
        default: throw Hub.HubClientError.httpStatusCode(response.statusCode)
        }

        return (data, response)
    }
    
    /// Throws error if page does not exist or is not accessible.
    /// Allows relative redirects but ignores absolute ones for LFS files.
    func httpHead(for url: URL) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        if let hfToken = hfToken {
            request.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization")
        }
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        
        let redirectDelegate = RedirectDelegate()
        let session = URLSession(configuration: .default, delegate: redirectDelegate, delegateQueue: nil)
        
        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else { throw Hub.HubClientError.unexpectedError }

        switch response.statusCode {
        case 200..<400: break // Allow redirects to pass through to the redirect delegate
        case 400..<500: throw Hub.HubClientError.authorizationRequired
        default: throw Hub.HubClientError.httpStatusCode(response.statusCode)
        }
                
        return (data, response)
    }
    
    func getFilenames(from repo: Repo, matching globs: [String] = []) async throws -> [String] {
        // Read repo info and only parse "siblings"
        let url = URL(string: "\(endpoint)/api/\(repo.type)/\(repo.id)")!
        let (data, _) = try await httpGet(for: url)
        let response = try JSONDecoder().decode(SiblingsResponse.self, from: data)
        let filenames = response.siblings.map { $0.rfilename }
        guard globs.count > 0 else { return filenames }
        
        var selected: Set<String> = []
        for glob in globs {
            selected = selected.union(filenames.matching(glob: glob))
        }
        return Array(selected)
    }
    
    func getFilenames(from repoId: String, matching globs: [String] = []) async throws -> [String] {
        return try await getFilenames(from: Repo(id: repoId), matching: globs)
    }
    
    func getFilenames(from repo: Repo, matching glob: String) async throws -> [String] {
        return try await getFilenames(from: repo, matching: [glob])
    }
    
    func getFilenames(from repoId: String, matching glob: String) async throws -> [String] {
        return try await getFilenames(from: Repo(id: repoId), matching: [glob])
    }
}

/// Additional Errors
public extension HubApi {
    enum EnvironmentError: LocalizedError {
        case invalidMetadataError(String)
        
        public var errorDescription: String? {
            switch self {
            case .invalidMetadataError(let message):
                return message
            }
        }
    }
}

/// Configuration loading helpers
public extension HubApi {
    /// Assumes the file has already been downloaded.
    /// `filename` is relative to the download base.
    func configuration(from filename: String, in repo: Repo) throws -> Config {
        let fileURL = localRepoLocation(repo).appending(path: filename)
        return try configuration(fileURL: fileURL)
    }
    
    /// Assumes the file is already present at local url.
    /// `fileURL` is a complete local file path for the given model
    func configuration(fileURL: URL) throws -> Config {
        let data = try Data(contentsOf: fileURL)
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [NSString: Any] else { throw Hub.HubClientError.parse }
        return Config(dictionary)
    }
}

/// Whoami
public extension HubApi {
    func whoami() async throws -> Config {
        guard hfToken != nil else { throw Hub.HubClientError.authorizationRequired }
        
        let url = URL(string: "\(endpoint)/api/whoami-v2")!
        let (data, _) = try await httpGet(for: url)

        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [NSString: Any] else { throw Hub.HubClientError.parse }
        return Config(dictionary)
    }
}

/// Snaphsot download
public extension HubApi {
    func localRepoLocation(_ repo: Repo) -> URL {
        downloadBase.appending(component: repo.type.rawValue).appending(component: repo.id)
    }
    
    struct HubFileDownloader {
        let repo: Repo
        let repoDestination: URL
        let relativeFilename: String
        let hfToken: String?
        let endpoint: String?
        let backgroundSession: Bool
        
        let sha256Pattern = "^[0-9a-f]{64}$"
        let commitHashPattern = "^[0-9a-f]{40}$"

        var source: URL {
            // https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/tokenizer.json?download=true
            var url = URL(string: endpoint ?? "https://huggingface.co")!
            if repo.type != .models {
                url = url.appending(component: repo.type.rawValue)
            }
            url = url.appending(path: repo.id)
            url = url.appending(path: "resolve/main") // TODO: revisions
            url = url.appending(path: relativeFilename)
            return url
        }
        
        var destination: URL {
            repoDestination.appending(path: relativeFilename)
        }
        
        var metadataDestination: URL {
            repoDestination
                .appendingPathComponent(".cache")
                .appendingPathComponent("huggingface")
                .appendingPathComponent("download")
        }
        
        var downloaded: Bool {
            FileManager.default.fileExists(atPath: destination.path)
        }
        
        func prepareDestination() throws {
            let directoryURL = destination.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil)
        }
        
        func prepareMetadataDestination() throws {
            try FileManager.default.createDirectory(at: metadataDestination, withIntermediateDirectories: true, attributes: nil)
        }
        
        /// Reads metadata about a file in the local directory related to a download process.
        ///
        /// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/_local_folder.py#L263
        ///
        /// - Parameters:
        ///   - localDir: The local directory where metadata files are downloaded.
        ///   - filePath: The path of the file for which metadata is being read.
        /// - Throws: An `EnvironmentError.invalidMetadataError` if the metadata file is invalid and cannot be removed.
        /// - Returns: A `LocalDownloadFileMetadata` object if the metadata file exists and is valid, or `nil` if the file is missing or invalid.
        func readDownloadMetadata(localDir: URL, filePath: String) throws -> LocalDownloadFileMetadata? {
            let metadataPath = localDir.appending(path: filePath)
            if FileManager.default.fileExists(atPath: metadataPath.path) {
                do {
                    let contents = try String(contentsOf: metadataPath, encoding: .utf8)
                    let lines = contents.components(separatedBy: .newlines)
                    
                    guard lines.count >= 3 else {
                        throw EnvironmentError.invalidMetadataError("Metadata file is missing required fields.")
                    }
                    
                    let commitHash = lines[0].trimmingCharacters(in: .whitespacesAndNewlines)
                    let etag = lines[1].trimmingCharacters(in: .whitespacesAndNewlines)
                    guard let timestamp = Double(lines[2].trimmingCharacters(in: .whitespacesAndNewlines)) else {
                        throw EnvironmentError.invalidMetadataError("Missing or invalid timestamp.")
                    }
                    let timestampDate = Date(timeIntervalSince1970: timestamp)
                            
                    // TODO: check if file hasn't been modified since the metadata was saved
                    // Reference: https://github.com/huggingface/huggingface_hub/blob/2fdc6f48ef5e6b22ee9bcdc1945948ac070da675/src/huggingface_hub/_local_folder.py#L303
                    
                    return LocalDownloadFileMetadata(commitHash: commitHash, etag: etag, filename: filePath, timestamp: timestampDate)
                } catch {
                    do {
                        logger.warning("Invalid metadata file \(metadataPath): \(error). Removing it from disk and continue.")
                        try FileManager.default.removeItem(at: metadataPath)
                    } catch {
                        throw EnvironmentError.invalidMetadataError("Could not remove corrupted metadata file \(metadataPath): \(error)")
                    }
                    return nil
                }
            }
                
            // metadata file does not exist
            return nil
        }
        
        func isValidHash(hash: String, pattern: String) -> Bool {
            let regex = try? NSRegularExpression(pattern: pattern)
            let range = NSRange(location: 0, length: hash.utf16.count)
            return regex?.firstMatch(in: hash, options: [], range: range) != nil
        }
        
        /// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/_local_folder.py#L391
        func writeDownloadMetadata(commitHash: String, etag: String, metadataRelativePath: String) throws {
            let metadataContent = "\(commitHash)\n\(etag)\n\(Date().timeIntervalSince1970)\n"
            let metadataPath = metadataDestination.appending(component: metadataRelativePath)
            
            do {
                try FileManager.default.createDirectory(at: metadataPath.deletingLastPathComponent(), withIntermediateDirectories: true)
                try metadataContent.write(to: metadataPath, atomically: true, encoding: .utf8)
            } catch {
                throw EnvironmentError.invalidMetadataError("Failed to write metadata file \(metadataPath)")
            }
        }
        
        func computeFileHash(file url: URL) throws -> String {
            // Open file for reading
            guard let fileHandle = try? FileHandle(forReadingFrom: url) else {
                throw Hub.HubClientError.unexpectedError
            }
            
            defer {
                try? fileHandle.close()
            }
            
            var hasher = SHA256()
            let chunkSize = 1024 * 1024 // 1MB chunks
            
            while autoreleasepool(invoking: {
                let nextChunk = try? fileHandle.read(upToCount: chunkSize)
                
                guard let nextChunk,
                        !nextChunk.isEmpty
                else {
                    return false
                }
                
                hasher.update(data: nextChunk)
                
                return true
            }) { }
            
            let digest = hasher.finalize()
            return digest.map { String(format: "%02x", $0) }.joined()
        }
        
        
        // Note we go from Combine in Downloader to callback-based progress reporting
        // We'll probably need to support Combine as well to play well with Swift UI
        // (See for example PipelineLoader in swift-coreml-diffusers)
        @discardableResult
        func download(progressHandler: @escaping (Double) -> Void) async throws -> URL {
            let metadataRelativePath = "\(relativeFilename).metadata"
                        
            let localMetadata = try readDownloadMetadata(localDir: metadataDestination, filePath: metadataRelativePath)
            let remoteMetadata = try await HubApi.shared.getFileMetadata(url: source)
            
            let localCommitHash = localMetadata?.commitHash ?? ""
            let remoteCommitHash = remoteMetadata.commitHash ?? ""
            
            // Local file exists + metadata exists + commit_hash matches => return file
            if isValidHash(hash: remoteCommitHash, pattern: commitHashPattern) && downloaded && localMetadata != nil && localCommitHash == remoteCommitHash {
                return destination
            }
            
            // From now on, etag, commit_hash, url and size are not empty
            guard let remoteCommitHash = remoteMetadata.commitHash,
                  let remoteEtag = remoteMetadata.etag,
                  let remoteSize = remoteMetadata.size,
                  remoteMetadata.location != "" else {
                throw EnvironmentError.invalidMetadataError("File metadata must have been retrieved from server")
            }
            
            // Local file exists => check if it's up-to-date
            if downloaded {
                // etag matches => update metadata and return file
                if localMetadata?.etag == remoteEtag {
                    try writeDownloadMetadata(commitHash: remoteCommitHash, etag: remoteEtag, metadataRelativePath: metadataRelativePath)
                    return destination
                }
                
                // etag is a sha256
                // => means it's an LFS file (large)
                // => let's compute local hash and compare
                // => if match, update metadata and return file
                if isValidHash(hash: remoteEtag, pattern: sha256Pattern) {
                    let fileHash = try computeFileHash(file: destination)
                    if fileHash == remoteEtag {
                        try writeDownloadMetadata(commitHash: remoteCommitHash, etag: remoteEtag, metadataRelativePath: metadataRelativePath)
                        return destination
                    }
                }
            }
            
            // Otherwise, let's download the file!
            try prepareDestination()
            try prepareMetadataDestination()

            let downloader = Downloader(from: source, to: destination, using: hfToken, inBackground: backgroundSession, expectedSize: remoteSize)
            let downloadSubscriber = downloader.downloadState.sink { state in
                if case .downloading(let progress) = state {
                    progressHandler(progress)
                }
            }
            _ = try withExtendedLifetime(downloadSubscriber) {
                try downloader.waitUntilDone()
            }
            
            try writeDownloadMetadata(commitHash: remoteCommitHash, etag: remoteEtag, metadataRelativePath: metadataRelativePath)
            
            return destination
        }
    }

    @discardableResult
    func snapshot(from repo: Repo, matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        let filenames = try await getFilenames(from: repo, matching: globs)
        let progress = Progress(totalUnitCount: Int64(filenames.count))
        let repoDestination = localRepoLocation(repo)
        for filename in filenames {
            let fileProgress = Progress(totalUnitCount: 100, parent: progress, pendingUnitCount: 1)
            let downloader = HubFileDownloader(
                repo: repo,
                repoDestination: repoDestination,
                relativeFilename: filename,
                hfToken: hfToken,
                endpoint: endpoint,
                backgroundSession: useBackgroundSession
            )
            try await downloader.download { fractionDownloaded in
                fileProgress.completedUnitCount = Int64(100 * fractionDownloaded)
                progressHandler(progress)
            }
            fileProgress.completedUnitCount = 100
        }
        progressHandler(progress)
        return repoDestination
    }
    
    @discardableResult
    func snapshot(from repoId: String, matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        return try await snapshot(from: Repo(id: repoId), matching: globs, progressHandler: progressHandler)
    }
    
    @discardableResult
    func snapshot(from repo: Repo, matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        return try await snapshot(from: repo, matching: [glob], progressHandler: progressHandler)
    }
    
    @discardableResult
    func snapshot(from repoId: String, matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        return try await snapshot(from: Repo(id: repoId), matching: [glob], progressHandler: progressHandler)
    }
}

/// Metadata
public extension HubApi {
    /// Data structure containing information about a file versioned on the Hub
    struct FileMetadata {
        /// The commit hash related to the file
        public let commitHash: String?
        
        /// Etag of the file on the server
        public let etag: String?
        
        /// Location where to download the file. Can be a Hub url or not (CDN).
        public let location: String
        
        /// Size of the file. In case of an LFS file, contains the size of the actual LFS file, not the pointer.
        public let size: Int?
    }
    
    /// Metadata about a file in the local directory related to a download process
    struct LocalDownloadFileMetadata {
        /// Commit hash of the file in the repo
        public let commitHash: String
        
        /// ETag of the file in the repo. Used to check if the file has changed.
        /// For LFS files, this is the sha256 of the file. For regular files, it corresponds to the git hash.
        public let etag: String
        
        /// Path of the file in the repo
        public let filename: String
        
        /// The timestamp of when the metadata was saved i.e. when the metadata was accurate
        public let timestamp: Date
    }

    private func normalizeEtag(_ etag: String?) -> String? {
        guard let etag = etag else { return nil }
        return etag.trimmingPrefix("W/").trimmingCharacters(in: CharacterSet(charactersIn: "\""))
    }
    
    func getFileMetadata(url: URL) async throws -> FileMetadata {
        let (_, response) = try await httpHead(for: url)
        let location = response.statusCode == 302 ? response.value(forHTTPHeaderField: "Location") : response.url?.absoluteString
        
        return FileMetadata(
            commitHash: response.value(forHTTPHeaderField: "X-Repo-Commit"),
            etag: normalizeEtag(
                (response.value(forHTTPHeaderField: "X-Linked-Etag")) ?? (response.value(forHTTPHeaderField: "Etag"))
            ),
            location: location ?? url.absoluteString,
            size: Int(response.value(forHTTPHeaderField: "X-Linked-Size") ?? response.value(forHTTPHeaderField: "Content-Length") ?? "")
        )
    }
    
    func getFileMetadata(from repo: Repo, matching globs: [String] = []) async throws -> [FileMetadata] {
        let files = try await getFilenames(from: repo, matching: globs)
        let url = URL(string: "\(endpoint)/\(repo.id)/resolve/main")! // TODO: revisions
        var selectedMetadata: Array<FileMetadata> = []
        for file in files {
            let fileURL = url.appending(path: file)
            selectedMetadata.append(try await getFileMetadata(url: fileURL))
        }
        return selectedMetadata
    }
    
    func getFileMetadata(from repoId: String, matching globs: [String] = []) async throws -> [FileMetadata] {
        return try await getFileMetadata(from: Repo(id: repoId), matching: globs)
    }
    
    func getFileMetadata(from repo: Repo, matching glob: String) async throws -> [FileMetadata] {
        return try await getFileMetadata(from: repo, matching: [glob])
    }
    
    func getFileMetadata(from repoId: String, matching glob: String) async throws -> [FileMetadata] {
        return try await getFileMetadata(from: Repo(id: repoId), matching: [glob])
    }
}

/// Stateless wrappers that use `HubApi` instances
public extension Hub {
    static func getFilenames(from repo: Hub.Repo, matching globs: [String] = []) async throws -> [String] {
        return try await HubApi.shared.getFilenames(from: repo, matching: globs)
    }
    
    static func getFilenames(from repoId: String, matching globs: [String] = []) async throws -> [String] {
        return try await HubApi.shared.getFilenames(from: Repo(id: repoId), matching: globs)
    }
    
    static func getFilenames(from repo: Repo, matching glob: String) async throws -> [String] {
        return try await HubApi.shared.getFilenames(from: repo, matching: glob)
    }
    
    static func getFilenames(from repoId: String, matching glob: String) async throws -> [String] {
        return try await HubApi.shared.getFilenames(from: Repo(id: repoId), matching: glob)
    }
    
    static func snapshot(from repo: Repo, matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        return try await HubApi.shared.snapshot(from: repo, matching: globs, progressHandler: progressHandler)
    }
    
    static func snapshot(from repoId: String, matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        return try await HubApi.shared.snapshot(from: Repo(id: repoId), matching: globs, progressHandler: progressHandler)
    }
    
    static func snapshot(from repo: Repo, matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        return try await HubApi.shared.snapshot(from: repo, matching: glob, progressHandler: progressHandler)
    }
    
    static func snapshot(from repoId: String, matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        return try await HubApi.shared.snapshot(from: Repo(id: repoId), matching: glob, progressHandler: progressHandler)
    }
    
    static func whoami(token: String) async throws -> Config {
        return try await HubApi(hfToken: token).whoami()
    }
    
    static func getFileMetadata(fileURL: URL) async throws -> HubApi.FileMetadata {
        return try await HubApi.shared.getFileMetadata(url: fileURL)
    }
    
    static func getFileMetadata(from repo: Repo, matching globs: [String] = []) async throws -> [HubApi.FileMetadata] {
        return try await HubApi.shared.getFileMetadata(from: repo, matching: globs)
    }
    
    static func getFileMetadata(from repoId: String, matching globs: [String] = []) async throws -> [HubApi.FileMetadata] {
        return try await HubApi.shared.getFileMetadata(from: Repo(id: repoId), matching: globs)
    }
    
    static func getFileMetadata(from repo: Repo, matching glob: String) async throws -> [HubApi.FileMetadata] {
        return try await HubApi.shared.getFileMetadata(from: repo, matching: [glob])
    }
    
    static func getFileMetadata(from repoId: String, matching glob: String) async throws -> [HubApi.FileMetadata] {
        return try await HubApi.shared.getFileMetadata(from: Repo(id: repoId), matching: [glob])
    }
}

public extension [String] {
    func matching(glob: String) -> [String] {
        filter { fnmatch(glob, $0, 0) == 0 }
    }
}

/// Only allow relative redirects and reject others
/// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/file_download.py#L258
private class RedirectDelegate: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        // Check if it's a redirect status code (300-399)
        if (300...399).contains(response.statusCode) {
            // Get the Location header
            if let locationString = response.value(forHTTPHeaderField: "Location"),
               let locationUrl = URL(string: locationString) {
                
                // Check if it's a relative redirect (no host component)
                if locationUrl.host == nil {
                    // For relative redirects, construct the new URL using the original request's base
                    if let originalUrl = task.originalRequest?.url,
                       var components = URLComponents(url: originalUrl, resolvingAgainstBaseURL: true) {
                        // Update the path component with the relative path
                        components.path = locationUrl.path
                        components.query = locationUrl.query
                        
                        // Create new request with the resolved URL
                        if let resolvedUrl = components.url {
                            var newRequest = URLRequest(url: resolvedUrl)
                            // Copy headers from original request
                            task.originalRequest?.allHTTPHeaderFields?.forEach { key, value in
                                newRequest.setValue(value, forHTTPHeaderField: key)
                            }
                            newRequest.setValue(resolvedUrl.absoluteString, forHTTPHeaderField: "Location")
                            completionHandler(newRequest)
                            return
                        }
                    }
                }
            }
        }
        
        // For all other cases (non-redirects or absolute redirects), prevent redirect
        completionHandler(nil)
    }
}
