//
//  HubApi.swift
//
//
//  Created by Pedro Cuenca on 20231230.
//

import Crypto
import Foundation
import HuggingFace

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
#if canImport(Network)
import Network
#endif
#if canImport(os)
import os
#endif

/// https://datatracker.ietf.org/doc/html/rfc7540#section-8.1.2
/// `requests` in Python leaves headers as their original casing,
/// where as Swift strictly adheres to RFC 7540 and can force lower case.
/// This is relevant for Xet
enum HFHttpHeaders {
    static let location = "Location"
    static let etag = "Etag"
    static let contentLength = "Content-Length"
    static let repoCommit = "X-Repo-Commit"
    static let linkedEtag = "X-Linked-Etag"
    static let linkedSize = "X-Linked-Size"
    static let xetHash = "x-xet-hash"
    static let xetRefreshRoute = "X-Xet-Refresh-Route"
    static let linkXetAuthKey = "xet-auth"
}

/// Data structure for Xet-enabled file information.
///
/// Contains metadata for files that support Xet protocol optimizations,
/// providing enhanced download performance for large files.
public struct XetFileData {
    /// The cryptographic hash of the file content.
    let fileHash: String
    /// The URL route for refreshing authentication tokens.
    let refreshRoute: String
}

/// `requests` automatically parses Link headers into `response.links`,
///  we implement similar functionality here.
extension HTTPURLResponse {
    func getLinkURL(for rel: String) -> String? {
        guard let linkHeader = allHeaderFields["Link"] as? String else {
            return nil
        }

        for link in linkHeader.split(separator: ",") {
            let trimmed = link.trimmingCharacters(in: .whitespaces)

            if trimmed.contains("rel=\"\(rel)\"") || trimmed.contains("rel=\(rel)") {
                if let start = trimmed.firstIndex(of: "<"),
                    let end = trimmed.firstIndex(of: ">"),
                    start < end
                {
                    let startIndex = trimmed.index(after: start)
                    return String(trimmed[startIndex..<end])
                }
            }
        }

        return nil
    }
}

/// Client for interacting with the Hugging Face Hub API.
///
/// HubApi provides methods for downloading files, retrieving metadata, managing repositories,
/// and handling authentication with the Hugging Face Hub. It supports offline mode,
/// background downloads, and automatic retry mechanisms for robust file transfers.
///
/// ## Storage model
///
/// `HubApi` uses two related but distinct local storage locations:
///
/// - `downloadBase`: materialized snapshot output returned by `snapshot(...)`.
///   Offline-mode validation and per-file metadata checks are performed here.
/// - `HubCache` (via `HubClient(cache: .default)`): shared content-addressed cache used for
///   deduplicated downloads and cache reuse across clients/tools.
///
/// During downloads, data may be sourced from `HubCache` and copied into `downloadBase`.
/// The two locations are intentionally independent for backward compatibility.
///
/// Endpoint and token resolution follow `HubClient` behavior:
/// explicit initializer arguments take precedence; otherwise environment-based
/// detection is used.
public struct HubApi: Sendable {
    var downloadBase: URL
    var endpoint: String
    var useBackgroundSession: Bool
    var useOfflineMode: Bool?
    private let hostURL: URL
    private let tokenProvider: TokenProvider
    private let foregroundCachedClient: HubClient
    private let foregroundUncachedClient: HubClient
    #if !canImport(FoundationNetworking)
    private let backgroundCachedClient: HubClient
    private let backgroundUncachedClient: HubClient
    #endif

    private let networkMonitor = NetworkMonitor()
    public typealias RepoType = Hub.RepoType
    public typealias Repo = Hub.Repo

    /// Session actor for metadata requests with relative redirect handling (used in HEAD requests).
    ///
    /// Static to share a single URLSession across all HubApi instances, preventing resource
    /// exhaustion when many instances are created. Persists for process lifetime.
    private static let redirectSession: RedirectSessionActor = .init()
    private static let hubRepoIdCache: HubRepoIDCacheActor = .init()
    #if !canImport(FoundationNetworking)
    private static let backgroundHubSession: URLSession = {
        let bundleIdentifier = Bundle.main.bundleIdentifier ?? "swift-transformers"
        let identifier = "\(bundleIdentifier).hub.hubclient.background"
        let configuration = URLSessionConfiguration.background(withIdentifier: identifier)
        configuration.isDiscretionary = false
        configuration.sessionSendsLaunchEvents = true
        return URLSession(configuration: configuration)
    }()
    #endif

    /// Initializes a new Hub API client.
    ///
    /// - Parameters:
    ///   - downloadBase: The base directory for local snapshot outputs.
    ///     Defaults to `Documents/huggingface` to preserve historical behavior.
    ///     This location is independent from `HubCache` storage used by cached
    ///     `HubClient` requests, and is the location used by offline snapshot checks.
    ///   - cache: The cache used by cached `HubClient` requests.
    ///     Defaults to `HubCache.default`. Pass `nil` to disable caching.
    ///   - hfToken: The Hugging Face authentication token override.
    ///     If `nil`, token resolution uses `TokenProvider.environment`.
    ///   - endpoint: The Hub endpoint URL override.
    ///     If `nil` (or invalid), host resolution follows `HubClient` defaults.
    ///   - useBackgroundSession: Whether to use background URL sessions for downloads
    ///   - useOfflineMode: Override for offline mode detection (defaults to automatic detection)
    public init(
        downloadBase: URL? = nil,
        cache: HubCache? = .default,
        hfToken: String? = nil,
        endpoint: String? = nil,
        useBackgroundSession: Bool = false,
        useOfflineMode: Bool? = nil
    ) {
        tokenProvider =
            if let hfToken {
                .fixed(token: hfToken)
            } else {
                .environment
            }
        if let downloadBase {
            self.downloadBase = downloadBase
        } else {
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            self.downloadBase = documents.appending(component: "huggingface")
        }
        if let endpoint,
            let parsed = URL(string: endpoint),
            let scheme = parsed.scheme, !scheme.isEmpty,
            let host = parsed.host, !host.isEmpty
        {
            hostURL = parsed
        } else {
            hostURL = HubClient(cache: nil).host
        }
        self.endpoint = hostURL.absoluteString
        self.useBackgroundSession = useBackgroundSession
        self.useOfflineMode = useOfflineMode
        self.foregroundCachedClient = Self.buildHubClient(
            host: hostURL,
            tokenProvider: tokenProvider,
            cache: cache,
            useBackgroundSession: false
        )
        self.foregroundUncachedClient = Self.buildHubClient(
            host: hostURL,
            tokenProvider: tokenProvider,
            cache: nil,
            useBackgroundSession: false
        )
        #if !canImport(FoundationNetworking)
        self.backgroundCachedClient = Self.buildHubClient(
            host: hostURL,
            tokenProvider: tokenProvider,
            cache: cache,
            useBackgroundSession: true
        )
        self.backgroundUncachedClient = Self.buildHubClient(
            host: hostURL,
            tokenProvider: tokenProvider,
            cache: nil,
            useBackgroundSession: true
        )
        #endif
        NetworkMonitor.shared.startMonitoring()
    }

    /// The shared Hub API instance with default configuration.
    public static let shared = HubApi()

    #if canImport(os)
    private static let logger = Logger()
    #else
    private static let logger = PrintLogger()
    #endif
}

#if !canImport(os)
/// Simple print-based logger for non-Apple platforms
private struct PrintLogger {
    func warning(_ message: String) {
        print("[warning] \(message)")
    }
}
#endif

private extension HubApi {
    static func buildHubClient(
        host: URL,
        tokenProvider: TokenProvider,
        cache: HubCache?,
        useBackgroundSession: Bool
    ) -> HubClient {
        #if canImport(FoundationNetworking)
        return HubClient(host: host, tokenProvider: tokenProvider, cache: cache)
        #else
        if useBackgroundSession {
            return HubClient(session: Self.backgroundHubSession, host: host, tokenProvider: tokenProvider, cache: cache)
        }
        return HubClient(host: host, tokenProvider: tokenProvider, cache: cache)
        #endif
    }

    func resolveHubClientRepoID(for repo: Repo) async throws -> HuggingFace.Repo.ID {
        if let parsed = HuggingFace.Repo.ID(rawValue: repo.id) {
            return parsed
        }

        // Legacy compatibility: resolve unqualified model IDs (e.g. "t5-base")
        // to canonical namespace-qualified IDs via the Hub models API.
        if repo.type == .models {
            if let cached = await Self.hubRepoIdCache.get(repo.id) {
                return cached
            }

            let url =
                hostURL
                .appending(path: "api")
                .appending(path: "models")
                .appending(path: repo.id)
            let (data, _) = try await httpGet(for: url)
            let response = try JSONDecoder().decode(ModelInfoResponse.self, from: data)
            if let canonical = HuggingFace.Repo.ID(rawValue: response.id) {
                await Self.hubRepoIdCache.set(repo.id, value: canonical)
                return canonical
            }
        }

        // Keep historical behavior for non-qualified IDs when canonicalization fails.
        return HuggingFace.Repo.ID(namespace: "", name: repo.id)
    }
}

private extension Hub.RepoType {
    var hubClientKind: HuggingFace.Repo.Kind {
        switch self {
        case .models:
            return .model
        case .datasets:
            return .dataset
        case .spaces:
            return .space
        }
    }
}

final class DownloadProgressBridge: @unchecked Sendable {
    private let progress: Progress
    private let handler: (Double, Double?) -> Void
    private let lock = NSLock()
    private var hasCompleted = false
    private var lastFractionCompleted: Double = -1
    private var lastCompletedUnitCount: Int64 = 0
    private var lastSampleDate: Date = .init()

    private var pollTask: Task<Void, Never>?

    init(progress: Progress, handler: @escaping (Double, Double?) -> Void) {
        self.progress = progress
        self.handler = handler
        lastFractionCompleted = min(max(progress.fractionCompleted, 0), 1)
        lastCompletedUnitCount = progress.completedUnitCount
    }

    func start() {
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                self?.emitIfNeeded(force: false)
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }

    func complete() {
        lock.lock()
        if progress.totalUnitCount <= 0 {
            progress.totalUnitCount = 1
        }
        hasCompleted = true
        lock.unlock()
        progress.completedUnitCount = progress.totalUnitCount
        emitIfNeeded(force: false)
        stop()
    }

    func emitCompletionIfFinished() {
        lock.lock()
        let isFinished = progress.fractionCompleted >= 1
        if isFinished {
            hasCompleted = true
        }
        lock.unlock()

        if isFinished {
            emitIfNeeded(force: false)
        }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    func emitIfNeeded(force: Bool) {
        lock.lock()
        let completedUnitCount = progress.completedUnitCount
        let fraction = min(max(progress.fractionCompleted, 0), 1)
        if fraction >= 1, !hasCompleted {
            lock.unlock()
            return
        }
        let fractionChanged = abs(fraction - lastFractionCompleted) >= 0.001
        let bytesChanged = completedUnitCount != lastCompletedUnitCount
        if !force, !fractionChanged, !bytesChanged {
            lock.unlock()
            return
        }

        let now = Date()
        let deltaBytes = completedUnitCount - lastCompletedUnitCount
        let deltaTime = now.timeIntervalSince(lastSampleDate)
        let speed: Double?
        if deltaBytes > 0 {
            speed = deltaTime > 0 ? Double(deltaBytes) / deltaTime : 0
        } else {
            speed = nil
        }

        if deltaBytes > 0 {
            lastCompletedUnitCount = completedUnitCount
            lastSampleDate = now
        }
        lastFractionCompleted = fraction
        lock.unlock()

        handler(fraction, speed)
    }

}

/// File retrieval
public extension HubApi {
    private struct ModelInfoResponse: Decodable {
        let id: String
    }

    /// Represents a file in a repository.
    ///
    /// Contains metadata about files available in a Hub repository,
    /// used for file discovery and listing operations.
    struct Sibling: Codable {
        /// The relative filename within the repository.
        let rfilename: String
    }

    /// Response structure for repository file listings.
    ///
    /// Contains the list of files available in a repository,
    /// returned by the Hub API when querying repository contents.
    struct SiblingsResponse: Codable {
        /// Array of files in the repository.
        let siblings: [Sibling]
    }

    /// Performs an HTTP GET request with authentication and error handling.
    ///
    /// - Parameter url: The URL to request
    /// - Returns: A tuple containing the response data and HTTP response
    /// - Throws: HubClientError for authentication, network, or HTTP errors
    func httpGet(for url: URL) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url)
        if let token = try await tokenProvider.getToken(), !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw Hub.HubClientError.unexpectedError
            }

            switch httpResponse.statusCode {
            case 200..<300:
                return (data, httpResponse)
            case 401, 403:
                throw Hub.HubClientError.authorizationRequired
            case 404:
                throw Hub.HubClientError.fileNotFound(url.lastPathComponent)
            default:
                throw Hub.HubClientError.httpStatusCode(httpResponse.statusCode)
            }
        } catch let error as Hub.HubClientError {
            throw error
        } catch {
            throw Hub.HubClientError.downloadError(error.localizedDescription)
        }
    }

    /// Performs an HTTP HEAD request to retrieve metadata without downloading content.
    ///
    /// Uses platform-specific redirect handling:
    /// - Apple platforms: custom session that only allows relative redirects and blocks absolute redirects.
    /// - Linux: default URLSession redirect handling.
    ///
    /// - Parameter url: The URL to request
    /// - Returns: The HTTP response containing headers and status code
    /// - Throws: HubClientError if the page does not exist or is not accessible
    func httpHead(for url: URL) async throws -> HTTPURLResponse {
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        if let token = try await tokenProvider.getToken(), !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")

        let response: URLResponse
        #if canImport(FoundationNetworking)
        // Linux: let URLSession handle redirects with default behavior.
        (_, response) = try await URLSession.shared.data(for: request)
        #else
        // Apple platforms: use shared session with custom relative-redirect handling.
        let session = await Self.redirectSession.get()
        (_, response) = try await session.data(for: request)
        #endif
        guard let response = response as? HTTPURLResponse else { throw Hub.HubClientError.unexpectedError }

        switch response.statusCode {
        case 200..<400: break // Allow redirects to pass through to the redirect delegate
        case 401, 403: throw Hub.HubClientError.authorizationRequired
        case 404: throw Hub.HubClientError.fileNotFound(url.lastPathComponent)
        default: throw Hub.HubClientError.httpStatusCode(response.statusCode)
        }

        return response
    }

    /// Retrieves the list of filenames in a repository that match the specified glob patterns.
    ///
    /// - Parameters:
    ///   - repo: The repository to query
    ///   - revision: The git revision to use (defaults to "main")
    ///   - globs: Array of glob patterns to filter files (empty array returns all files)
    /// - Returns: Array of matching filenames
    /// - Throws: HubClientError if the repository cannot be accessed or parsed
    func getFilenames(from repo: Repo, revision: String = "main", matching globs: [String] = []) async throws -> [String] {
        // Read repo info and only parse "siblings"
        let url =
            hostURL
            .appending(path: "api")
            .appending(path: repo.type.rawValue)
            .appending(path: repo.id)
            .appending(path: "revision")
            .appending(component: revision) // Encode slashes (e.g., "pr/1" -> "pr%2F1")
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
        try await getFilenames(from: Repo(id: repoId), matching: globs)
    }

    func getFilenames(from repo: Repo, matching glob: String) async throws -> [String] {
        try await getFilenames(from: repo, matching: [glob])
    }

    func getFilenames(from repoId: String, matching glob: String) async throws -> [String] {
        try await getFilenames(from: Repo(id: repoId), matching: [glob])
    }
}

/// Additional Errors
public extension HubApi {
    enum EnvironmentError: LocalizedError {
        case invalidMetadataError(String)
        case offlineModeError(String)
        case fileIntegrityError(String)
        case fileWriteError(String)

        public var errorDescription: String? {
            switch self {
            case let .invalidMetadataError(message):
                String(localized: "Invalid metadata: \(message)")
            case let .offlineModeError(message):
                String(localized: "Offline mode error: \(message)")
            case let .fileIntegrityError(message):
                String(localized: "File integrity check failed: \(message)")
            case let .fileWriteError(message):
                String(localized: "Failed to write file: \(message)")
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
        do {
            return try YYJSONParser.parseToConfig(data)
        } catch {
            throw Hub.HubClientError.jsonSerialization(
                fileURL: fileURL,
                message: "JSON parsing failed for \(fileURL): \(error.localizedDescription). If this is a private model, verify that HF_TOKEN is set."
            )
        }
    }
}

/// Whoami
public extension HubApi {
    func whoami() async throws -> Config {
        guard let token = try await tokenProvider.getToken(), !token.isEmpty else { throw Hub.HubClientError.authorizationRequired }
        let url =
            hostURL
            .appending(path: "api")
            .appending(path: "whoami-v2")
        let (data, _) = try await httpGet(for: url)

        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [NSString: Any] else { throw Hub.HubClientError.parse }
        return Config(dictionary)
    }
}

/// Snaphsot download
public extension HubApi {
    /// Returns the on-disk snapshot root for a repository under `downloadBase`.
    ///
    /// This path is the materialized repository location used by `snapshot(...)`
    /// and offline-mode checks. It is separate from the internal `HubCache` directory.
    func localRepoLocation(_ repo: Repo) -> URL {
        downloadBase.appending(component: repo.type.rawValue).appending(component: repo.id)
    }

    /// Reads metadata about a file in the local directory related to a download process.
    ///
    /// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/_local_folder.py#L263
    ///
    /// - Parameter metadataPath: The path of the metadata file to read.
    /// - Throws: An `EnvironmentError.invalidMetadataError` if the metadata file is invalid and cannot be removed.
    /// - Returns: A `LocalDownloadFileMetadata` object if the metadata file exists and is valid, or `nil` if the file is missing or invalid.
    func readDownloadMetadata(metadataPath: URL) throws -> LocalDownloadFileMetadata? {
        if FileManager.default.fileExists(atPath: metadataPath.path) {
            do {
                let contents = try String(contentsOf: metadataPath, encoding: .utf8)
                let lines = contents.components(separatedBy: .newlines)

                guard lines.count >= 3 else {
                    throw EnvironmentError.invalidMetadataError(("Metadata file is missing required fields"))
                }

                let commitHash = lines[0].trimmingCharacters(in: .whitespacesAndNewlines)
                let etag = lines[1].trimmingCharacters(in: .whitespacesAndNewlines)

                guard let timestamp = Double(lines[2].trimmingCharacters(in: .whitespacesAndNewlines)) else {
                    throw EnvironmentError.invalidMetadataError(("Invalid timestamp format"))
                }

                let timestampDate = Date(timeIntervalSince1970: timestamp)
                let filename = metadataPath.lastPathComponent.replacingOccurrences(of: ".metadata", with: "")

                return LocalDownloadFileMetadata(commitHash: commitHash, etag: etag, filename: filename, timestamp: timestampDate)
            } catch let error as EnvironmentError {
                do {
                    HubApi.logger.warning("Invalid metadata file \(metadataPath): \(error.localizedDescription). Removing it from disk and continuing.")
                    try FileManager.default.removeItem(at: metadataPath)
                } catch {
                    throw EnvironmentError.invalidMetadataError(("Could not remove corrupted metadata file: \(error.localizedDescription)"))
                }
                return nil
            } catch {
                do {
                    HubApi.logger.warning("Error reading metadata file \(metadataPath): \(error.localizedDescription). Removing it from disk and continuing.")
                    try FileManager.default.removeItem(at: metadataPath)
                } catch {
                    throw EnvironmentError.invalidMetadataError(("Could not remove corrupted metadata file: \(error.localizedDescription)"))
                }
                return nil
            }
        }

        // metadata file does not exist
        return nil
    }

    func isValidCommitHash(_ hash: String) -> Bool {
        guard hash.count == 40 else { return false }
        guard hash.lowercased() == hash else { return false }
        return hash.allSatisfy { $0.isHexDigit }
    }

    func isValidSHA256(_ hash: String) -> Bool {
        guard hash.count == 64 else { return false }
        guard hash.lowercased() == hash else { return false }
        return hash.allSatisfy { $0.isHexDigit }
    }

    func computeFileHash(file url: URL) throws -> String {
        // Open file for reading
        guard let fileHandle = try? FileHandle(forReadingFrom: url) else {
            throw Hub.HubClientError.fileNotFound(url.lastPathComponent)
        }

        defer {
            try? fileHandle.close()
        }

        var hasher = SHA256()
        let chunkSize = 1024 * 1024 // 1MB chunks

        func readNextChunk() -> Bool {
            let nextChunk = try? fileHandle.read(upToCount: chunkSize)

            guard let nextChunk,
                !nextChunk.isEmpty
            else {
                return false
            }

            hasher.update(data: nextChunk)

            return true
        }

        #if canImport(ObjectiveC)
        while autoreleasepool(invoking: readNextChunk) {}
        #else
        while readNextChunk() {}
        #endif

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/_local_folder.py#L391
    func writeDownloadMetadata(commitHash: String, etag: String, metadataPath: URL) throws {
        let metadataContent = "\(commitHash)\n\(etag)\n\(Date().timeIntervalSince1970)\n"
        do {
            try FileManager.default.createDirectory(at: metadataPath.deletingLastPathComponent(), withIntermediateDirectories: true)
            try metadataContent.write(to: metadataPath, atomically: true, encoding: .utf8)
        } catch {
            throw EnvironmentError.fileWriteError(("Failed to write metadata to \(metadataPath.path): \(error.localizedDescription)"))
        }
    }

    struct HubFileDownloader {
        let hub: HubApi
        let repo: Repo
        let revision: String
        let repoDestination: URL
        let repoMetadataDestination: URL
        let relativeFilename: String
        let backgroundSession: Bool

        var source: URL {
            // https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/tokenizer.json?download=true
            var url = hub.hostURL
            if repo.type != .models {
                url = url.appending(path: repo.type.rawValue)
            }
            url = url.appending(path: repo.id)
            url = url.appending(path: "resolve")
            url = url.appending(component: revision) // Encode slashes (e.g., "pr/1" -> "pr%2F1")
            url = url.appending(path: relativeFilename)
            return url
        }

        var destination: URL {
            repoDestination.appending(path: relativeFilename)
        }

        var metadataDestination: URL {
            repoMetadataDestination.appending(path: relativeFilename + ".metadata")
        }

        var downloaded: Bool {
            FileManager.default.fileExists(atPath: destination.path)
        }

        /// We're using incomplete destination to prepare cache destination because incomplete files include lfs + non-lfs files (vs only lfs for metadata files)
        func prepareCacheDestination(_ incompleteDestination: URL) throws {
            let directoryURL = incompleteDestination.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil)
        }

        /// Downloads the file with progress tracking.
        /// - Parameter progressHandler: Called with download progress (0.0-1.0) and speed in bytes/sec, if available.
        /// - Returns: Local file URL (uses cached file if commit hash matches).
        /// - Throws: ``EnvironmentError`` errors for file and metadata validation failures, ``Hub.HubClientError`` errors during transfer, or ``CancellationError`` if the task is cancelled.
        @discardableResult
        func download(progressHandler: @escaping (Double, Double?) -> Void) async throws -> URL {
            let localMetadata = try hub.readDownloadMetadata(metadataPath: metadataDestination)
            let remoteMetadata = try await hub.getFileMetadata(url: source)

            let localCommitHash = localMetadata?.commitHash ?? ""
            let remoteCommitHash = remoteMetadata.commitHash ?? ""

            // Local file exists + metadata exists + commit_hash matches => return file
            if hub.isValidCommitHash(remoteCommitHash), downloaded, localMetadata != nil,
                localCommitHash == remoteCommitHash
            {
                return destination
            }

            // From now on, etag, commit_hash, url and size are not empty
            guard let remoteCommitHash = remoteMetadata.commitHash,
                let remoteEtag = remoteMetadata.etag,
                let remoteSize = remoteMetadata.size,
                remoteMetadata.location != ""
            else {
                throw EnvironmentError.invalidMetadataError("File metadata must have been retrieved from server")
            }

            // Local file exists => check if it's up-to-date
            if downloaded {
                // etag matches => update metadata and return file
                if localMetadata?.etag == remoteEtag {
                    try hub.writeDownloadMetadata(commitHash: remoteCommitHash, etag: remoteEtag, metadataPath: metadataDestination)
                    return destination
                }

                // etag is a sha256
                // => means it's an LFS file (large)
                // => let's compute local hash and compare
                // => if match, update metadata and return file
                if hub.isValidSHA256(remoteEtag) {
                    let fileHash = try hub.computeFileHash(file: destination)
                    if fileHash == remoteEtag {
                        try hub.writeDownloadMetadata(commitHash: remoteCommitHash, etag: remoteEtag, metadataPath: metadataDestination)
                        return destination
                    }
                }
            }

            // Otherwise, let's download the file!
            let incompleteDestination = repoMetadataDestination.appending(path: relativeFilename + ".\(remoteEtag).incomplete")
            try prepareCacheDestination(incompleteDestination)
            if FileManager.default.fileExists(atPath: incompleteDestination.path) {
                try? FileManager.default.removeItem(at: incompleteDestination)
            }
            let forceDownload = downloaded
            let downloadProgress = Progress(totalUnitCount: Int64(max(remoteSize, 1)))
            let progressBridge = DownloadProgressBridge(progress: downloadProgress, handler: progressHandler)
            progressBridge.start()
            defer { progressBridge.stop() }

            try await withTaskCancellationHandler {
                do {
                    let client: HubClient
                    #if canImport(FoundationNetworking)
                    client = !forceDownload ? hub.foregroundCachedClient : hub.foregroundUncachedClient
                    #else
                    if backgroundSession, !forceDownload {
                        client = hub.backgroundCachedClient
                    } else if backgroundSession {
                        client = hub.backgroundUncachedClient
                    } else {
                        client = !forceDownload ? hub.foregroundCachedClient : hub.foregroundUncachedClient
                    }
                    #endif

                    let hubRepoID = try await hub.resolveHubClientRepoID(for: repo)
                    _ = try await client.downloadFile(
                        at: relativeFilename,
                        from: hubRepoID,
                        to: destination,
                        kind: repo.type.hubClientKind,
                        revision: revision,
                        progress: downloadProgress
                    )

                    try hub.writeDownloadMetadata(commitHash: remoteCommitHash, etag: remoteEtag, metadataPath: metadataDestination)
                    progressBridge.complete()
                } catch let error as Hub.HubClientError {
                    let context = "\(repo.id)@\(revision)/\(relativeFilename) (\(source.absoluteString) -> \(destination.path))"
                    let missingPath = "\(repo.id)@\(revision)/\(relativeFilename)"
                    switch error {
                    case let .httpStatusCode(statusCode):
                        switch statusCode {
                        case 401, 403:
                            throw Hub.HubClientError.authorizationRequired
                        case 404:
                            throw Hub.HubClientError.fileNotFound(missingPath)
                        case 429:
                            throw Hub.HubClientError.downloadError("Rate limited while downloading \(context)")
                        default:
                            throw error
                        }
                    case .fileNotFound:
                        throw Hub.HubClientError.fileNotFound(missingPath)
                    default:
                        throw error
                    }
                }
            } onCancel: {
                progressBridge.emitCompletionIfFinished()
                progressBridge.stop()
            }

            return destination
        }
    }

    @discardableResult
    func snapshot(from repo: Repo, revision: String = "main", matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in })
        async throws -> URL
    {
        let repoDestination = localRepoLocation(repo)
        let repoMetadataDestination =
            repoDestination
            .appending(path: ".cache")
            .appending(path: "huggingface")
            .appending(path: "download")

        let shouldUseOfflineMode = await NetworkMonitor.shared.state.shouldUseOfflineMode()

        if useOfflineMode ?? shouldUseOfflineMode {
            if !FileManager.default.fileExists(atPath: repoDestination.path) {
                throw EnvironmentError.offlineModeError(("Repository not available locally"))
            }

            let fileUrls = try FileManager.default.getFileUrls(at: repoDestination)
            if fileUrls.isEmpty {
                throw EnvironmentError.offlineModeError(("No files available locally for this repository"))
            }

            for fileUrl in fileUrls {
                let metadataPath = URL(
                    fileURLWithPath: fileUrl.path.replacingOccurrences(
                        of: repoDestination.path,
                        with: repoMetadataDestination.path
                    ) + ".metadata"
                )

                let localMetadata = try readDownloadMetadata(metadataPath: metadataPath)

                guard let localMetadata else {
                    throw EnvironmentError.offlineModeError(("Metadata not available for \(fileUrl.lastPathComponent)"))
                }
                let localEtag = localMetadata.etag

                // LFS file so check file integrity
                if isValidSHA256(localEtag) {
                    let fileHash = try computeFileHash(file: fileUrl)
                    if fileHash != localEtag {
                        throw EnvironmentError.fileIntegrityError(("Hash mismatch for \(fileUrl.lastPathComponent)"))
                    }
                }
            }

            return repoDestination
        }

        let filenames = try await getFilenames(from: repo, revision: revision, matching: globs)
        let progress = Progress(totalUnitCount: Int64(filenames.count))
        for filename in filenames {
            let fileProgress = Progress(totalUnitCount: 100, parent: progress, pendingUnitCount: 1)
            let downloader = HubFileDownloader(
                hub: self,
                repo: repo,
                revision: revision,
                repoDestination: repoDestination,
                repoMetadataDestination: repoMetadataDestination,
                relativeFilename: filename,
                backgroundSession: useBackgroundSession
            )

            try await downloader.download { fractionDownloaded, speed in
                fileProgress.completedUnitCount = Int64(100 * fractionDownloaded)
                let throughputValue: Any = speed ?? NSNull()
                fileProgress.setUserInfoObject(throughputValue, forKey: .throughputKey)
                progress.setUserInfoObject(throughputValue, forKey: .throughputKey)
                progressHandler(progress)
            }
            if Task.isCancelled {
                return repoDestination
            }

            fileProgress.completedUnitCount = 100
        }

        progressHandler(progress)
        return repoDestination
    }

    /// New overloads exposing speed directly in the snapshot progress handler
    @discardableResult func snapshot(from repo: Repo, revision: String = "main", matching globs: [String] = [], progressHandler: @escaping (Progress, Double?) -> Void) async throws -> URL {
        try await snapshot(from: repo, revision: revision, matching: globs) { progress in
            let speed = progress.userInfo[.throughputKey] as? Double
            progressHandler(progress, speed)
        }
    }

    @discardableResult
    func snapshot(from repoId: String, revision: String = "main", matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await snapshot(from: Repo(id: repoId), revision: revision, matching: globs, progressHandler: progressHandler)
    }

    @discardableResult
    func snapshot(from repo: Repo, revision: String = "main", matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await snapshot(from: repo, revision: revision, matching: [glob], progressHandler: progressHandler)
    }

    @discardableResult
    func snapshot(from repoId: String, revision: String = "main", matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await snapshot(from: Repo(id: repoId), revision: revision, matching: [glob], progressHandler: progressHandler)
    }

    /// Convenience overloads for other snapshot entry points with speed
    @discardableResult
    func snapshot(from repoId: String, revision: String = "main", matching globs: [String] = [], progressHandler: @escaping (Progress, Double?) -> Void) async throws -> URL {
        try await snapshot(from: Repo(id: repoId), revision: revision, matching: globs, progressHandler: progressHandler)
    }

    @discardableResult
    func snapshot(from repo: Repo, revision: String = "main", matching glob: String, progressHandler: @escaping (Progress, Double?) -> Void) async throws -> URL {
        try await snapshot(from: repo, revision: revision, matching: [glob], progressHandler: progressHandler)
    }

    @discardableResult
    func snapshot(from repoId: String, revision: String = "main", matching glob: String, progressHandler: @escaping (Progress, Double?) -> Void) async throws -> URL {
        try await snapshot(from: Repo(id: repoId), revision: revision, matching: [glob], progressHandler: progressHandler)
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

        /// Xet file data, if available. Contains the file hash and the refresh route.
        public let xetFileData: XetFileData?
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
        guard let etag else { return nil }
        return etag.trimmingPrefix("W/").trimmingCharacters(in: CharacterSet(charactersIn: "\""))
    }

    func getFileMetadata(url: URL) async throws -> FileMetadata {
        let response = try await httpHead(for: url)
        let location = response.statusCode == 302 ? response.value(forHTTPHeaderField: "Location") : response.url?.absoluteString

        return FileMetadata(
            commitHash: response.value(forHTTPHeaderField: HFHttpHeaders.repoCommit),
            etag: normalizeEtag(
                (response.value(forHTTPHeaderField: HFHttpHeaders.linkedEtag)) ?? (response.value(forHTTPHeaderField: HFHttpHeaders.etag))
            ),
            location: location ?? url.absoluteString,
            size: Int(response.value(forHTTPHeaderField: HFHttpHeaders.linkedSize) ?? response.value(forHTTPHeaderField: HFHttpHeaders.contentLength) ?? ""),
            xetFileData: parseXetFileDataFromResponse(response: response)
        )
    }

    /// https://github.com/huggingface/huggingface_hub/blob/b698915d6b582c72806ac3e91c43bfd8dde35856/src/huggingface_hub/utils/_xet.py#L29
    private func parseXetFileDataFromResponse(
        response: HTTPURLResponse?
    ) -> XetFileData? {
        guard let response else {
            return nil
        }

        guard let fileHash = response.allHeaderFields[HFHttpHeaders.xetHash] as? String else {
            return nil
        }

        guard
            var refreshRoute = response.getLinkURL(for: HFHttpHeaders.linkXetAuthKey)
                ?? response.allHeaderFields[HFHttpHeaders.xetRefreshRoute] as? String
        else {
            return nil
        }

        let defaultEndpoint = HubClient.defaultHost.absoluteString

        if refreshRoute.hasPrefix(defaultEndpoint) {
            refreshRoute = refreshRoute.replacingOccurrences(
                of: defaultEndpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/")),
                with: endpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            )
        }

        return XetFileData(
            fileHash: fileHash,
            refreshRoute: refreshRoute
        )
    }

    func getFileMetadata(from repo: Repo, revision: String = "main", matching globs: [String] = []) async throws -> [FileMetadata] {
        let files = try await getFilenames(from: repo, matching: globs)
        let url =
            hostURL
            .appending(path: repo.id)
            .appending(path: "resolve")
            .appending(component: revision) // Encode slashes (e.g., "pr/1" -> "pr%2F1")
        var selectedMetadata: [FileMetadata] = []
        for file in files {
            let fileURL = url.appending(path: file)
            try await selectedMetadata.append(getFileMetadata(url: fileURL))
        }
        return selectedMetadata
    }

    func getFileMetadata(from repoId: String, revision: String = "main", matching globs: [String] = []) async throws -> [FileMetadata] {
        try await getFileMetadata(from: Repo(id: repoId), revision: revision, matching: globs)
    }

    func getFileMetadata(from repo: Repo, revision: String = "main", matching glob: String) async throws -> [FileMetadata] {
        try await getFileMetadata(from: repo, revision: revision, matching: [glob])
    }

    func getFileMetadata(from repoId: String, revision: String = "main", matching glob: String) async throws -> [FileMetadata] {
        try await getFileMetadata(from: Repo(id: repoId), revision: revision, matching: [glob])
    }
}

/// Network monitor helper class to help decide whether to use offline mode
extension HubApi {
    private actor NetworkStateActor {
        /// Assume best case connection until updated by the monitor
        public var isConnected: Bool = true
        public var isExpensive: Bool = false
        public var isConstrained: Bool = false

        #if canImport(Network)
        func update(path: NWPath) {
            isConnected = path.status == .satisfied
            isExpensive = path.isExpensive
            isConstrained = path.isConstrained
        }
        #endif

        func shouldUseOfflineMode() -> Bool {
            if ProcessInfo.processInfo.environment["CI_DISABLE_NETWORK_MONITOR"] == "1" {
                return false
            }
            return !isConnected
        }
    }

    private final class NetworkMonitor: Sendable {
        #if canImport(Network)
        private let monitor: NWPathMonitor
        private let queue: DispatchQueue
        #endif

        public let state: NetworkStateActor = .init()

        static let shared = NetworkMonitor()

        init() {
            #if canImport(Network)
            monitor = NWPathMonitor()
            queue = DispatchQueue(label: "HubApi.NetworkMonitor")
            startMonitoring()
            #endif
        }

        func startMonitoring() {
            #if canImport(Network)
            monitor.pathUpdateHandler = { [weak self] path in
                guard let self else { return }
                Task {
                    await self.state.update(path: path)
                }
            }

            monitor.start(queue: queue)
            #endif
        }

        func stopMonitoring() {
            #if canImport(Network)
            monitor.cancel()
            #endif
        }

        deinit {
            stopMonitoring()
        }
    }
}

/// Convenience methods that use the shared `HubApi` instance
public extension Hub {
    /// Retrieves filenames from a repository using the shared Hub API instance.
    ///
    /// - Parameters:
    ///   - repo: The repository to query
    ///   - globs: Array of glob patterns to filter files (defaults to all files)
    /// - Returns: Array of matching filenames
    /// - Throws: HubClientError if the operation fails
    static func getFilenames(from repo: Hub.Repo, matching globs: [String] = []) async throws -> [String] {
        try await HubApi.shared.getFilenames(from: repo, matching: globs)
    }

    /// Retrieves filenames from a repository by ID using the shared Hub API instance.
    ///
    /// - Parameters:
    ///   - repoId: The repository ID to query
    ///   - globs: Array of glob patterns to filter files (defaults to all files)
    /// - Returns: Array of matching filenames
    /// - Throws: HubClientError if the operation fails
    static func getFilenames(from repoId: String, matching globs: [String] = []) async throws -> [String] {
        try await HubApi.shared.getFilenames(from: Repo(id: repoId), matching: globs)
    }

    /// Retrieves filenames from a repository matching a single glob pattern.
    ///
    /// - Parameters:
    ///   - repo: The repository to query
    ///   - glob: The glob pattern to filter files
    /// - Returns: Array of matching filenames
    /// - Throws: HubClientError if the operation fails
    static func getFilenames(from repo: Repo, matching glob: String) async throws -> [String] {
        try await HubApi.shared.getFilenames(from: repo, matching: glob)
    }

    /// Retrieves filenames from a repository by ID matching a single glob pattern.
    ///
    /// - Parameters:
    ///   - repoId: The repository ID to query
    ///   - glob: The glob pattern to filter files
    /// - Returns: Array of matching filenames
    /// - Throws: HubClientError if the operation fails
    static func getFilenames(from repoId: String, matching glob: String) async throws -> [String] {
        try await HubApi.shared.getFilenames(from: Repo(id: repoId), matching: glob)
    }

    /// Downloads a repository snapshot using the shared Hub API instance.
    ///
    /// - Parameters:
    ///   - repo: The repository to download
    ///   - globs: Array of glob patterns to filter files (defaults to all files)
    ///   - progressHandler: Closure called with download progress updates
    /// - Returns: URL to the local repository directory
    /// - Throws: HubClientError if the download fails
    static func snapshot(from repo: Repo, matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await HubApi.shared.snapshot(from: repo, matching: globs, progressHandler: progressHandler)
    }

    static func snapshot(from repoId: String, matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in }) async throws
        -> URL
    {
        try await HubApi.shared.snapshot(from: Repo(id: repoId), matching: globs, progressHandler: progressHandler)
    }

    static func snapshot(from repo: Repo, matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await HubApi.shared.snapshot(from: repo, matching: glob, progressHandler: progressHandler)
    }

    static func snapshot(from repoId: String, matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await HubApi.shared.snapshot(from: Repo(id: repoId), matching: glob, progressHandler: progressHandler)
    }

    /// Overloads exposing speed via (Progress, Double?) where Double is bytes/sec
    static func snapshot(from repo: Repo, matching globs: [String] = [], progressHandler: @escaping (Progress, Double?) -> Void) async throws -> URL {
        try await HubApi.shared.snapshot(from: repo, matching: globs, progressHandler: progressHandler)
    }

    static func snapshot(from repoId: String, matching globs: [String] = [], progressHandler: @escaping (Progress, Double?) -> Void) async throws -> URL {
        try await HubApi.shared.snapshot(from: Repo(id: repoId), matching: globs, progressHandler: progressHandler)
    }

    static func snapshot(from repo: Repo, matching glob: String, progressHandler: @escaping (Progress, Double?) -> Void) async throws -> URL {
        try await HubApi.shared.snapshot(from: repo, matching: glob, progressHandler: progressHandler)
    }

    static func snapshot(from repoId: String, matching glob: String, progressHandler: @escaping (Progress, Double?) -> Void) async throws -> URL {
        try await HubApi.shared.snapshot(from: Repo(id: repoId), matching: glob, progressHandler: progressHandler)
    }

    /// Retrieves user information using the provided authentication token.
    ///
    /// - Parameter token: The Hugging Face authentication token
    /// - Returns: Configuration containing user information
    /// - Throws: HubClientError if authentication fails or the request is invalid
    static func whoami(token: String) async throws -> Config {
        try await HubApi(hfToken: token).whoami()
    }

    static func getFileMetadata(fileURL: URL) async throws -> HubApi.FileMetadata {
        try await HubApi.shared.getFileMetadata(url: fileURL)
    }

    static func getFileMetadata(from repo: Repo, matching globs: [String] = []) async throws -> [HubApi.FileMetadata] {
        try await HubApi.shared.getFileMetadata(from: repo, matching: globs)
    }

    static func getFileMetadata(from repoId: String, matching globs: [String] = []) async throws -> [HubApi.FileMetadata] {
        try await HubApi.shared.getFileMetadata(from: Repo(id: repoId), matching: globs)
    }

    static func getFileMetadata(from repo: Repo, matching glob: String) async throws -> [HubApi.FileMetadata] {
        try await HubApi.shared.getFileMetadata(from: repo, matching: [glob])
    }

    static func getFileMetadata(from repoId: String, matching glob: String) async throws -> [HubApi.FileMetadata] {
        try await HubApi.shared.getFileMetadata(from: Repo(id: repoId), matching: [glob])
    }
}

private extension [String] {
    func matching(glob: String) -> [String] {
        filter { fnmatch(glob, $0, 0) == 0 }
    }
}

private extension FileManager {
    func getFileUrls(at directoryUrl: URL) throws -> [URL] {
        var fileUrls = [URL]()

        // Get all contents including subdirectories
        guard
            let enumerator = FileManager.default.enumerator(
                at: directoryUrl,
                includingPropertiesForKeys: [.isRegularFileKey, .isHiddenKey],
                options: [.skipsHiddenFiles]
            )
        else {
            return fileUrls
        }

        for case let fileURL as URL in enumerator {
            do {
                let resourceValues = try fileURL.resourceValues(forKeys: [.isRegularFileKey, .isHiddenKey])
                if resourceValues.isRegularFile == true, resourceValues.isHidden != true {
                    fileUrls.append(fileURL)
                }
            } catch {
                throw error
            }
        }

        return fileUrls
    }
}

/// Only allow relative redirects and reject others
/// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/file_download.py#L258
private final class RedirectDelegate: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        // Check if it's a redirect status code (300-399)
        if (300...399).contains(response.statusCode) {
            // Get the Location header
            if let locationString = response.value(forHTTPHeaderField: "Location"),
                let locationUrl = URL(string: locationString)
            {
                // Check if it's a relative redirect (no host component)
                if locationUrl.host == nil {
                    // For relative redirects, construct the new URL using the original request's base
                    if let originalUrl = task.originalRequest?.url,
                        var components = URLComponents(url: originalUrl, resolvingAgainstBaseURL: true)
                    {
                        // Update the path component with the relative path
                        components.path = locationUrl.path
                        components.query = locationUrl.query

                        // Create new request with the resolved URL
                        if let resolvedUrl = components.url {
                            var newRequest = URLRequest(url: resolvedUrl)
                            // Copy headers from original request
                            if let headers = task.originalRequest?.allHTTPHeaderFields {
                                for (key, value) in headers {
                                    newRequest.setValue(value, forHTTPHeaderField: key)
                                }
                            }
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

/// Actor to manage shared URLSession for redirect handling.
///
/// Lazily initializes and reuses a single URLSession across all HubApi instances
/// to avoid resource exhaustion when running multiple tests or creating many instances.
private actor RedirectSessionActor {
    private var urlSession: URLSession?

    func get() -> URLSession {
        if let urlSession = urlSession {
            return urlSession
        }

        // Create session once and reuse
        let redirectDelegate = RedirectDelegate()
        let session = URLSession(configuration: .default, delegate: redirectDelegate, delegateQueue: nil)
        self.urlSession = session
        return session
    }
}

private actor HubRepoIDCacheActor {
    private var values: [String: HuggingFace.Repo.ID] = [:]

    func get(_ key: String) -> HuggingFace.Repo.ID? {
        values[key]
    }

    func set(_ key: String, value: HuggingFace.Repo.ID) {
        values[key] = value
    }
}

#if !canImport(Darwin)
// Linux Foundation may not provide String(localized:comment:), so keep call sites portable.
private extension String {
    init(localized key: String, comment: String? = nil) {
        self = key
    }
}
#endif
