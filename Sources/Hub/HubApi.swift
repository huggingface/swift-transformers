//
//  HubApi.swift
//
//
//  Created by Pedro Cuenca on 20231230.
//

import CryptoKit
import Foundation
import Network
import os

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

public struct XetFileData {
    let fileHash: String
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

public struct HubApi: Sendable {
    var downloadBase: URL
    var hfToken: String?
    var endpoint: String
    var useBackgroundSession: Bool
    var useOfflineMode: Bool?

    private let networkMonitor = NetworkMonitor()
    public typealias RepoType = Hub.RepoType
    public typealias Repo = Hub.Repo

    /// Proxy configuration dictionary - computed property for Sendable compliance
    private var proxyConfig: [String: Any]? {
        Self.configureProxySettings()
    }

    public init(
        downloadBase: URL? = nil,
        hfToken: String? = nil,
        endpoint: String? = nil,
        useBackgroundSession: Bool = false,
        useOfflineMode: Bool? = nil
    ) {
        self.hfToken = hfToken ?? Self.hfTokenFromEnv()

        let debugPrint = ProcessInfo.processInfo.environment["CI_DISABLE_NETWORK_MONITOR"] == "1"
        if debugPrint {
            print(self.hfToken == nil ? "🔴 NO TOKEN **" : "✅ got token")
        }
        if let downloadBase {
            self.downloadBase = downloadBase
        } else {
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            self.downloadBase = documents.appending(component: "huggingface")
        }
        self.endpoint = endpoint ?? Self.hfEndpointfromEnv()
        self.useBackgroundSession = useBackgroundSession
        self.useOfflineMode = useOfflineMode

        NetworkMonitor.shared.startMonitoring()
    }

    let sha256Pattern = "^[0-9a-f]{64}$"
    let commitHashPattern = "^[0-9a-f]{40}$"

    public static let shared = HubApi()

    private static let logger = Logger()
}

private extension HubApi {
    static func hfEndpointfromEnv() -> String {
        ProcessInfo.processInfo.environment["HF_ENDPOINT"] ?? "https://huggingface.co"
    }

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
            { try? String(contentsOf: .homeDirectory.appendingPathComponent(".huggingface/token"), encoding: .utf8) },
        ]
        return possibleTokens
            .lazy
            .compactMap { $0() }
            .filter { !$0.isEmpty }
            .first
    }

    /// Configure proxy settings from environment variables
    static func configureProxySettings() -> [String: Any]? {
        #if os(macOS)
        var proxyConfig: [String: Any] = [:]
        var hasProxyConfig = false

        // Configure HTTPS proxy
        if let httpsProxy = ProcessInfo.processInfo.environment["https_proxy"],
           let proxySettings = parseProxyURL(httpsProxy, type: "HTTPS")
        {
            proxyConfig.merge(proxySettings) { _, new in new }
            hasProxyConfig = true
        }

        // Configure HTTP proxy
        if let httpProxy = ProcessInfo.processInfo.environment["http_proxy"],
           let proxySettings = parseProxyURL(httpProxy, type: "HTTP")
        {
            proxyConfig.merge(proxySettings) { _, new in new }
            hasProxyConfig = true
        }

        return hasProxyConfig ? proxyConfig : nil
        #else
        // Proxy configuration not available on iOS
        return nil
        #endif
    }

    /// Parse proxy URL and return configuration dictionary
    public static func parseProxyURL(_ proxyURLString: String, type: String) -> [String: Any]? {
        #if os(macOS)
        guard let proxyURL = URL(string: proxyURLString),
              let host = proxyURL.host,
              let port = proxyURL.port
        else {
            HubApi.logger.warning("Invalid \(type) proxy URL: \(proxyURLString)")
            return nil
        }

        let config: [String: Any]
        switch type {
        case "HTTPS":
            config = [
                kCFNetworkProxiesHTTPSEnable as String: true,
                kCFNetworkProxiesHTTPSProxy as String: host,
                kCFNetworkProxiesHTTPSPort as String: port,
            ]
        case "HTTP":
            config = [
                kCFNetworkProxiesHTTPEnable as String: true,
                kCFNetworkProxiesHTTPProxy as String: host,
                kCFNetworkProxiesHTTPPort as String: port,
            ]
        default:
            return nil
        }

        HubApi.logger.info("Configured \(type) proxy: \(host):\(port)")
        return config
        #else
        // Proxy configuration not available on iOS
        return nil
        #endif
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
        if let hfToken {
            request.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization")
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

    /// Throws error if page does not exist or is not accessible.
    /// Allows relative redirects but ignores absolute ones for LFS files.
    func httpHead(for url: URL) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        if let hfToken {
            request.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization")
        }
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")

        let redirectDelegate = RedirectDelegate()
        let session = URLSession(configuration: .default, delegate: redirectDelegate, delegateQueue: nil)

        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else { throw Hub.HubClientError.unexpectedError }

        switch response.statusCode {
        case 200..<400: break // Allow redirects to pass through to the redirect delegate
        case 401, 403: throw Hub.HubClientError.authorizationRequired
        case 404: throw Hub.HubClientError.fileNotFound(url.lastPathComponent)
        default: throw Hub.HubClientError.httpStatusCode(response.statusCode)
        }

        return (data, response)
    }

    func getFilenames(from repo: Repo, revision: String = "main", matching globs: [String] = []) async throws -> [String] {
        // Read repo info and only parse "siblings"
        let url = URL(string: "\(endpoint)/api/\(repo.type)/\(repo.id)/revision/\(revision)")!
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

    /// Reads metadata about a file in the local directory related to a download process.
    ///
    /// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/_local_folder.py#L263
    ///
    /// - Parameters:
    ///   - localDir: The local directory where metadata files are downloaded.
    ///   - filePath: The path of the file for which metadata is being read.
    /// - Throws: An `EnvironmentError.invalidMetadataError` if the metadata file is invalid and cannot be removed.
    /// - Returns: A `LocalDownloadFileMetadata` object if the metadata file exists and is valid, or `nil` if the file is missing or invalid.
    func readDownloadMetadata(metadataPath: URL) throws -> LocalDownloadFileMetadata? {
        if FileManager.default.fileExists(atPath: metadataPath.path) {
            do {
                let contents = try String(contentsOf: metadataPath, encoding: .utf8)
                let lines = contents.components(separatedBy: .newlines)

                guard lines.count >= 3 else {
                    throw EnvironmentError.invalidMetadataError(String(localized: "Metadata file is missing required fields"))
                }

                let commitHash = lines[0].trimmingCharacters(in: .whitespacesAndNewlines)
                let etag = lines[1].trimmingCharacters(in: .whitespacesAndNewlines)

                guard let timestamp = Double(lines[2].trimmingCharacters(in: .whitespacesAndNewlines)) else {
                    throw EnvironmentError.invalidMetadataError(String(localized: "Invalid timestamp format"))
                }

                let timestampDate = Date(timeIntervalSince1970: timestamp)
                let filename = metadataPath.lastPathComponent.replacingOccurrences(of: ".metadata", with: "")

                return LocalDownloadFileMetadata(commitHash: commitHash, etag: etag, filename: filename, timestamp: timestampDate)
            } catch let error as EnvironmentError {
                do {
                    HubApi.logger.warning("Invalid metadata file \(metadataPath): \(error.localizedDescription). Removing it from disk and continuing.")
                    try FileManager.default.removeItem(at: metadataPath)
                } catch {
                    throw EnvironmentError.invalidMetadataError(String(localized: "Could not remove corrupted metadata file: \(error.localizedDescription)"))
                }
                return nil
            } catch {
                do {
                    HubApi.logger.warning("Error reading metadata file \(metadataPath): \(error.localizedDescription). Removing it from disk and continuing.")
                    try FileManager.default.removeItem(at: metadataPath)
                } catch {
                    throw EnvironmentError.invalidMetadataError(String(localized: "Could not remove corrupted metadata file: \(error.localizedDescription)"))
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

    /// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/_local_folder.py#L391
    func writeDownloadMetadata(commitHash: String, etag: String, metadataPath: URL) throws {
        let metadataContent = "\(commitHash)\n\(etag)\n\(Date().timeIntervalSince1970)\n"
        do {
            try FileManager.default.createDirectory(at: metadataPath.deletingLastPathComponent(), withIntermediateDirectories: true)
            try metadataContent.write(to: metadataPath, atomically: true, encoding: .utf8)
        } catch {
            throw EnvironmentError.fileWriteError(String(localized: "Failed to write metadata to \(metadataPath.path): \(error.localizedDescription)"))
        }
    }

    struct HubFileDownloader {
        let hub: HubApi
        let repo: Repo
        let revision: String
        let repoDestination: URL
        let repoMetadataDestination: URL
        let relativeFilename: String
        let hfToken: String?
        let endpoint: String?
        let backgroundSession: Bool
        let proxyConfig: [String: Any]?

        var source: URL {
            // https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/tokenizer.json?download=true
            var url = URL(string: endpoint ?? "https://huggingface.co")!
            if repo.type != .models {
                url = url.appending(component: repo.type.rawValue)
            }
            url = url.appending(path: repo.id)
            url = url.appending(path: "resolve/\(revision)")
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
            if !FileManager.default.fileExists(atPath: incompleteDestination.path) {
                try "".write(to: incompleteDestination, atomically: true, encoding: .utf8)
            }
        }

        /// Note we go from Combine in Downloader to callback-based progress reporting
        /// We'll probably need to support Combine as well to play well with Swift UI
        /// (See for example PipelineLoader in swift-coreml-diffusers)
        @discardableResult
        func download(progressHandler: @escaping (Double, Double?) -> Void) async throws -> URL {
            let localMetadata = try hub.readDownloadMetadata(metadataPath: metadataDestination)
            let remoteMetadata = try await hub.getFileMetadata(url: source)

            let localCommitHash = localMetadata?.commitHash ?? ""
            let remoteCommitHash = remoteMetadata.commitHash ?? ""

            // Local file exists + metadata exists + commit_hash matches => return file
            if hub.isValidHash(hash: remoteCommitHash, pattern: hub.commitHashPattern), downloaded, localMetadata != nil,
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
                if hub.isValidHash(hash: remoteEtag, pattern: hub.sha256Pattern) {
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

            let downloader = Downloader(to: destination, incompleteDestination: incompleteDestination, inBackground: backgroundSession, proxyConfig: proxyConfig)

            try await withTaskCancellationHandler {
                let sub = await downloader.download(from: source, using: hfToken, expectedSize: remoteSize)
                listen: for await state in sub {
                    switch state {
                    case .notStarted:
                        continue
                    case let .downloading(progress, speed):
                        progressHandler(progress, speed)
                    case let .failed(error):
                        throw error
                    case .completed:
                        break listen
                    }
                }
            } onCancel: {
                Task {
                    await downloader.cancel()
                }
            }

            // Validate file size after download completion
            try HubApi.validateFileSize(at: destination, expectedSize: remoteSize)

            try hub.writeDownloadMetadata(commitHash: remoteCommitHash, etag: remoteEtag, metadataPath: metadataDestination)

            return destination
        }
    }

    @discardableResult
    func snapshot(from repo: Repo, revision: String = "main", matching globs: [String] = [], checkForUpdates: Bool = false, progressHandler: @escaping (Progress) -> Void = { _ in })
        async throws -> URL
    {
        let repoDestination = localRepoLocation(repo)
        let repoMetadataDestination =
            repoDestination
                .appendingPathComponent(".cache")
                .appendingPathComponent("huggingface")
                .appendingPathComponent("download")

        if await NetworkMonitor.shared.state.shouldUseOfflineMode() || useOfflineMode == true {
            if !FileManager.default.fileExists(atPath: repoDestination.path) {
                throw EnvironmentError.offlineModeError(String(localized: "Repository not available locally"))
            }

            let fileUrls = try FileManager.default.getFileUrls(at: repoDestination)
            if fileUrls.isEmpty {
                throw EnvironmentError.offlineModeError(String(localized: "No files available locally for this repository"))
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
                    throw EnvironmentError.offlineModeError(String(localized: "Metadata not available for \(fileUrl.lastPathComponent)"))
                }
                let localEtag = localMetadata.etag

                // LFS file so check file integrity
                if isValidHash(hash: localEtag, pattern: sha256Pattern) {
                    let fileHash = try computeFileHash(file: fileUrl)
                    if fileHash != localEtag {
                        throw EnvironmentError.fileIntegrityError(String(localized: "Hash mismatch for \(fileUrl.lastPathComponent)"))
                    }
                }
            }

            return repoDestination
        }

        // Check for upstream changes if requested and repository already exists
        if checkForUpdates, FileManager.default.fileExists(atPath: repoDestination.path) {
            HubApi.logger.info("Checking for upstream changes in \(repo.id)...")
            let changedFiles = try await redownloadChangedFiles(
                repo: repo,
                revision: revision,
                localDirectory: repoDestination,
                progressHandler: progressHandler
            )

            if !changedFiles.isEmpty {
                HubApi.logger.info("Updated \(changedFiles.count) files for \(repo.id)")
            }
        }

        var filenames = try await getFilenames(from: repo, revision: revision, matching: globs)

        // Filter to essential files if no specific globs provided
        if globs.isEmpty {
            filenames = filenames.filter { HubApi.isEssentialFile($0) }
        }
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
                hfToken: hfToken,
                endpoint: endpoint,
                backgroundSession: useBackgroundSession,
                proxyConfig: proxyConfig
            )

            // Retry download with exponential backoff
            var lastError: Error?
            let retryConfig = RetryConfig.default

            for attempt in 1...retryConfig.maxRetries {
                do {
                    try await downloader.download { fractionDownloaded, speed in
                        fileProgress.completedUnitCount = Int64(100 * fractionDownloaded)
                        if let speed {
                            fileProgress.setUserInfoObject(speed, forKey: .throughputKey)
                            progress.setUserInfoObject(speed, forKey: .throughputKey)
                        }
                        progressHandler(progress)
                    }
                    lastError = nil
                    break
                } catch {
                    lastError = error
                    HubApi.logger.warning("Download attempt \(attempt)/\(retryConfig.maxRetries) failed for \(filename): \(error.localizedDescription)")

                    if attempt < retryConfig.maxRetries {
                        let delay = retryConfig.delay(for: attempt)
                        HubApi.logger.info("Retrying download for \(filename) in \(String(format: "%.1f", delay)) seconds...")
                        try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    }
                }

                if Task.isCancelled {
                    return repoDestination
                }
            }

            // If all retries failed, throw the last error
            if let error = lastError {
                HubApi.logger.error("Failed to download \(filename) after \(retryConfig.maxRetries) attempts: \(error.localizedDescription)")
                throw error
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
    func snapshot(from repoId: String, revision: String = "main", matching globs: [String] = [], checkForUpdates: Bool = false, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await snapshot(from: Repo(id: repoId), revision: revision, matching: globs, checkForUpdates: checkForUpdates, progressHandler: progressHandler)
    }

    @discardableResult
    func snapshot(from repo: Repo, revision: String = "main", matching glob: String, checkForUpdates: Bool = false, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await snapshot(from: repo, revision: revision, matching: [glob], checkForUpdates: checkForUpdates, progressHandler: progressHandler)
    }

    @discardableResult
    func snapshot(from repoId: String, revision: String = "main", matching glob: String, checkForUpdates: Bool = false, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await snapshot(from: Repo(id: repoId), revision: revision, matching: [glob], checkForUpdates: checkForUpdates, progressHandler: progressHandler)
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
        let (_, response) = try await httpHead(for: url)
        let location = response.statusCode == 302 ? response.value(forHTTPHeaderField: "Location") : response.url?.absoluteString

        return FileMetadata(
            commitHash: response.value(forHTTPHeaderField: HFHttpHeaders.repoCommit),
            etag: normalizeEtag(
                (response.value(forHTTPHeaderField: HFHttpHeaders.linkedEtag)) ?? (response.value(forHTTPHeaderField: HFHttpHeaders.etag))
            ),
            location: location ?? url.absoluteString,
            size: Int(response.value(forHTTPHeaderField: HFHttpHeaders.linkedSize) ?? response.value(forHTTPHeaderField: HFHttpHeaders.contentLength) ?? ""),
            xetFileData: parseXetFileDataFromResponse(response: response, endpoint: endpoint)
        )
    }

    /// https://github.com/huggingface/huggingface_hub/blob/b698915d6b582c72806ac3e91c43bfd8dde35856/src/huggingface_hub/utils/_xet.py#L29
    private func parseXetFileDataFromResponse(
        response: HTTPURLResponse?,
        endpoint: String? = nil
    ) -> XetFileData? {
        guard let response else {
            return nil
        }

        guard let fileHash = response.allHeaderFields[HFHttpHeaders.xetHash] as? String else {
            return nil
        }

        guard var refreshRoute = response.getLinkURL(for: HFHttpHeaders.linkXetAuthKey)
            ?? response.allHeaderFields[HFHttpHeaders.xetRefreshRoute] as? String
        else {
            return nil
        }

        let endpoint = endpoint ?? "https://huggingface.co"

        let defaultEndpoint = "https://huggingface.co"

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
        let url = URL(string: "\(endpoint)/\(repo.id)/resolve/\(revision)")!
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
        public var isConnected: Bool = false
        public var isExpensive: Bool = false
        public var isConstrained: Bool = false

        func update(path: NWPath) {
            isConnected = path.status == .satisfied
            isExpensive = path.isExpensive
            isConstrained = path.isConstrained
        }

        func shouldUseOfflineMode() -> Bool {
            if ProcessInfo.processInfo.environment["CI_DISABLE_NETWORK_MONITOR"] == "1" {
                return false
            }
            return !isConnected || isExpensive || isConstrained
        }
    }

    private final class NetworkMonitor: Sendable {
        private let monitor: NWPathMonitor
        private let queue: DispatchQueue

        public let state: NetworkStateActor = .init()

        static let shared = NetworkMonitor()

        init() {
            monitor = NWPathMonitor()
            queue = DispatchQueue(label: "HubApi.NetworkMonitor")
            startMonitoring()
        }

        func startMonitoring() {
            monitor.pathUpdateHandler = { [weak self] path in
                guard let self else { return }
                Task {
                    await self.state.update(path: path)
                }
            }

            monitor.start(queue: queue)
        }

        func stopMonitoring() {
            monitor.cancel()
        }

        deinit {
            stopMonitoring()
        }
    }
}

/// Stateless wrappers that use `HubApi` instances
public extension Hub {
    static func getFilenames(from repo: Hub.Repo, matching globs: [String] = []) async throws -> [String] {
        try await HubApi.shared.getFilenames(from: repo, matching: globs)
    }

    static func getFilenames(from repoId: String, matching globs: [String] = []) async throws -> [String] {
        try await HubApi.shared.getFilenames(from: Repo(id: repoId), matching: globs)
    }

    static func getFilenames(from repo: Repo, matching glob: String) async throws -> [String] {
        try await HubApi.shared.getFilenames(from: repo, matching: glob)
    }

    static func getFilenames(from repoId: String, matching glob: String) async throws -> [String] {
        try await HubApi.shared.getFilenames(from: Repo(id: repoId), matching: glob)
    }

    static func snapshot(from repo: Repo, matching globs: [String] = [], checkForUpdates: Bool = false, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await HubApi.shared.snapshot(from: repo, matching: globs, checkForUpdates: checkForUpdates, progressHandler: progressHandler)
    }

    static func snapshot(from repoId: String, matching globs: [String] = [], checkForUpdates: Bool = false, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws
        -> URL
    {
        try await HubApi.shared.snapshot(from: Repo(id: repoId), matching: globs, checkForUpdates: checkForUpdates, progressHandler: progressHandler)
    }

    static func snapshot(from repo: Repo, matching glob: String, checkForUpdates: Bool = false, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await HubApi.shared.snapshot(from: repo, matching: glob, checkForUpdates: checkForUpdates, progressHandler: progressHandler)
    }

    static func snapshot(from repoId: String, matching glob: String, checkForUpdates: Bool = false, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await HubApi.shared.snapshot(from: Repo(id: repoId), matching: glob, checkForUpdates: checkForUpdates, progressHandler: progressHandler)
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

public extension [String] {
    func matching(glob: String) -> [String] {
        filter { fnmatch(glob, $0, 0) == 0 }
    }
}

public extension FileManager {
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

/// Advanced filtering capabilities
public extension HubApi {
    /// Model filter for advanced repository searching
    struct ModelFilter {
        public var author: String?
        public var library: [String]?
        public var language: [String]?
        public var modelName: String?
        public var task: [String]?
        public var tags: [String]?
        public var trainedDataset: [String]?

        public init(
            author: String? = nil,
            library: [String]? = nil,
            language: [String]? = nil,
            modelName: String? = nil,
            task: [String]? = nil,
            tags: [String]? = nil,
            trainedDataset: [String]? = nil
        ) {
            self.author = author
            self.library = library
            self.language = language
            self.modelName = modelName
            self.task = task
            self.tags = tags
            self.trainedDataset = trainedDataset
        }

        /// Convert filter to query parameters
        var queryParameters: [String: String] {
            var params: [String: String] = [:]

            if let author { params["author"] = author }
            if let library { params["library"] = library.joined(separator: ",") }
            if let language { params["language"] = language.joined(separator: ",") }
            if let modelName { params["model_name"] = modelName }
            if let task { params["task"] = task.joined(separator: ",") }
            if let tags { params["tags"] = tags.joined(separator: ",") }
            if let trainedDataset { params["dataset"] = trainedDataset.joined(separator: ",") }

            return params
        }
    }

    /// Dataset filter for advanced repository searching
    struct DatasetFilter {
        public var author: String?
        public var benchmark: [String]?
        public var datasetName: String?
        public var languageCreators: [String]?
        public var languages: [String]?
        public var multilinguality: [String]?
        public var sizeCategories: [String]?
        public var taskCategories: [String]?
        public var taskIds: [String]?

        public init(
            author: String? = nil,
            benchmark: [String]? = nil,
            datasetName: String? = nil,
            languageCreators: [String]? = nil,
            languages: [String]? = nil,
            multilinguality: [String]? = nil,
            sizeCategories: [String]? = nil,
            taskCategories: [String]? = nil,
            taskIds: [String]? = nil
        ) {
            self.author = author
            self.benchmark = benchmark
            self.datasetName = datasetName
            self.languageCreators = languageCreators
            self.languages = languages
            self.multilinguality = multilinguality
            self.sizeCategories = sizeCategories
            self.taskCategories = taskCategories
            self.taskIds = taskIds
        }

        /// Convert filter to query parameters
        var queryParameters: [String: String] {
            var params: [String: String] = [:]

            if let author { params["author"] = author }
            if let benchmark { params["benchmark"] = benchmark.joined(separator: ",") }
            if let datasetName { params["dataset_name"] = datasetName }
            if let languageCreators { params["language_creators"] = languageCreators.joined(separator: ",") }
            if let languages { params["languages"] = languages.joined(separator: ",") }
            if let multilinguality { params["multilinguality"] = multilinguality.joined(separator: ",") }
            if let sizeCategories { params["size_categories"] = sizeCategories.joined(separator: ",") }
            if let taskCategories { params["task_categories"] = taskCategories.joined(separator: ",") }
            if let taskIds { params["task_ids"] = taskIds.joined(separator: ",") }

            return params
        }
    }

    /// Search models with advanced filtering
    func searchModels(filter: ModelFilter, limit: Int = 10) async throws -> [Config] {
        let url = URL(string: "\(endpoint)/api/models")!
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!

        var queryItems = [URLQueryItem]()
        for (key, value) in filter.queryParameters {
            queryItems.append(URLQueryItem(name: key, value: value))
        }
        queryItems.append(URLQueryItem(name: "limit", value: String(limit)))
        components.queryItems = queryItems

        let (data, _) = try await httpGet(for: components.url!)
        let models = try JSONSerialization.jsonObject(with: data, options: []) as? [[String: Any]] ?? []

        return models.map { Config($0 as [NSString: Any]) }
    }

    /// Search datasets with advanced filtering
    func searchDatasets(filter: DatasetFilter, limit: Int = 10) async throws -> [Config] {
        let url = URL(string: "\(endpoint)/api/datasets")!
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!

        var queryItems = [URLQueryItem]()
        for (key, value) in filter.queryParameters {
            queryItems.append(URLQueryItem(name: key, value: value))
        }
        queryItems.append(URLQueryItem(name: "limit", value: String(limit)))
        components.queryItems = queryItems

        let (data, _) = try await httpGet(for: components.url!)
        let datasets = try JSONSerialization.jsonObject(with: data, options: []) as? [[String: Any]] ?? []

        return datasets.map { Config($0 as [NSString: Any]) }
    }

    /// Get model tags for filtering
    func getModelTags() async throws -> Config {
        let url = URL(string: "\(endpoint)/api/models-tags-by-type")!
        let (data, _) = try await httpGet(for: url)
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [NSString: Any] else { throw Hub.HubClientError.parse }
        return Config(dictionary)
    }

    /// Get dataset tags for filtering
    func getDatasetTags() async throws -> Config {
        let url = URL(string: "\(endpoint)/api/datasets-tags-by-type")!
        let (data, _) = try await httpGet(for: url)
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [NSString: Any] else { throw Hub.HubClientError.parse }
        return Config(dictionary)
    }
}

/// File validation utilities
public extension HubApi {
    /// Check if a file is essential for model operation
    static func isEssentialFile(_ path: String) -> Bool {
        path.hasSuffix(".json") || path.hasSuffix(".txt") || path == "config.json"
    }

    /// Validate downloaded file size matches expected size
    static func validateFileSize(at fileURL: URL, expectedSize: Int?) throws {
        guard let expectedSize else { return }

        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
            if let fileSize = attributes[.size] as? Int64 {
                if fileSize != expectedSize {
                    throw EnvironmentError.fileIntegrityError(
                        String(localized: "File size mismatch for \(fileURL.lastPathComponent): got \(fileSize), expected \(expectedSize)")
                    )
                }
            }
        } catch let error as EnvironmentError {
            throw error
        } catch {
            throw EnvironmentError.fileIntegrityError(
                String(localized: "Failed to validate file size for \(fileURL.lastPathComponent): \(error.localizedDescription)")
            )
        }
    }

    /// Format bytes for display
    static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }

    /// Check if upstream files have changed and need re-downloading
    func checkForUpstreamChanges(repo: Repo, revision: String = "main", localDirectory: URL) async throws -> [String] {
        var changedFiles: [String] = []

        // Get list of local files
        guard FileManager.default.fileExists(atPath: localDirectory.path) else {
            return changedFiles
        }

        let localFileUrls = try FileManager.default.getFileUrls(at: localDirectory)

        // Check each local file against upstream
        for localFileUrl in localFileUrls {
            let relativePath = localFileUrl.path.replacingOccurrences(of: localDirectory.path + "/", with: "")

            // Skip metadata files and directories
            if relativePath.hasSuffix(".metadata") || localFileUrl.hasDirectoryPath {
                continue
            }

            // Get local metadata
            let metadataPath = URL(
                fileURLWithPath: localFileUrl.path.replacingOccurrences(
                    of: localDirectory.path,
                    with: localDirectory.appendingPathComponent(".cache/huggingface/download").path
                ) + ".metadata"
            )

            let localMetadata = try readDownloadMetadata(metadataPath: metadataPath)

            // Get remote metadata
            let remoteUrl = URL(string: "\(endpoint)/\(repo.id)/resolve/\(revision)/\(relativePath)")!
            let remoteMetadata = try await getFileMetadata(url: remoteUrl)

            // Check if file has changed
            if let localEtag = localMetadata?.etag,
               let remoteEtag = remoteMetadata.etag,
               localEtag != remoteEtag
            {
                changedFiles.append(relativePath)
                HubApi.logger.info("Upstream change detected for \(relativePath): local etag \(localEtag) != remote etag \(remoteEtag)")
            } else if localMetadata == nil {
                // No local metadata - file needs to be downloaded
                changedFiles.append(relativePath)
                HubApi.logger.info("No metadata found for \(relativePath), marking for download")
            }
        }

        return changedFiles
    }

    /// Retry configuration for network operations
    struct RetryConfig: Sendable {
        let maxRetries: Int
        let baseDelay: TimeInterval
        let maxDelay: TimeInterval

        static let `default` = RetryConfig(maxRetries: 3, baseDelay: 1.0, maxDelay: 30.0)

        func delay(for attempt: Int) -> TimeInterval {
            let exponentialDelay = baseDelay * pow(2.0, Double(attempt - 1))
            return min(exponentialDelay, maxDelay)
        }
    }

    /// Force re-download of files that have changed upstream
    @discardableResult
    func redownloadChangedFiles(
        repo: Repo,
        revision: String = "main",
        localDirectory: URL,
        retryConfig: RetryConfig? = nil,
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> [String] {
        let changedFiles = try await checkForUpstreamChanges(repo: repo, revision: revision, localDirectory: localDirectory)

        guard !changedFiles.isEmpty else {
            HubApi.logger.info("No upstream changes detected for \(repo.id)")
            return []
        }

        HubApi.logger.info("Re-downloading \(changedFiles.count) changed files for \(repo.id)")

        let config = retryConfig ?? RetryConfig.default
        let progress = Progress(totalUnitCount: Int64(changedFiles.count))
        var downloadedFiles: [String] = []

        for filename in changedFiles {
            let fileProgress = Progress(totalUnitCount: 100, parent: progress, pendingUnitCount: 1)

            let downloader = HubFileDownloader(
                hub: self,
                repo: repo,
                revision: revision,
                repoDestination: localDirectory,
                repoMetadataDestination: localDirectory.appendingPathComponent(".cache/huggingface/download"),
                relativeFilename: filename,
                hfToken: hfToken,
                endpoint: endpoint,
                backgroundSession: useBackgroundSession,
                proxyConfig: proxyConfig
            )

            // Retry download with exponential backoff
            var lastError: Error?
            for attempt in 1...config.maxRetries {
                do {
                    try await downloader.download { fractionDownloaded, speed in
                        fileProgress.completedUnitCount = Int64(100 * fractionDownloaded)
                        if let speed {
                            fileProgress.setUserInfoObject(speed, forKey: .throughputKey)
                            progress.setUserInfoObject(speed, forKey: .throughputKey)
                        }
                        progressHandler(progress)
                    }
                    downloadedFiles.append(filename)
                    fileProgress.completedUnitCount = 100
                    lastError = nil
                    break
                } catch {
                    lastError = error
                    HubApi.logger.warning("Download attempt \(attempt)/\(config.maxRetries) failed for \(filename): \(error.localizedDescription)")

                    if attempt < config.maxRetries {
                        let delay = config.delay(for: attempt)
                        HubApi.logger.info("Retrying download for \(filename) in \(String(format: "%.1f", delay)) seconds...")
                        try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    }
                }

                if Task.isCancelled {
                    break
                }
            }

            // If all retries failed, throw the last error
            if let error = lastError {
                HubApi.logger.error("Failed to download \(filename) after \(config.maxRetries) attempts: \(error.localizedDescription)")
                throw error
            }

            if Task.isCancelled {
                break
            }
        }

        progressHandler(progress)
        HubApi.logger.info("Re-downloaded \(downloadedFiles.count) files for \(repo.id)")
        return downloadedFiles
    }

    /// Clean up corrupted or incomplete downloads
    func cleanupCorruptedDownloads(repo: Repo, localDirectory: URL) throws {
        let metadataDir = localDirectory.appendingPathComponent(".cache/huggingface/download")

        guard FileManager.default.fileExists(atPath: metadataDir.path) else {
            return
        }

        let localFileUrls = try FileManager.default.getFileUrls(at: localDirectory)
        var corruptedFiles: [String] = []

        for localFileUrl in localFileUrls {
            let relativePath = localFileUrl.path.replacingOccurrences(of: localDirectory.path + "/", with: "")

            // Skip directories and metadata files
            if localFileUrl.hasDirectoryPath || relativePath.hasSuffix(".metadata") {
                continue
            }

            // Check if metadata exists for this file
            let metadataPath = metadataDir.appendingPathComponent(relativePath + ".metadata")

            if !FileManager.default.fileExists(atPath: metadataPath.path) {
                corruptedFiles.append(relativePath)
                HubApi.logger.info("Found file without metadata: \(relativePath)")
                continue
            }

            // Validate file integrity if possible
            let localMetadata = try readDownloadMetadata(metadataPath: metadataPath)
            if let localEtag = localMetadata?.etag, isValidHash(hash: localEtag, pattern: sha256Pattern) {
                // This is an LFS file, check hash
                let fileHash = try computeFileHash(file: localFileUrl)
                if fileHash != localEtag {
                    corruptedFiles.append(relativePath)
                    HubApi.logger.info("Hash mismatch for \(relativePath): expected \(localEtag), got \(fileHash)")
                }
            }
        }

        // Remove corrupted files and their metadata
        for corruptedFile in corruptedFiles {
            let filePath = localDirectory.appendingPathComponent(corruptedFile)
            let metadataPath = metadataDir.appendingPathComponent(corruptedFile + ".metadata")

            try? FileManager.default.removeItem(at: filePath)
            try? FileManager.default.removeItem(at: metadataPath)

            HubApi.logger.info("Cleaned up corrupted file: \(corruptedFile)")
        }

        if !corruptedFiles.isEmpty {
            HubApi.logger.info("Cleaned up \(corruptedFiles.count) corrupted files for \(repo.id)")
        }
    }

    /// Get repository information for different repo types
    func getRepositoryInfo(repo: Repo, revision: String = "main") async throws -> Config {
        let url = URL(string: "\(endpoint)/api/\(repo.type)/\(repo.id)/revision/\(revision)")!
        let (data, _) = try await httpGet(for: url)
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [NSString: Any] else { throw Hub.HubClientError.parse }
        return Config(dictionary)
    }

    /// List repositories of a specific type with filtering
    func listRepositories(
        type: RepoType,
        filter: ModelFilter? = nil,
        limit: Int = 10
    ) async throws -> [Config] {
        let url = URL(string: "\(endpoint)/api/\(type.rawValue)")!
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!

        var queryItems = [URLQueryItem]()
        if let filter {
            for (key, value) in filter.queryParameters {
                queryItems.append(URLQueryItem(name: key, value: value))
            }
        }
        queryItems.append(URLQueryItem(name: "limit", value: String(limit)))
        components.queryItems = queryItems

        let (data, _) = try await httpGet(for: components.url!)
        let repos = try JSONSerialization.jsonObject(with: data, options: []) as? [[String: Any]] ?? []

        return repos.map { Config($0 as [NSString: Any]) }
    }

    /// Check if repository exists and is accessible
    func repositoryExists(repo: Repo, revision: String = "main") async throws -> Bool {
        do {
            _ = try await getRepositoryInfo(repo: repo, revision: revision)
            return true
        } catch Hub.HubClientError.fileNotFound {
            return false
        } catch {
            throw error
        }
    }

    /// Get repository size information
    func getRepositorySize(repo: Repo, revision: String = "main") async throws -> Int64 {
        let files = try await getFilenames(from: repo, revision: revision)
        var totalSize: Int64 = 0

        for filename in files {
            let metadata = try await getFileMetadata(from: repo, revision: revision, matching: [filename])
            if let size = metadata.first?.size {
                totalSize += Int64(size)
            }
        }

        return totalSize
    }

    /// Create a repository (if supported by API)
    func createRepository(
        repo: Repo,
        private: Bool = false,
        description: String? = nil
    ) async throws -> Config {
        let url = URL(string: "\(endpoint)/api/repos/create")!

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let hfToken {
            request.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization")
        }

        var body: [String: Any] = [
            "name": repo.id,
            "type": repo.type.rawValue,
            "private": `private`,
        ]

        if let description {
            body["description"] = description
        }

        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw Hub.HubClientError.unexpectedError
        }

        switch httpResponse.statusCode {
        case 200..<300:
            break // Success
        case 401, 403:
            throw Hub.HubClientError.authorizationRequired
        case 404:
            throw Hub.HubClientError.fileNotFound("Repository creation endpoint")
        default:
            throw Hub.HubClientError.httpStatusCode(httpResponse.statusCode)
        }
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [NSString: Any] else { throw Hub.HubClientError.parse }
        return Config(dictionary)
    }

    /// Recover from a failed download state
    func recoverFromFailedDownload(repo: Repo, localDirectory: URL) async throws {
        HubApi.logger.info("Attempting to recover from failed download state for \(repo.id)")

        // Clean up corrupted files first
        try cleanupCorruptedDownloads(repo: repo, localDirectory: localDirectory)

        // Re-download any missing essential files
        let essentialFiles = try await getFilenames(from: repo).filter { HubApi.isEssentialFile($0) }

        for filename in essentialFiles {
            let filePath = localDirectory.appendingPathComponent(filename)
            if !FileManager.default.fileExists(atPath: filePath.path) {
                HubApi.logger.info("Re-downloading missing essential file: \(filename)")

                let downloader = HubFileDownloader(
                    hub: self,
                    repo: repo,
                    revision: "main",
                    repoDestination: localDirectory,
                    repoMetadataDestination: localDirectory.appendingPathComponent(".cache/huggingface/download"),
                    relativeFilename: filename,
                    hfToken: hfToken,
                    endpoint: endpoint,
                    backgroundSession: useBackgroundSession,
                    proxyConfig: proxyConfig
                )

                try await downloader.download { _, _ in }
            }
        }

        HubApi.logger.info("Recovery completed for \(repo.id)")
    }
}
