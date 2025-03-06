//
//  Downloader.swift
//
//  Adapted from https://github.com/huggingface/swift-coreml-diffusers/blob/d041577b9f5e201baa3465bc60bc5d0a1cf7ed7f/Diffusion/Common/Downloader.swift
//  Created by Pedro Cuenca on December 2022.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//

import Foundation
import Combine

class Downloader: NSObject, ObservableObject {
    private(set) var destination: URL
    private(set) var sourceURL: URL

    private let chunkSize = 10 * 1024 * 1024  // 10MB

    enum DownloadState {
        case notStarted
        case downloading(Double)
        case completed(URL)
        case failed(Error)
    }

    enum DownloadError: Error {
        case invalidDownloadLocation
        case unexpectedError
        case tempFileNotFound
    }

    private(set) lazy var downloadState: CurrentValueSubject<DownloadState, Never> = CurrentValueSubject(.notStarted)
    private var stateSubscriber: Cancellable?
    
    private(set) var tempFilePath: URL?
    private(set) var expectedSize: Int?
    private(set) var downloadedSize: Int = 0

    private var urlSession: URLSession? = nil

    init(
        from url: URL,
        to destination: URL,
        using authToken: String? = nil,
        inBackground: Bool = false,
        resumeSize: Int = 0,
        headers: [String: String]? = nil,
        expectedSize: Int? = nil,
        timeout: TimeInterval = 10,
        numRetries: Int = 5,
        existingTempFile: URL? = nil
    ) {
        self.destination = destination
        self.sourceURL = url
        self.expectedSize = expectedSize
        self.downloadedSize = resumeSize
        self.tempFilePath = existingTempFile
        
        super.init()
        let sessionIdentifier = "swift-transformers.hub.downloader"

        var config = URLSessionConfiguration.default
        if inBackground {
            config = URLSessionConfiguration.background(withIdentifier: sessionIdentifier)
            config.isDiscretionary = false
            config.sessionSendsLaunchEvents = true
        }

        self.urlSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)

        setupDownload(from: url, with: authToken, resumeSize: resumeSize, headers: headers, expectedSize: expectedSize, timeout: timeout, numRetries: numRetries)
    }

    /// Sets up and initiates a file download operation
    ///
    /// - Parameters:
    ///   - url: Source URL to download from
    ///   - authToken: Bearer token for authentication with Hugging Face
    ///   - resumeSize: Number of bytes already downloaded for resuming interrupted downloads
    ///   - headers: Additional HTTP headers to include in the request
    ///   - expectedSize: Expected file size in bytes for validation
    ///   - timeout: Time interval before the request times out
    ///   - numRetries: Number of retry attempts for failed downloads
    private func setupDownload(
        from url: URL,
        with authToken: String?,
        resumeSize: Int,
        headers: [String: String]?,
        expectedSize: Int?,
        timeout: TimeInterval,
        numRetries: Int
    ) {
        print("[Downloader] Setting up download for \(url.lastPathComponent)")
        print("[Downloader] Destination: \(destination.path)")
        print("[Downloader] Temp file: \(tempFilePath?.path ?? "none")")
        
        // If we have an expected size and resumeSize, calculate initial progress
        if let expectedSize = expectedSize, expectedSize > 0 && resumeSize > 0 {
            let initialProgress = Double(resumeSize) / Double(expectedSize)
            downloadState.value = .downloading(initialProgress)
            print("[Downloader] Resuming from \(resumeSize)/\(expectedSize) bytes (\(Int(initialProgress * 100))%)")
        } else {
            downloadState.value = .downloading(0)
            print("[Downloader] Starting new download")
        }
        
        urlSession?.getAllTasks { tasks in
            // If there's an existing pending background task with the same URL, let it proceed.
            if let existing = tasks.filter({ $0.originalRequest?.url == url }).first {
                switch existing.state {
                case .running:
                    // print("Already downloading \(url)")
                    return
                case .suspended:
                    // print("Resuming suspended download task for \(url)")
                    existing.resume()
                    return
                case .canceling:
                    // print("Starting new download task for \(url), previous was canceling")
                    break
                case .completed:
                    // print("Starting new download task for \(url), previous is complete but the file is no longer present (I think it's cached)")
                    break
                @unknown default:
                    // print("Unknown state for running task; cancelling and creating a new one")
                    existing.cancel()
                }
            }
            var request = URLRequest(url: url)
            
            // Use headers from argument else create an empty header dictionary
            var requestHeaders = headers ?? [:]
            
            // Populate header auth and range fields
            if let authToken = authToken {
                requestHeaders["Authorization"] = "Bearer \(authToken)"
            }
            if resumeSize > 0 {
                requestHeaders["Range"] = "bytes=\(resumeSize)-"
            }
            
            request.timeoutInterval = timeout
            request.allHTTPHeaderFields = requestHeaders

            Task {
                do {
                    // Create or use existing temp file
                    let tempURL: URL
                    var existingSize = 0
                    
                    if let existingTempFile = self.tempFilePath, FileManager.default.fileExists(atPath: existingTempFile.path) {
                        tempURL = existingTempFile
                        let attributes = try FileManager.default.attributesOfItem(atPath: tempURL.path)
                        existingSize = attributes[.size] as? Int ?? 0
                        // If the reported resumeSize doesn't match the file size, trust the file size
                        if existingSize != resumeSize {
                            self.downloadedSize = existingSize
                            print("[Downloader] Found existing temp file with \(existingSize) bytes (different from resumeSize: \(resumeSize))")
                        } else {
                            print("[Downloader] Found existing temp file with \(existingSize) bytes")
                        }
                    } else {
                        // Create new temp file with predictable path for future resume
                        let filename = url.lastPathComponent
                        // Create a stable hash by extracting just the path component
                        let urlPath = url.absoluteString
                        // Use a deterministic hash that doesn't change between app launches
                        let stableHash = abs(urlPath.data(using: .utf8)!.reduce(5381) {
                            ($0 << 5) &+ $0 &+ Int32($1)
                        })
                        let hashedName = "\(filename)-\(stableHash)"
                        tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(hashedName)
                        print("[Downloader] Creating new temp file at \(tempURL.path)")
                        FileManager.default.createFile(atPath: tempURL.path, contents: nil)
                    }
                    
                    self.tempFilePath = tempURL
                    let tempFile = try FileHandle(forWritingTo: tempURL)
                    
                    // If we're resuming, seek to end of file first
                    if existingSize > 0 {
                        print("[Downloader] Seeking to end of existing file (\(existingSize) bytes)")
                        try tempFile.seekToEnd()
                    }
                    
                    defer { tempFile.closeFile() }
                    try await self.httpGet(request: request, tempFile: tempFile, resumeSize: self.downloadedSize, numRetries: numRetries, expectedSize: expectedSize)
                    
                    // Clean up and move the completed download to its final destination
                    tempFile.closeFile()
                    print("[Downloader] Download completed with total size \(self.downloadedSize) bytes")
                    print("[Downloader] Moving temp file to destination: \(self.destination.path)")
                    try FileManager.default.moveDownloadedFile(from: tempURL, to: self.destination)
                    
                    // Clear temp file reference since it's been moved
                    self.tempFilePath = nil
                    print("[Downloader] Download successfully completed")
                    self.downloadState.value = .completed(self.destination)
                } catch {
                    print("[Downloader] Error: \(error)")
                    self.downloadState.value = .failed(error)
                }
            }
        }
    }

    /// Downloads a file from given URL using chunked transfer and handles retries.
    ///
    /// Reference: https://github.com/huggingface/huggingface_hub/blob/418a6ffce7881f5c571b2362ed1c23ef8e4d7d20/src/huggingface_hub/file_download.py#L306
    ///
    /// - Parameters:
    ///   - request: The URLRequest for the file to download
    ///   - resumeSize: The number of bytes already downloaded. If set to 0 (default), the whole file is download. If set to a positive number, the download will resume at the given position
    ///   - numRetries: The number of retry attempts remaining for failed downloads
    ///   - expectedSize: The expected size of the file to download. If set, the download will raise an error if the size of the received content is different from the expected one.
    /// - Throws: `DownloadError.unexpectedError` if the response is invalid or file size mismatch occurs
    ///           `URLError` if the download fails after all retries are exhausted
    private func httpGet(
        request: URLRequest,
        tempFile: FileHandle,
        resumeSize: Int,
        numRetries: Int,
        expectedSize: Int?
    ) async throws {
        guard let session = self.urlSession else {
            throw DownloadError.unexpectedError
        }
        
        // Create a new request with Range header for resuming
        var newRequest = request
        if resumeSize > 0 {
            newRequest.setValue("bytes=\(resumeSize)-", forHTTPHeaderField: "Range")
            print("[Downloader] Adding Range header: bytes=\(resumeSize)-")
        }
        
        // Start the download and get the byte stream
        let (asyncBytes, response) = try await session.bytes(for: newRequest)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            print("[Downloader] Error: Non-HTTP response received")
            throw DownloadError.unexpectedError
        }
        
        print("[Downloader] Received HTTP \(httpResponse.statusCode) response")
        if let contentRange = httpResponse.value(forHTTPHeaderField: "Content-Range") {
            print("[Downloader] Content-Range: \(contentRange)")
        }
        if let contentLength = httpResponse.value(forHTTPHeaderField: "Content-Length") {
            print("[Downloader] Content-Length: \(contentLength)")
        }
                
        guard (200..<300).contains(httpResponse.statusCode) else {
            print("[Downloader] Error: HTTP status code \(httpResponse.statusCode)")
            throw DownloadError.unexpectedError
        }

        self.downloadedSize = resumeSize
        
        // Create a buffer to collect bytes before writing to disk
        var buffer = Data(capacity: chunkSize)
        
        var newNumRetries = numRetries
        do {
            for try await byte in asyncBytes {
                buffer.append(byte)
                // When buffer is full, write to disk
                if buffer.count == chunkSize {
                    if !buffer.isEmpty { // Filter out keep-alive chunks
                        try tempFile.write(contentsOf: buffer)
                        buffer.removeAll(keepingCapacity: true)
                        self.downloadedSize += chunkSize
                        newNumRetries = 5
                        guard let expectedSize = expectedSize else { continue }
                        let progress = expectedSize != 0 ? Double(self.downloadedSize) / Double(expectedSize) : 0
                        downloadState.value = .downloading(progress)
                    }
                }
            }
            
            if !buffer.isEmpty {
                try tempFile.write(contentsOf: buffer)
                self.downloadedSize += buffer.count
                buffer.removeAll(keepingCapacity: true)
                newNumRetries = 5
            }
        } catch let error as URLError {
            if newNumRetries <= 0 {
                throw error
            }
            try await Task.sleep(nanoseconds: 1_000_000_000)
            
            let config = URLSessionConfiguration.default
            self.urlSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)
            
            try await httpGet(
                request: request,
                tempFile: tempFile,
                resumeSize: self.downloadedSize,
                numRetries: newNumRetries - 1,
                expectedSize: expectedSize
            )
        }
        
        // Verify the downloaded file size matches the expected size
        let actualSize = try tempFile.seekToEnd()
        if let expectedSize = expectedSize, expectedSize != actualSize {
            print("[Downloader] Error: Size mismatch - expected \(expectedSize) bytes but got \(actualSize) bytes")
            throw DownloadError.unexpectedError
        } else {
            print("[Downloader] Final verification passed, size: \(actualSize) bytes")
        }
    }
    
    @discardableResult
    func waitUntilDone() throws -> URL {
        // It's either this, or stream the bytes ourselves (add to a buffer, save to disk, etc; boring and finicky)
        let semaphore = DispatchSemaphore(value: 0)
        stateSubscriber = downloadState.sink { state in
            switch state {
            case .completed: semaphore.signal()
            case .failed:    semaphore.signal()
            default:         break
            }
        }
        semaphore.wait()

        switch downloadState.value {
        case .completed(let url): return url
        case .failed(let error):  throw error
        default:                  throw DownloadError.unexpectedError
        }
    }

    func cancel() {
        urlSession?.invalidateAndCancel()
    }
}

extension Downloader: URLSessionDownloadDelegate {
    func urlSession(_: URLSession, downloadTask: URLSessionDownloadTask, didWriteData _: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        downloadState.value = .downloading(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }

    func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        do {
            // If the downloaded file already exists on the filesystem, overwrite it
            try FileManager.default.moveDownloadedFile(from: location, to: self.destination)
            downloadState.value = .completed(destination)
        } catch {
            downloadState.value = .failed(error)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            downloadState.value = .failed(error)
//        } else if let response = task.response as? HTTPURLResponse {
//            print("HTTP response status code: \(response.statusCode)")
//            let headers = response.allHeaderFields
//            print("HTTP response headers: \(headers)")
        }
    }
}

extension FileManager {
    func moveDownloadedFile(from srcURL: URL, to dstURL: URL) throws {
        if fileExists(atPath: dstURL.path) {
            try removeItem(at: dstURL)
        }
        try moveItem(at: srcURL, to: dstURL)
    }
}

/// Structs for persisting download state
public struct PersistableDownloadState: Codable {
    let sourceURL: URL
    let destinationURL: URL
    let tempFilePath: URL
    let downloadedSize: Int
    let expectedSize: Int?
    
    init(downloader: Downloader) {
        self.sourceURL = downloader.sourceURL
        self.destinationURL = downloader.destination
        self.tempFilePath = downloader.tempFilePath ?? FileManager.default.temporaryDirectory.appendingPathComponent("unknown")
        self.downloadedSize = downloader.downloadedSize
        self.expectedSize = downloader.expectedSize
    }
}

/// Extension for managing persisted download states
extension Downloader {
    /// Persists the current download state to UserDefaults
    func persistState() {
        guard let tempFilePath = self.tempFilePath else {
            print("[Downloader] Cannot persist state: No temp file path")
            return // Nothing to persist if no temp file
        }
        
        let state = PersistableDownloadState(downloader: self)
        
        do {
            let encoder = JSONEncoder()
            let data = try encoder.encode(state)
            
            // Store in UserDefaults
            var states = Downloader.getPersistedStates()
            states[sourceURL.absoluteString] = data
            UserDefaults.standard.set(states, forKey: "SwiftTransformers.ActiveDownloads")
            print("[Downloader] Persisted download state for \(sourceURL.lastPathComponent) - \(downloadedSize) bytes downloaded")
        } catch {
            print("[Downloader] Error persisting download state: \(error)")
        }
    }
    
    /// Removes this download from persisted states
    func removePersistedState() {
        var states = Downloader.getPersistedStates()
        states.removeValue(forKey: sourceURL.absoluteString)
        UserDefaults.standard.set(states, forKey: "SwiftTransformers.ActiveDownloads")
        print("[Downloader] Removed persisted state for \(sourceURL.lastPathComponent)")
    }
    
    /// Get all persisted download states
    static func getPersistedStates() -> [String: Data] {
        return UserDefaults.standard.dictionary(forKey: "SwiftTransformers.ActiveDownloads") as? [String: Data] ?? [:]
    }
    
    /// Resume all persisted downloads
    static func resumeAllPersistedDownloads(authToken: String? = nil) -> [Downloader] {
        let states = getPersistedStates()
        let decoder = JSONDecoder()
        
        print("[Downloader] Found \(states.count) persisted download states")
        var resumedDownloaders: [Downloader] = []
        
        for (url, stateData) in states {
            do {
                let state = try decoder.decode(PersistableDownloadState.self, from: stateData)
                print("[Downloader] Trying to resume download for: \(state.sourceURL.lastPathComponent)")
                
                // Check if temp file still exists
                if FileManager.default.fileExists(atPath: state.tempFilePath.path) {
                    let attributes = try FileManager.default.attributesOfItem(atPath: state.tempFilePath.path)
                    let fileSize = attributes[.size] as? Int ?? 0
                    print("[Downloader] Found existing temp file with \(fileSize) bytes")
                    
                    // Create a new downloader that resumes from the temp file
                    let downloader = Downloader(
                        from: state.sourceURL,
                        to: state.destinationURL,
                        using: authToken,
                        resumeSize: fileSize,
                        expectedSize: state.expectedSize,
                        existingTempFile: state.tempFilePath
                    )
                    
                    resumedDownloaders.append(downloader)
                    print("[Downloader] Successfully resumed download for \(state.sourceURL.lastPathComponent)")
                } else {
                    print("[Downloader] Temp file not found at \(state.tempFilePath.path)")
                }
            } catch {
                print("[Downloader] Error restoring download for \(url): \(error)")
            }
        }
        
        print("[Downloader] Successfully resumed \(resumedDownloaders.count) downloads")
        return resumedDownloaders
    }
}
