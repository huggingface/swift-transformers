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
    }

    private(set) lazy var downloadState: CurrentValueSubject<DownloadState, Never> = CurrentValueSubject(.notStarted)
    private var stateSubscriber: Cancellable?

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
        numRetries: Int = 5
    ) {
        self.destination = destination
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
        downloadState.value = .downloading(0)
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
                    // Create a temp file to write
                    let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                    FileManager.default.createFile(atPath: tempURL.path, contents: nil)
                    let tempFile = try FileHandle(forWritingTo: tempURL)
                    
                    defer { tempFile.closeFile() }
                    try await self.httpGet(request: request, tempFile: tempFile, resumeSize: resumeSize, numRetries: numRetries, expectedSize: expectedSize)
                    
                    // Clean up and move the completed download to its final destination
                    tempFile.closeFile()
                    try FileManager.default.moveDownloadedFile(from: tempURL, to: self.destination)
                    
                    self.downloadState.value = .completed(self.destination)
                } catch {
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
        }
        
        // Start the download and get the byte stream
        let (asyncBytes, response) = try await session.bytes(for: newRequest)
        
        guard let response = response as? HTTPURLResponse else {
            throw DownloadError.unexpectedError
        }
                
        guard (200..<300).contains(response.statusCode) else {
            throw DownloadError.unexpectedError
        }

        var downloadedSize = resumeSize
        
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
                        downloadedSize += chunkSize
                        newNumRetries = 5
                        guard let expectedSize = expectedSize else { continue }
                        let progress = expectedSize != 0 ? Double(downloadedSize) / Double(expectedSize) : 0
                        downloadState.value = .downloading(progress)
                    }
                }
            }
            
            if !buffer.isEmpty {
                try tempFile.write(contentsOf: buffer)
                downloadedSize += buffer.count
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
                resumeSize: downloadedSize,
                numRetries: newNumRetries - 1,
                expectedSize: expectedSize
            )
        }
        
        // Verify the downloaded file size matches the expected size
        let actualSize = try tempFile.seekToEnd()
        if let expectedSize = expectedSize, expectedSize != actualSize {
            throw DownloadError.unexpectedError
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
