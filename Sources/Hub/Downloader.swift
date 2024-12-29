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
    private(set) var metadataDestination: URL

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
        metadataDirURL: URL,
        using authToken: String? = nil,
        inBackground: Bool = false,
        resumeSize: Int = 0,
        headers: [String: String]? = nil,
        expectedSize: Int? = nil,
        timeout: TimeInterval = 10,
        numRetries: Int = 5
    ) {
        self.destination = destination
        let filename = (destination.lastPathComponent as NSString).deletingPathExtension
        self.metadataDestination = metadataDirURL.appending(component: "\(filename).metadata")
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
            var requestHeaders = headers ?? [:]
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
                    try await self.downloadWithStreaming(request: request, resumeSize: resumeSize, numRetries: numRetries, expectedSize: expectedSize)
                } catch {
                    self.downloadState.value = .failed(error)
                }
            }
        }
    }

    private func downloadWithStreaming(
        request: URLRequest,
        resumeSize: Int,
        numRetries: Int,
        expectedSize: Int?
    ) async throws {
        guard let session = self.urlSession else {
            throw DownloadError.unexpectedError
        }
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        FileManager.default.createFile(atPath: tempURL.path, contents: nil)
        let tempFile = try FileHandle(forWritingTo: tempURL)
        
        defer { tempFile.closeFile() }
        
        let (asyncBytes, response) = try await session.bytes(for: request)
        guard let response = response as? HTTPURLResponse else {
            throw DownloadError.unexpectedError
        }
        
        guard (200..<300).contains(response.statusCode) else {
            throw DownloadError.unexpectedError
        }
        
        let totalSize = Int(response.value(forHTTPHeaderField: "Content-Length") ?? "0") ?? 0
        var downloadedSize = resumeSize
        
        var buffer = Data(capacity: chunkSize)
        var newNumRetries = numRetries
        
        do {
            for try await byte in asyncBytes {
                buffer.append(byte)
                if buffer.count == chunkSize {
                    if !buffer.isEmpty { // Filter out keep-alive chunks
                        try tempFile.write(contentsOf: buffer)
                        buffer.removeAll(keepingCapacity: true)
                        downloadedSize += chunkSize
                        let progress = Double(downloadedSize) / Double(totalSize + resumeSize)
                        newNumRetries = 5
                        downloadState.value = .downloading(progress)
                    }
                }
            }
            
            if !buffer.isEmpty {
                try tempFile.write(contentsOf: buffer)
                downloadedSize += buffer.count
                buffer.removeAll(keepingCapacity: true)
                let progress = Double(downloadedSize) / Double(totalSize + resumeSize)
                newNumRetries = 5
                downloadState.value = .downloading(progress)
            }
        } catch let error as URLError {
            if newNumRetries <= 0 {
                throw error
            }
            try await Task.sleep(nanoseconds: 1_000_000_000)
            
            try await downloadWithStreaming(
                request: request,
                resumeSize: downloadedSize,
                numRetries: newNumRetries - 1,
                expectedSize: expectedSize
            )
        }
        
        let actualSize = try tempFile.seekToEnd()
        if let expectedSize = expectedSize, expectedSize != actualSize {
            throw DownloadError.unexpectedError
        }
        
        tempFile.closeFile()
        try FileManager.default.moveDownloadedFile(from: tempURL, to: destination)
        
        downloadState.value = .completed(destination)
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
