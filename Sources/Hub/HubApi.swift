//
//  HubApi.swift
//
//
//  Created by Pedro Cuenca on 20231230.
//

import Foundation

public struct HubApi {
    let endpoint = "https://huggingface.co/api"
    var downloadBase: URL
    
    static let shared = {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return HubApi(downloadBase: documents.appending(component: "huggingface"))
    }()
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
    
    enum RepoType: String {
        case models
        case datasets
        case spaces
    }
    
    func getFilenames(from repoId: String, repoType: RepoType = .models, matching glob: String? = nil) async throws -> [String] {
        // Read repo info and only parse "siblings"
        let url = URL(string: "\(endpoint)/\(repoType)/\(repoId)")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let response = try JSONDecoder().decode(SiblingsResponse.self, from: data)
        let filenames = response.siblings.map { $0.rfilename }
        guard let glob = glob else { return filenames }
        return filenames.matching(glob: glob)
    }
}

/// Snaphsot download
public extension HubApi {
    func destination(repoId: String, repoType: RepoType = .models) -> URL {
        downloadBase.appending(component: repoType.rawValue).appending(component: repoId)
    }
    
    struct HubFileDownloader {
        let repoId: String
        let repoType: RepoType
        let repoDestination: URL
        let relativeFilename: String
        
        var source: URL {
            // https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/tokenizer.json?download=true
            var url = URL(string: "https://huggingface.co")!
            if repoType != .models {
                url = url.appending(component: repoType.rawValue)
            }
            url = url.appending(path: repoId)
            url = url.appending(path: "resolve/main")  // TODO: revisions
            url = url.appending(path: relativeFilename)
            return url
        }
        
        var destination: URL {
            repoDestination.appending(path: relativeFilename)
        }
        
        var downloaded: Bool {
            FileManager.default.fileExists(atPath: destination.path)
        }
        
        func prepareDestination() throws {
            let directoryURL = destination.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil)
        }

        // Note we go from Combine in Downloader to callback-based progress reporting
        // We'll probably need to support Combine as well to play well with Swift UI
        // (See for example PipelineLoader in swift-coreml-diffusers)
        @discardableResult
        func download(progressHandler: @escaping (Double) -> Void) async throws -> URL {
            guard !downloaded else { return destination }
            
            try prepareDestination()
            let downloader = Downloader(from: source, to: destination)
            let _ = downloader.downloadState.sink { state in
                if case .downloading(let progress) = state {
                    progressHandler(progress)
                }
            }
            try downloader.waitUntilDone()
            return destination
        }
    }
    
    func snapshot(from repoId: String, repoType: RepoType = .models, matching glob: String? = nil, progressHandler: @escaping (Progress) -> Void) async throws -> URL {
        let filenames = try await getFilenames(from: repoId, repoType: repoType, matching: glob)
        let progress = Progress(totalUnitCount: Int64(filenames.count))
        let repoDestination = destination(repoId: repoId, repoType: repoType)
        for filename in filenames {
            let fileProgress = Progress(totalUnitCount: 100, parent: progress, pendingUnitCount: 1)
            let downloader = HubFileDownloader(repoId: repoId, repoType: repoType, repoDestination: repoDestination, relativeFilename: filename)
            try await downloader.download { fractionDownloaded in
                fileProgress.completedUnitCount = Int64(100 * fractionDownloaded)
                progressHandler(progress)
            }
            fileProgress.completedUnitCount = 100
        }
        progressHandler(progress)
        return repoDestination
    }
}

/// Hub API wrapper (simple, incomplete)
public extension Hub {
    static func getFilenames(from repoId: String, repoType: HubApi.RepoType = .models, matching glob: String? = nil) async throws -> [String] {
        return try await HubApi.shared.getFilenames(from: repoId, repoType: repoType, matching: glob)
    }
    
    static func snapshot(from repoId: String, repoType: HubApi.RepoType = .models, matching glob: String? = nil, progressHandler: @escaping (Progress) -> Void) async throws -> URL {
        return try await HubApi.shared.snapshot(from: repoId, repoType: repoType, matching: glob, progressHandler: progressHandler)
    }
}

public extension Array<String> {
    func matching(glob: String) -> [String] {
        self.filter { fnmatch(glob, $0, 0) == 0 }
    }
}

