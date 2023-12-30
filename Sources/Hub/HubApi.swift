//
//  HubApi.swift
//
//
//  Created by Pedro Cuenca on 20231230.
//

import Foundation

/// Hub API wrapper (simple, incomplete)
public extension Hub {
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
    
    static func getFilenames(from repoId: String, repoType: RepoType = .models, matching glob: String? = nil) async throws -> [String] {
        // Read repo info and only parse "siblings"
        let url = URL(string: "https://huggingface.co/api/\(repoType)/\(repoId)")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let response = try JSONDecoder().decode(SiblingsResponse.self, from: data)
        let filenames = response.siblings.map { $0.rfilename }
        guard let glob = glob else { return filenames }
        return filenames.matching(glob: glob)
    }
}

public extension Array<String> {
    func matching(glob: String) -> [String] {
        self.filter { fnmatch(glob, $0, 0) == 0 }
    }
}

