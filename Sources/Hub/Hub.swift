//
//  Hub.swift
//  
//
//  Created by Pedro Cuenca on 18/5/23.
//

import Foundation

public struct Hub {}

public extension Hub {
    enum HubClientError: Error {
        case download
        case parse
    }
    
    static func download(url: URL) async throws -> Data {
        let (data, _) = try await URLSession.shared.data(from: url)
        return data
    }
    
    static func download(url: String) async throws -> Data {
        guard let realUrl = URL(string: url) else { throw HubClientError.download }
        let (data, _) = try await URLSession.shared.data(from: realUrl)
        return data
    }
    
    /// Downloads file from the given repo, and JSON-decodes it
    /// Returns a `Config` (just a dictionary wrapper) as I'm not sure we can use the same object structure for all tokenizers or models
    static func downloadConfig(repoId: String, filename: String) async throws -> Config {
        let url = "https://huggingface.co/\(repoId)/resolve/main/\(filename)"
        let data = try await download(url: url)
        
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [String: Any] else { throw HubClientError.parse }
        return Config(dictionary)
    }
}

@dynamicMemberLookup
public struct Config {
    public private(set) var dictionary: [String: Any]

    init(_ dictionary: [String: Any]) {
        self.dictionary = dictionary
    }

    func camelCase(_ string: String) -> String {
        return string
            .split(separator: "_")
            .enumerated()
            .map { $0.offset == 0 ? $0.element.lowercased() : $0.element.capitalized }
            .joined()
    }

    public subscript(dynamicMember member: String) -> Config? {
        let key = dictionary[member] != nil ? member : camelCase(member)
        if let value = dictionary[key] as? [String: Any] {
            return Config(value)
        } else if let value = dictionary[key] {
            return Config(["value": value])
        }
        return nil
    }

    public var value: Any? {
        return dictionary["value"]
    }
    
    public var intValue: Int? { value as? Int }
    public var boolValue: Bool? { value as? Bool }
    public var stringValue: String? { value as? String }
}
