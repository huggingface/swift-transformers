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

    public init(_ dictionary: [String: Any]) {
        self.dictionary = dictionary
    }

    func camelCase(_ string: String) -> String {
        return string
            .split(separator: "_")
            .enumerated()
            .map { $0.offset == 0 ? $0.element.lowercased() : $0.element.capitalized }
            .joined()
    }
    
    func uncamelCase(_ string: String) -> String {
        let scalars = string.unicodeScalars
        var result = ""
        
        var previousCharacterIsLowercase = false
        for scalar in scalars {
            if CharacterSet.uppercaseLetters.contains(scalar) {
                if previousCharacterIsLowercase {
                    result += "_"
                }
                let lowercaseChar = Character(scalar).lowercased()
                result += lowercaseChar
                previousCharacterIsLowercase = false
            } else {
                result += String(scalar)
                previousCharacterIsLowercase = true
            }
        }
        
        return result
    }


    public subscript(dynamicMember member: String) -> Config? {
        let key = dictionary[member] != nil ? member : uncamelCase(member)
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
    
    // Instead of doing this we could provide custom classes and decode to them
    public var arrayValue: [Config]? {
        guard let list = value as? [Any] else { return nil }
        return list.map { Config($0 as! [String : Any]) }
    }
}

public class LanguageModelConfigurationFromHub {
    struct Configurations {
        var modelConfig: Config
        var tokenizerConfig: Config?
        var tokenizerData: Config
    }

    private var configPromise: Task<Configurations, Error>? = nil

    public init(modelName: String) {
        self.configPromise = Task.init {
            return try await self.loadConfig(modelName: modelName)
        }
    }

    public var modelConfig: Config {
        get async throws {
            try await configPromise!.value.modelConfig
        }
    }

    public var tokenizerConfig: Config? {
        get async throws {
            if let hubConfig = try await configPromise!.value.tokenizerConfig {
                // Try to guess the class if it's not present and the modelType is
                if let _ = hubConfig.tokenizerClass?.stringValue { return hubConfig }
                guard let modelType = try await modelType else { return hubConfig }
                var configuration = hubConfig.dictionary
                configuration["tokenizer_class"] = "\(modelType.capitalized)Tokenizer"
                return Config(configuration)
            }

            // Fallback tokenizer config, if available
            guard let modelType = try await modelType else { return nil }
            return Self.fallbackTokenizerConfig(for: modelType)
        }
    }

    public var tokenizerData: Config {
        get async throws {
            try await configPromise!.value.tokenizerData
        }
    }

    public var modelType: String? {
        get async throws {
            try await modelConfig.modelType?.stringValue
        }
    }

    func loadConfig(modelName: String) async throws -> Configurations {
        // TODO: caching
        async let modelConfig = try Hub.downloadConfig(repoId: modelName, filename: "config.json")
        async let tokenizerConfig = try Hub.downloadConfig(repoId: modelName, filename: "tokenizer_config.json")
        async let tokenizerVocab = try Hub.downloadConfig(repoId: modelName, filename: "tokenizer.json")

        // Note tokenizerConfig may be nil (does not exist in all models)
        let configs = await Configurations(modelConfig: try modelConfig, tokenizerConfig: try? tokenizerConfig, tokenizerData: try tokenizerVocab)
        return configs
    }

    static func fallbackTokenizerConfig(for modelType: String) -> Config? {
        guard let url = Bundle.module.url(forResource: "\(modelType)_tokenizer_config", withExtension: "json") else { return nil }
        do {
            let data = try Data(contentsOf: url)
            let parsed = try JSONSerialization.jsonObject(with: data, options: [])
            guard let dictionary = parsed as? [String: Any] else { return nil }
            return Config(dictionary)
        } catch {
            return nil
        }
    }
}
