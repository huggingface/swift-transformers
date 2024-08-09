//
//  Hub.swift
//  
//
//  Created by Pedro Cuenca on 18/5/23.
//

import Foundation

public struct Hub {}

public extension Hub {
    enum HubClientError: LocalizedError {
        case parse
        case jsonSerialization(fileURL: URL, message: String)
        case authorizationRequired
        case unexpectedError
        case httpStatusCode(Int)
        
        public var errorDescription: String? {
            switch self {
            case .parse:
                return NSLocalizedString("Failed to parse the configuration.", comment: "Parse Error")
            case .jsonSerialization(_, let message):
                return message
            case .authorizationRequired:
                return NSLocalizedString("Authorization is required.", comment: "Authorization Error")
            case .unexpectedError:
                return NSLocalizedString("An unexpected error occurred.", comment: "Unexpected Error")
            case .httpStatusCode(let statusCode):
                return String(format: NSLocalizedString("HTTP error with status code: %d", comment: "HTTP Status Code Error"), statusCode)
            }
        }
    }
    enum RepoType: String {
        case models
        case datasets
        case spaces
    }
    
    struct Repo {
        let id: String
        let type: RepoType
        
        public init(id: String, type: RepoType = .models) {
            self.id = id
            self.type = type
        }
    }
}

// MARK: - Configuration files with dynamic lookup

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
    
    /// Tuple of token identifier and string value
    public var tokenValue: (UInt, String)? { value as? (UInt, String) }
}

public class LanguageModelConfigurationFromHub {
    struct Configurations {
        var modelConfig: Config
        var tokenizerConfig: Config?
        var tokenizerData: Config
    }

    private var configPromise: Task<Configurations, Error>? = nil

    public init(
        modelName: String,
        hubApi: HubApi = .shared
    ) {
        self.configPromise = Task.init {
            return try await self.loadConfig(modelName: modelName, hubApi: hubApi)
        }
    }
    
    public init(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) {
        self.configPromise = Task {
            return try await self.loadConfig(modelFolder: modelFolder, hubApi: hubApi)
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

                // If the config exists but doesn't contain a tokenizerClass, use a fallback config if we have it
                if let fallbackConfig = Self.fallbackTokenizerConfig(for: modelType) {
                    let configuration = fallbackConfig.dictionary.merging(hubConfig.dictionary, uniquingKeysWith: { current, _ in current })
                    return Config(configuration)
                }

                // Guess by capitalizing
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

    func loadConfig(
        modelName: String,
        hubApi: HubApi = .shared
    ) async throws -> Configurations {
        let filesToDownload = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        let repo = Hub.Repo(id: modelName)
        let downloadedModelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)

        return try await loadConfig(modelFolder: downloadedModelFolder, hubApi: hubApi)
    }
    
    func loadConfig(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) async throws -> Configurations {
        // Note tokenizerConfig may be nil (does not exist in all models)
        let modelConfig = try hubApi.configuration(fileURL: modelFolder.appending(path: "config.json"))
        let tokenizerConfig = try? hubApi.configuration(fileURL: modelFolder.appending(path: "tokenizer_config.json"))
        let tokenizerVocab = try hubApi.configuration(fileURL: modelFolder.appending(path: "tokenizer.json"))
        
        let configs = Configurations(
            modelConfig: modelConfig,
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerVocab
        )
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
