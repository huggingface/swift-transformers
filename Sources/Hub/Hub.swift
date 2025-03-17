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
        case authorizationRequired
        case httpStatusCode(Int)
        case parse
        case unexpectedError
        case downloadError(String)
        case fileNotFound(String)
        case networkError(URLError)
        case resourceNotFound(String)
        case configurationMissing(String)
        case fileSystemError(Error)
        case parseError(String)

        public var errorDescription: String? {
            switch self {
                case .authorizationRequired:
                    return String(localized: "Authentication required. Please provide a valid Hugging Face token.")
                case .httpStatusCode(let code):
                    return String(localized: "HTTP error with status code: \(code)")
                case .parse:
                    return String(localized: "Failed to parse server response.")
                case .unexpectedError:
                    return String(localized: "An unexpected error occurred.")
                case .downloadError(let message):
                    return String(localized: "Download failed: \(message)")
                case .fileNotFound(let filename):
                    return String(localized: "File not found: \(filename)")
                case .networkError(let error):
                    return String(localized: "Network error: \(error.localizedDescription)")
                case .resourceNotFound(let resource):
                    return String(localized: "Resource not found: \(resource)")
                case .configurationMissing(let file):
                    return String(localized: "Required configuration file missing: \(file)")
                case .fileSystemError(let error):
                    return String(localized: "File system error: \(error.localizedDescription)")
                case .parseError(let message):
                    return String(localized: "Parse error: \(message)")
            }
        }
    }

    enum RepoType: String {
        case models
        case datasets
        case spaces
    }

    struct Repo {
        public let id: String
        public let type: RepoType

        public init(id: String, type: RepoType = .models) {
            self.id = id
            self.type = type
        }
    }
}

// MARK: - Configuration files with dynamic lookup

@dynamicMemberLookup
public struct Config {
    public private(set) var dictionary: [NSString: Any]

    public init(_ dictionary: [NSString: Any]) {
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
        let key = (dictionary[member as NSString] != nil ? member : uncamelCase(member)) as NSString
        if let value = dictionary[key] as? [NSString: Any] {
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
        return list.map { Config($0 as! [NSString : Any]) }
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
        let filesToDownload = ["config.json", "tokenizer_config.json", "chat_template.json", "tokenizer.json"]
        let repo = Hub.Repo(id: modelName)

        do {
            let downloadedModelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)
            return try await loadConfig(modelFolder: downloadedModelFolder, hubApi: hubApi)
        } catch {
            // Convert generic errors to more specific ones
            if let urlError = error as? URLError {
                switch urlError.code {
                    case .notConnectedToInternet, .networkConnectionLost:
                        throw Hub.HubClientError.networkError(urlError)
                    case .resourceUnavailable:
                        throw Hub.HubClientError.resourceNotFound(modelName)
                    default:
                        throw Hub.HubClientError.networkError(urlError)
                }
            } else {
                throw error
            }
        }
    }

    func loadConfig(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) async throws -> Configurations {
        do {
            // Load required configurations
            let modelConfigURL = modelFolder.appending(path: "config.json")
            guard FileManager.default.fileExists(atPath: modelConfigURL.path) else {
                throw Hub.HubClientError.configurationMissing("config.json")
            }

            let modelConfig = try hubApi.configuration(fileURL: modelConfigURL)

            let tokenizerDataURL = modelFolder.appending(path: "tokenizer.json")
            guard FileManager.default.fileExists(atPath: tokenizerDataURL.path) else {
                throw Hub.HubClientError.configurationMissing("tokenizer.json")
            }

            let tokenizerData = try hubApi.configuration(fileURL: tokenizerDataURL)

            // Load tokenizer config (optional)
            var tokenizerConfig: Config? = nil
            let tokenizerConfigURL = modelFolder.appending(path: "tokenizer_config.json")
            if FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
                tokenizerConfig = try hubApi.configuration(fileURL: tokenizerConfigURL)
            }

            // Check for chat template and merge if available
            let chatTemplateURL = modelFolder.appending(path: "chat_template.json")
            if FileManager.default.fileExists(atPath: chatTemplateURL.path),
               let chatTemplateConfig = try? hubApi.configuration(fileURL: chatTemplateURL),
               let chatTemplate = chatTemplateConfig.chatTemplate?.stringValue {
                // Create or update tokenizer config with chat template
                if var configDict = tokenizerConfig?.dictionary {
                    configDict["chat_template"] = chatTemplate
                    tokenizerConfig = Config(configDict)
                } else {
                    tokenizerConfig = Config(["chat_template": chatTemplate])
                }
            }

            return Configurations(
                modelConfig: modelConfig,
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData
            )
        } catch let error as Hub.HubClientError {
            throw error
        } catch {
            if let nsError = error as NSError? {
                if nsError.domain == NSCocoaErrorDomain && nsError.code == NSFileReadNoSuchFileError {
                    throw Hub.HubClientError.fileSystemError(error)
                } else if nsError.domain == "NSJSONSerialization" {
                    throw Hub.HubClientError.parseError("Invalid JSON format: \(nsError.localizedDescription)")
                }
            }
            throw Hub.HubClientError.fileSystemError(error)
        }
    }

    static func fallbackTokenizerConfig(for modelType: String) -> Config? {
        guard let url = Bundle.module.url(forResource: "\(modelType)_tokenizer_config", withExtension: "json") else {
            return nil
        }

        do {
            let data = try Data(contentsOf: url)
            let parsed = try JSONSerialization.jsonObject(with: data, options: [])
            guard let dictionary = parsed as? [NSString: Any] else {
                throw Hub.HubClientError.parseError("Failed to parse fallback tokenizer config")
            }
            return Config(dictionary)
        } catch let error as Hub.HubClientError {
            print("Error loading fallback tokenizer config: \(error.localizedDescription)")
            return nil
        } catch {
            print("Error loading fallback tokenizer config: \(error.localizedDescription)")
            return nil
        }
    }
}
