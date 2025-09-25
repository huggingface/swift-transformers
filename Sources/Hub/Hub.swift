//
//  Hub.swift
//
//
//  Created by Pedro Cuenca on 18/5/23.
//

import Foundation

/// A namespace struct providing access to Hugging Face Hub functionality.
///
/// The Hub struct serves as the entry point for interacting with the Hugging Face model repository,
/// providing static methods for downloading models, retrieving file metadata, and managing repository snapshots.
/// All operations are performed through the shared HubApi instance unless specified otherwise.
public enum Hub {}

public extension Hub {
    /// Errors that can occur during Hub client operations.
    ///
    /// This enumeration covers all possible error conditions that may arise when
    /// interacting with the Hugging Face Hub, including network issues, authentication
    /// problems, file system errors, and parsing failures.
    enum HubClientError: LocalizedError {
        /// Authentication is required but no valid token was provided.
        case authorizationRequired
        /// An HTTP error occurred with the specified status code.
        case httpStatusCode(Int)
        /// Failed to parse server response or configuration data.
        case parse
        /// An unexpected error occurred during operation.
        case unexpectedError
        /// A download operation failed with the specified error message.
        case downloadError(String)
        /// The requested file was not found on the server or locally.
        case fileNotFound(String)
        /// A network error occurred during communication.
        case networkError(URLError)
        /// The requested resource could not be found.
        case resourceNotFound(String)
        /// A required configuration file is missing.
        case configurationMissing(String)
        /// A file system operation failed.
        case fileSystemError(Error)
        /// Failed to parse data with the specified error message.
        case parseError(String)

        public var errorDescription: String? {
            switch self {
            case .authorizationRequired:
                String(localized: "Authentication required. Please provide a valid Hugging Face token.")
            case let .httpStatusCode(code):
                String(localized: "HTTP error with status code: \(code)")
            case .parse:
                String(localized: "Failed to parse server response.")
            case .unexpectedError:
                String(localized: "An unexpected error occurred.")
            case let .downloadError(message):
                String(localized: "Download failed: \(message)")
            case let .fileNotFound(filename):
                String(localized: "File not found: \(filename)")
            case let .networkError(error):
                String(localized: "Network error: \(error.localizedDescription)")
            case let .resourceNotFound(resource):
                String(localized: "Resource not found: \(resource)")
            case let .configurationMissing(file):
                String(localized: "Required configuration file missing: \(file)")
            case let .fileSystemError(error):
                String(localized: "File system error: \(error.localizedDescription)")
            case let .parseError(message):
                String(localized: "Parse error: \(message)")
            }
        }
    }

    /// The type of repository on the Hugging Face Hub.
    enum RepoType: String, Codable {
        /// Model repositories containing machine learning models.
        case models
        /// Dataset repositories containing training and evaluation data.
        case datasets
        /// Spaces repositories containing applications and demos.
        case spaces
    }

    /// Represents a repository on the Hugging Face Hub.
    ///
    /// A repository is identified by its unique ID and type, allowing access to
    /// different kinds of resources hosted on the Hub platform.
    struct Repo: Codable {
        /// The unique identifier for the repository (e.g., "microsoft/DialoGPT-medium").
        public let id: String
        /// The type of repository (models, datasets, or spaces).
        public let type: RepoType

        /// Creates a new repository reference.
        ///
        /// - Parameters:
        ///   - id: The unique identifier for the repository
        ///   - type: The type of repository (defaults to .models)
        public init(id: String, type: RepoType = .models) {
            self.id = id
            self.type = type
        }
    }
}

/// Manages language model configuration loading from the Hugging Face Hub.
///
/// This class handles the asynchronous loading and processing of model configurations,
/// tokenizer configurations, and tokenizer data from either remote Hub repositories
/// or local model directories. It provides fallback mechanisms for missing configurations
/// and manages the complexities of different model types and their specific requirements.
public final class LanguageModelConfigurationFromHub: Sendable {
    struct Configurations {
        var modelConfig: Config?
        var tokenizerConfig: Config?
        var tokenizerData: Config
    }

    private let configPromise: Task<Configurations, Error>

    /// Initializes configuration loading from a remote Hub repository.
    ///
    /// - Parameters:
    ///   - modelName: The name/ID of the model repository (e.g., "microsoft/DialoGPT-medium")
    ///   - revision: The git revision to use (defaults to "main")
    ///   - hubApi: The Hub API client to use (defaults to shared instance)
    public init(
        modelName: String,
        revision: String = "main",
        hubApi: HubApi = .shared
    ) {
        configPromise = Task.init {
            try await Self.loadConfig(modelName: modelName, revision: revision, hubApi: hubApi)
        }
    }

    /// Initializes configuration loading from a local model directory.
    ///
    /// - Parameters:
    ///   - modelFolder: The local directory containing model configuration files
    ///   - hubApi: The Hub API client to use for parsing configurations (defaults to shared instance)
    public init(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) {
        configPromise = Task {
            try await Self.loadConfig(modelFolder: modelFolder, hubApi: hubApi)
        }
    }

    /// The main model configuration containing architecture and parameter settings.
    ///
    /// - Returns: The loaded model configuration
    /// - Throws: Hub errors if configuration loading fails
    public var modelConfig: Config? {
        get async throws {
            try await configPromise.value.modelConfig
        }
    }

    /// The tokenizer configuration with automatic fallback handling.
    ///
    /// This property attempts to load the tokenizer configuration from the Hub,
    /// applying fallback configurations when needed and inferring tokenizer classes
    /// based on the model type when not explicitly specified.
    ///
    /// - Returns: The tokenizer configuration, or nil if not available
    /// - Throws: Hub errors if configuration loading fails
    public var tokenizerConfig: Config? {
        get async throws {
            if let hubConfig = try await configPromise.value.tokenizerConfig {
                // Try to guess the class if it's not present and the modelType is
                if hubConfig.tokenizerClass?.string() != nil { return hubConfig }
                guard let modelType = try await modelType else { return hubConfig }

                // If the config exists but doesn't contain a tokenizerClass, use a fallback config if we have it
                if let fallbackConfig = Self.fallbackTokenizerConfig(for: modelType) {
                    let configuration = fallbackConfig.dictionary()?.merging(hubConfig.dictionary(or: [:]), strategy: { current, _ in current }) ?? [:]
                    return Config(configuration)
                }

                // Guess by capitalizing
                var configuration = hubConfig.dictionary(or: [:])
                configuration["tokenizer_class"] = .init("\(modelType.capitalized)Tokenizer")
                return Config(configuration)
            }

            // Fallback tokenizer config, if available
            guard let modelType = try await modelType else { return nil }
            return Self.fallbackTokenizerConfig(for: modelType)
        }
    }

    /// The tokenizer data containing vocabulary and merge rules.
    ///
    /// - Returns: The loaded tokenizer data configuration
    /// - Throws: Hub errors if configuration loading fails
    public var tokenizerData: Config {
        get async throws {
            try await configPromise.value.tokenizerData
        }
    }

    /// The model architecture type extracted from the configuration.
    ///
    /// - Returns: The model type string, or nil if not specified
    /// - Throws: Hub errors if configuration loading fails
    public var modelType: String? {
        get async throws {
            try await modelConfig?.modelType.string()
        }
    }

    static func loadConfig(
        modelName: String,
        revision: String,
        hubApi: HubApi = .shared
    ) async throws -> Configurations {
        let filesToDownload = ["config.json", "tokenizer_config.json", "chat_template.jinja", "chat_template.json", "tokenizer.json"]
        let repo = Hub.Repo(id: modelName)

        do {
            let downloadedModelFolder = try await hubApi.snapshot(from: repo, revision: revision, matching: filesToDownload)
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

    static func loadConfig(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) async throws -> Configurations {
        do {
            // Load required configurations
            let modelConfigURL = modelFolder.appending(path: "config.json")

            var modelConfig: Config? = nil
            if FileManager.default.fileExists(atPath: modelConfigURL.path) {
                modelConfig = try hubApi.configuration(fileURL: modelConfigURL)
            }

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
            // Prefer .jinja template over .json template
            var chatTemplate: String? = nil
            let chatTemplateJinjaURL = modelFolder.appending(path: "chat_template.jinja")
            let chatTemplateJsonURL = modelFolder.appending(path: "chat_template.json")

            if FileManager.default.fileExists(atPath: chatTemplateJinjaURL.path) {
                // Try to load .jinja template as plain text
                chatTemplate = try? String(contentsOf: chatTemplateJinjaURL, encoding: .utf8)
            } else if FileManager.default.fileExists(atPath: chatTemplateJsonURL.path),
                let chatTemplateConfig = try? hubApi.configuration(fileURL: chatTemplateJsonURL)
            {
                // Fall back to .json template
                chatTemplate = chatTemplateConfig.chatTemplate.string()
            }

            if let chatTemplate {
                // Create or update tokenizer config with chat template
                if var configDict = tokenizerConfig?.dictionary() {
                    configDict["chat_template"] = .init(chatTemplate)
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
                if nsError.domain == NSCocoaErrorDomain, nsError.code == NSFileReadNoSuchFileError {
                    throw Hub.HubClientError.fileSystemError(error)
                } else if nsError.domain == "NSJSONSerialization" {
                    throw Hub.HubClientError.parseError("Invalid JSON format: \(nsError.localizedDescription)")
                }
            }
            throw Hub.HubClientError.fileSystemError(error)
        }
    }

    static func fallbackTokenizerConfig(for modelType: String) -> Config? {
        // Fallback tokenizer configuration files are located in the `Sources/Hub/Resources` directory
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
