//
//  HubApi+CoreML.swift
//
//  CoreML-specific convenience methods for HubApi
//

#if canImport(CoreML)
import CoreML
import Foundation

public extension HubApi {
    /// Download and load CoreML models from a HuggingFace repository
    /// - Parameters:
    ///   - repo: The repository containing CoreML models
    ///   - modelNames: Array of model file names to load (e.g., ["model.mlmodelc"])
    ///   - revision: The revision to download from
    ///   - computeUnits: MLComputeUnits to use for model loading
    ///   - validateModel: Whether to validate model structure before loading
    ///   - progressHandler: Optional progress handler
    /// - Returns: Dictionary mapping model names to loaded MLModel instances
    func loadCoreMLModels(
        from repo: Repo,
        modelNames: [String],
        revision: String = "main",
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        validateModel: Bool = true,
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> [String: MLModel] {
        // Download the repository
        let repoDirectory = try await snapshot(from: repo, revision: revision, progressHandler: progressHandler)

        // Configure CoreML
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        // Load each model
        var models: [String: MLModel] = [:]

        for modelName in modelNames {
            let modelPath = repoDirectory.appendingPathComponent(modelName)

            // Validate model exists and is a directory
            guard FileManager.default.fileExists(atPath: modelPath.path) else {
                throw EnvironmentError.fileIntegrityError("Model file not found: \(modelName)")
            }

            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: modelPath.path, isDirectory: &isDirectory),
                  isDirectory.boolValue
            else {
                throw EnvironmentError.fileIntegrityError("Model path is not a directory: \(modelName)")
            }

            // Validate essential model files if requested
            if validateModel {
                try validateCoreMLModel(at: modelPath, modelName: modelName)
            }

            // Load the model
            let model = try MLModel(contentsOf: modelPath, configuration: mlConfig)
            models[modelName] = model

            print("Loaded CoreML model: \(modelName)")
        }

        return models
    }

    /// Validate CoreML model structure and essential files
    private func validateCoreMLModel(at modelPath: URL, modelName: String) throws {
        // Check for essential CoreML files
        let coremlDataPath = modelPath.appendingPathComponent("coremldata.bin")
        guard FileManager.default.fileExists(atPath: coremlDataPath.path) else {
            throw EnvironmentError.fileIntegrityError("Missing coremldata.bin in CoreML model: \(modelName)")
        }

        // Check for model metadata
        let metadataPath = modelPath.appendingPathComponent("metadata.json")
        if !FileManager.default.fileExists(atPath: metadataPath.path) {
            print("Missing metadata.json in CoreML model: \(modelName)")
        }

        // Check for model weights if applicable
        let weightsPath = modelPath.appendingPathComponent("weights")
        if !FileManager.default.fileExists(atPath: weightsPath.path) {
            print("Missing weights directory in CoreML model: \(modelName)")
        }
    }

    /// Download and load a single CoreML model
    /// - Parameters:
    ///   - repoId: The repository ID
    ///   - modelName: The model file name to load
    ///   - revision: The revision to download from
    ///   - computeUnits: MLComputeUnits to use for model loading
    ///   - validateModel: Whether to validate model structure before loading
    ///   - progressHandler: Optional progress handler
    /// - Returns: The loaded MLModel instance
    func loadCoreMLModel(
        from repoId: String,
        modelName: String,
        revision: String = "main",
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        validateModel: Bool = true,
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> MLModel {
        let models = try await loadCoreMLModels(
            from: Repo(id: repoId),
            modelNames: [modelName],
            revision: revision,
            computeUnits: computeUnits,
            validateModel: validateModel,
            progressHandler: progressHandler
        )

        guard let model = models[modelName] else {
            throw EnvironmentError.fileIntegrityError("Failed to load CoreML model: \(modelName)")
        }

        return model
    }

    /// Download CoreML models with automatic retry and recovery
    /// - Parameters:
    ///   - repo: The repository containing CoreML models
    ///   - modelNames: Array of model file names to load
    ///   - revision: The revision to download from
    ///   - computeUnits: MLComputeUnits to use for model loading (default: .cpuAndNeuralEngine)
    ///   - validateModel: Whether to validate model structure before loading (default: true)
    ///   - retryConfig: Retry configuration for failed downloads
    ///   - progressHandler: Optional progress handler
    /// - Returns: Dictionary mapping model names to loaded MLModel instances
    func loadCoreMLModelsWithRetry(
        from repo: Repo,
        modelNames: [String],
        revision: String = "main",
        computeUnits: MLComputeUnits? = nil,
        validateModel: Bool? = nil,
        retryConfig: RetryConfig? = nil,
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> [String: MLModel] {
        let mlComputeUnits = computeUnits ?? .cpuAndNeuralEngine
        let mlValidateModel = validateModel ?? true
        let retryConf = retryConfig ?? RetryConfig.default
        var lastError: Error?

        for attempt in 1...retryConf.maxRetries {
            do {
                return try await loadCoreMLModels(
                    from: repo,
                    modelNames: modelNames,
                    revision: revision,
                    computeUnits: mlComputeUnits,
                    validateModel: mlValidateModel,
                    progressHandler: progressHandler
                )
            } catch {
                lastError = error
                print("CoreML model loading attempt \(attempt)/\(retryConf.maxRetries) failed: \(error.localizedDescription)")

                if attempt < retryConf.maxRetries {
                    let delay = retryConf.delay(for: attempt)
                    print("Retrying CoreML model loading in \(String(format: "%.1f", delay)) seconds...")

                    // Clean up potentially corrupted downloads
                    let repoDirectory = localRepoLocation(repo)
                    try? cleanupCorruptedDownloads(repo: repo, localDirectory: repoDirectory)

                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                }
            }
        }

        // If all retries failed, throw the last error
        if let error = lastError {
            print("Failed to load CoreML models after \(retryConf.maxRetries) attempts: \(error.localizedDescription)")
            throw error
        }

        throw EnvironmentError.fileIntegrityError("Unexpected error in CoreML model loading")
    }

    /// Get model specifications from a CoreML model repository
    /// - Parameters:
    ///   - repo: The repository containing the model
    ///   - revision: The revision to query
    /// - Returns: Array of available model file names
    func getCoreMLModelNames(from repo: Repo, revision: String = "main") async throws -> [String] {
        let allFiles = try await getFilenames(from: repo, revision: revision)
        return allFiles.filter { $0.hasSuffix(".mlmodelc") }
    }

    /// Check if a repository contains CoreML models
    /// - Parameters:
    ///   - repo: The repository to check
    ///   - revision: The revision to query
    /// - Returns: True if the repository contains CoreML models
    func containsCoreMLModels(repo: Repo, revision: String = "main") async throws -> Bool {
        let modelNames = try await getCoreMLModelNames(from: repo, revision: revision)
        return !modelNames.isEmpty
    }
}
#endif // canImport(CoreML)
