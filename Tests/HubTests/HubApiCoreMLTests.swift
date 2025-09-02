//
//  HubApiCoreMLTests.swift
//  swift-transformers
//
//  Created for testing CoreML-specific Hub functionality
//

import Foundation
@testable import Hub
import XCTest

// MARK: - CoreML Tests

final class HubApiCoreMLTests: XCTestCase {
    var tempDir: URL!
    var mockSession: URLSession!

    override func setUp() {
        super.setUp()

        #if canImport(CoreML)
        // Create temporary directory
        tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        /// Set up mock URL session
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [CoreMLMockURLProtocol.self]
        mockSession = URLSession(configuration: configuration)
        #endif
    }

    override func tearDown() {
        #if canImport(CoreML)
        if let tempDir, FileManager.default.fileExists(atPath: tempDir.path) {
            try? FileManager.default.removeItem(at: tempDir)
        }

        CoreMLMockURLProtocol.mockResponse = nil
        CoreMLMockURLProtocol.mockError = nil
        #endif

        super.tearDown()
    }

    // MARK: - CoreML Model Loading Tests

    func testLoadCoreMLModels() async {
        #if canImport(CoreML)
        // This test will fail in actual loading because we're using mock data
        // but it tests the API structure and error handling
        let repo = Hub.Repo(id: "nonexistent/repo")
        let hubApi = HubApi()

        do {
            _ = try await hubApi.loadCoreMLModels(
                from: repo,
                modelNames: ["TestModel.mlmodelc"],
                computeUnits: .cpuOnly,
                validateModel: false
            )
            XCTFail("Should fail with nonexistent repository")
        } catch {
            // Expected with nonexistent repository - any error type is acceptable
            XCTAssertTrue(true)
        }
        #endif
    }

    func testLoadCoreMLModel() async {
        #if canImport(CoreML)
        let hubApi = HubApi()

        do {
            _ = try await hubApi.loadCoreMLModel(
                from: "nonexistent/repo",
                modelName: "SingleModel.mlmodelc",
                computeUnits: .cpuOnly,
                validateModel: false
            )
            XCTFail("Should fail with nonexistent repository")
        } catch {
            // Expected with nonexistent repository - any error type is acceptable
            XCTAssertTrue(true)
        }
        #endif
    }

    func testLoadCoreMLModelsWithRetry() async {
        #if canImport(CoreML)
        let hubApi = HubApi()
        let repo = Hub.Repo(id: "nonexistent/repo")

        do {
            _ = try await hubApi.loadCoreMLModelsWithRetry(
                from: repo,
                modelNames: ["TestModel.mlmodelc"],
                computeUnits: .cpuOnly,
                validateModel: false
            )
            XCTFail("Should fail with nonexistent repository")
        } catch {
            // Expected with nonexistent repository - any error type is acceptable
            XCTAssertTrue(true)
        }
        #endif
    }

    // MARK: - CoreML Model Discovery Tests

    func testGetCoreMLModelNames() async {
        #if canImport(CoreML)
        // Test with a nonexistent repository - should fail gracefully
        let hubApi = HubApi()
        let repo = Hub.Repo(id: "nonexistent/repo")

        do {
            _ = try await hubApi.getCoreMLModelNames(from: repo)
            XCTFail("Should fail with nonexistent repository")
        } catch {
            // Expected - we're testing error handling for API calls
            XCTAssertTrue(error is HubApi.EnvironmentError || error is URLError)
        }
        #endif
    }

    func testContainsCoreMLModels() async {
        #if canImport(CoreML)
        // Test with a nonexistent repository - should fail gracefully
        let hubApi = HubApi()
        let repo = Hub.Repo(id: "nonexistent/repo")

        do {
            _ = try await hubApi.containsCoreMLModels(repo: repo)
            XCTFail("Should fail with nonexistent repository")
        } catch {
            // Expected - we're testing error handling for API calls
            XCTAssertTrue(error is HubApi.EnvironmentError || error is URLError)
        }
        #endif
    }

    // MARK: - CoreML Model Validation Tests

    func testValidateCoreMLModel() {
        #if canImport(CoreML)
        /// Create mock model directory
        let modelDir = tempDir.appendingPathComponent("ValidModel.mlmodelc")
        try? FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        /// Test missing coremldata.bin
        let hubApi = HubApi()
        XCTAssertThrowsError(try hubApi.validateCoreMLModel(at: modelDir, modelName: "ValidModel.mlmodelc")) { error in
            XCTAssertTrue(error is HubApi.EnvironmentError)
        }

        // Add coremldata.bin
        let coreMLDataPath = modelDir.appendingPathComponent("coremldata.bin")
        let mockData = "mock data".data(using: .utf8)!
        try! mockData.write(to: coreMLDataPath)

        // Should now pass basic validation
        XCTAssertNoThrow(try hubApi.validateCoreMLModel(at: modelDir, modelName: "ValidModel.mlmodelc"))

        // Test missing metadata warning (this should not throw but log)
        // We can't easily test logging, but we can verify the method doesn't throw
        XCTAssertNoThrow(try hubApi.validateCoreMLModel(at: modelDir, modelName: "ValidModel.mlmodelc"))
        #endif
    }

    // MARK: - Error Handling Tests

    func testCoreMLLoadingWithInvalidModel() async {
        #if canImport(CoreML)
        let hubApi = HubApi()

        do {
            _ = try await hubApi.loadCoreMLModels(
                from: Hub.Repo(id: "nonexistent/repo"),
                modelNames: ["InvalidModel.mlmodelc"],
                validateModel: true
            )
            XCTFail("Should have failed with invalid model")
        } catch {
            // Expected - we're testing error handling
            XCTAssertTrue(error is HubApi.EnvironmentError || error is URLError)
        }
        #endif
    }

    func testCoreMLModelNotFound() async {
        #if canImport(CoreML)
        let hubApi = HubApi()

        do {
            _ = try await hubApi.loadCoreMLModel(
                from: "nonexistent/repo",
                modelName: "NonExistentModel.mlmodelc"
            )
            XCTFail("Should have failed with model not found")
        } catch {
            // Expected - we're testing error handling
            XCTAssertTrue(error is HubApi.EnvironmentError || error is URLError)
        }
        #endif
    }
}

#if canImport(CoreML)

// MARK: - Mock URL Protocol for Testing

final class CoreMLMockURLProtocol: URLProtocol {
    static var mockResponse: (Data, HTTPURLResponse)?
    static var mockError: Error?

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        if let error = CoreMLMockURLProtocol.mockError {
            client?.urlProtocol(self, didFailWithError: error)
            return
        }

        if let (data, response) = CoreMLMockURLProtocol.mockResponse {
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
        }

        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() { }
}
#endif
