//
//  JSONParserBenchmarkTests.swift
//  swift-transformers
//
//  Benchmark tests comparing JSONSerialization vs yyjson performance.
//

import Dispatch
import Foundation
import Testing
import Tokenizers
import yyjson

@testable import Hub

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct JSONParserBenchmarkTests {
    static let modelId = "mlx-community/Qwen3-0.6B-Base-DQ5"

    let modelFolder: URL
    let benchmarkData: Data
    let offlineHubApi: HubApi

    init() async throws {
        // Download model files first (with network)
        let hubApi = HubApi()
        let repo = Hub.Repo(id: Self.modelId)
        let tokenizerFiles = ["tokenizer.json", "tokenizer_config.json"]
        modelFolder = try await hubApi.snapshot(from: repo, matching: tokenizerFiles)

        let tokenizerURL = modelFolder.appending(path: "tokenizer.json")
        benchmarkData = try Data(contentsOf: tokenizerURL)
        print("Loaded benchmark file: \(ByteCountFormatter.string(fromByteCount: Int64(benchmarkData.count), countStyle: .file))")

        // Create offline HubApi for benchmarking (no network calls)
        offlineHubApi = HubApi(useOfflineMode: true)
    }

    // MARK: - Benchmark Utilities

    struct BenchmarkStats {
        let mean: Double
        let stdDev: Double
        let min: Double
        let max: Double

        var formatted: String {
            String(format: "%.1f ms (± %.1f)", mean, stdDev)
        }
    }

    /// Measures execution time using monotonic clock, returning individual timings in milliseconds.
    private func measure(
        label: String,
        labelWidth: Int,
        iterations: Int,
        warmup: Int = 2,
        _ block: () throws -> Void
    ) rethrows -> [Double] {
        let paddedLabel = label.padding(toLength: labelWidth, withPad: " ", startingAt: 0)
        print("\(paddedLabel) ", terminator: "")
        fflush(stdout)

        // Warmup runs (not measured)
        for _ in 0..<warmup {
            try block()
        }

        var times: [Double] = []
        times.reserveCapacity(iterations)

        for i in 0..<iterations {
            let start = DispatchTime.now()
            try block()
            let end = DispatchTime.now()
            let nanoseconds = end.uptimeNanoseconds - start.uptimeNanoseconds
            times.append(Double(nanoseconds) / 1_000_000)

            if (i + 1) % 10 == 0 {
                print(String(format: "%2d", i + 1), terminator: "")
            } else {
                print(".", terminator: "")
            }
            fflush(stdout)
        }

        let mean = times.reduce(0, +) / Double(times.count)
        print(String(format: " %6.1f ms", mean))

        return times
    }

    /// Async version of measure for async operations.
    private func measureAsync(
        label: String,
        labelWidth: Int,
        iterations: Int,
        warmup: Int = 2,
        _ block: () async throws -> Void
    ) async rethrows -> [Double] {
        let paddedLabel = label.padding(toLength: labelWidth, withPad: " ", startingAt: 0)
        print("\(paddedLabel) ", terminator: "")
        fflush(stdout)

        // Warmup runs (not measured)
        for _ in 0..<warmup {
            try await block()
        }

        var times: [Double] = []
        times.reserveCapacity(iterations)

        for i in 0..<iterations {
            let start = DispatchTime.now()
            try await block()
            let end = DispatchTime.now()
            let nanoseconds = end.uptimeNanoseconds - start.uptimeNanoseconds
            times.append(Double(nanoseconds) / 1_000_000)

            if (i + 1) % 5 == 0 {
                print(String(format: "%2d", i + 1), terminator: "")
            } else {
                print(".", terminator: "")
            }
            fflush(stdout)
        }

        let mean = times.reduce(0, +) / Double(times.count)
        print(String(format: " %6.1f ms", mean))

        return times
    }

    private func stats(_ times: [Double]) -> BenchmarkStats {
        let mean = times.reduce(0, +) / Double(times.count)
        let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(times.count)
        let stdDev = sqrt(variance)
        let min = times.min() ?? 0
        let max = times.max() ?? 0
        return BenchmarkStats(mean: mean, stdDev: stdDev, min: min, max: max)
    }

    @Test
    func compareParsingSpeed() throws {
        let iterations = 50
        let labelWidth = 25

        print("Benchmarking with \(iterations) iterations...\n")

        let yyjsonRawTimes = measure(label: "yyjson (raw)", labelWidth: labelWidth, iterations: iterations) {
            benchmarkData.withUnsafeBytes { buffer in
                let doc = yyjson_read(buffer.baseAddress?.assumingMemoryBound(to: CChar.self), buffer.count, 0)
                yyjson_doc_free(doc)
            }
        }

        let yyjsonConfigTimes = try measure(label: "yyjson -> Config", labelWidth: labelWidth, iterations: iterations) {
            let _ = try YYJSONParser.parseToConfig(benchmarkData)
        }

        let jsonSerRawTimes = try measure(label: "JSONSerialization (raw)", labelWidth: labelWidth, iterations: iterations) {
            let _ = try JSONSerialization.jsonObject(with: benchmarkData, options: [])
        }

        let jsonSerConfigTimes = try measure(label: "JSONSerialization+Config", labelWidth: labelWidth, iterations: iterations) {
            let parsed = try JSONSerialization.jsonObject(with: benchmarkData, options: [])
            let _ = Config(parsed as! [NSString: Any])
        }

        let yyjsonRawStats = stats(yyjsonRawTimes)
        let yyjsonConfigStats = stats(yyjsonConfigTimes)
        let jsonSerRawStats = stats(jsonSerRawTimes)
        let jsonSerConfigStats = stats(jsonSerConfigTimes)

        let rawSpeedup = jsonSerRawStats.mean / yyjsonRawStats.mean
        let configSpeedup = jsonSerConfigStats.mean / yyjsonConfigStats.mean
        let rawTimeSaved = jsonSerRawStats.mean - yyjsonRawStats.mean
        let configTimeSaved = jsonSerConfigStats.mean - yyjsonConfigStats.mean

        print(
            """

            ============================================
            JSON Parsing Benchmark Results (\(iterations) iterations)
            File size: \(ByteCountFormatter.string(fromByteCount: Int64(benchmarkData.count), countStyle: .file))
            ============================================
            yyjson (raw parse):       \(yyjsonRawStats.formatted)
            yyjson -> Config:         \(yyjsonConfigStats.formatted)
            JSONSerialization (raw):  \(jsonSerRawStats.formatted)
            JSONSerialization+Config: \(jsonSerConfigStats.formatted)
            --------------------------------------------
            Raw parse speedup:        \(String(format: "%.2f", rawSpeedup))x (\(String(format: "%.0f", rawTimeSaved)) ms saved)
            Full path speedup:        \(String(format: "%.2f", configSpeedup))x (\(String(format: "%.0f", configTimeSaved)) ms saved)
            ============================================

            """)
    }

    @Test
    func parsingResultsMatch() throws {
        let yyjsonResult = try YYJSONParser.parseToConfig(benchmarkData)
        let jsonSerParsed = try JSONSerialization.jsonObject(with: benchmarkData, options: []) as! [NSString: Any]
        let jsonSerResult = Config(jsonSerParsed)

        // Compare top-level keys
        let yyjsonKeys = Set(yyjsonResult.dictionary()?.keys.map { $0.string } ?? [])
        let jsonSerKeys = Set(jsonSerResult.dictionary()?.keys.map { $0.string } ?? [])

        #expect(yyjsonKeys == jsonSerKeys, "Top-level keys should match")

        // Compare vocab size if present
        if let yyjsonVocab = yyjsonResult.model.vocab.dictionary(),
            let jsonSerVocab = jsonSerResult.model.vocab.dictionary()
        {
            #expect(yyjsonVocab.count == jsonSerVocab.count, "Vocab sizes should match")
            print("Vocab size: \(yyjsonVocab.count) tokens")
        }
    }

    @Test
    func compareTokenizerLoadingSpeed() async throws {
        let iterations = 20
        let labelWidth = 18

        print("Benchmarking tokenizer loading with \(iterations) iterations...\n")

        let yyjsonTimes = try await measureAsync(label: "yyjson (current)", labelWidth: labelWidth, iterations: iterations) {
            let _ = try await AutoTokenizer.from(modelFolder: modelFolder, hubApi: offlineHubApi)
        }

        let jsonSerTimes = try await measureAsync(label: "JSONSerialization", labelWidth: labelWidth, iterations: iterations) {
            let _ = try await loadTokenizerWithJSONSerialization()
        }

        let yyjsonStats = stats(yyjsonTimes)
        let jsonSerStats = stats(jsonSerTimes)
        let speedup = jsonSerStats.mean / yyjsonStats.mean
        let timeSaved = jsonSerStats.mean - yyjsonStats.mean

        print(
            """

            ============================================
            Tokenizer Loading Benchmark (\(iterations) iterations)
            Model: \(Self.modelId)
            ============================================
            yyjson (current):     \(yyjsonStats.formatted)
            JSONSerialization:    \(jsonSerStats.formatted)
            --------------------------------------------
            Speedup: \(String(format: "%.2f", speedup))x faster (\(String(format: "%.0f", timeSaved)) ms saved)
            ============================================

            """)
    }

    /// Loads a tokenizer using JSONSerialization instead of yyjson for comparison.
    private func loadTokenizerWithJSONSerialization() async throws -> Tokenizer {
        let tokenizerDataURL = modelFolder.appending(path: "tokenizer.json")
        let tokenizerConfigURL = modelFolder.appending(path: "tokenizer_config.json")

        // Load tokenizer data with JSONSerialization
        let tokenizerDataRaw = try Data(contentsOf: tokenizerDataURL)
        let tokenizerDataParsed = try JSONSerialization.jsonObject(with: tokenizerDataRaw, options: []) as! [NSString: Any]
        let tokenizerData = Config(tokenizerDataParsed)

        // Load tokenizer config with JSONSerialization
        let tokenizerConfigRaw = try Data(contentsOf: tokenizerConfigURL)
        let tokenizerConfigParsed = try JSONSerialization.jsonObject(with: tokenizerConfigRaw, options: []) as! [NSString: Any]
        let tokenizerConfig = Config(tokenizerConfigParsed)

        return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }
}
