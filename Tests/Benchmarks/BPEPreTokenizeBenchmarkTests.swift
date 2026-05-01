//
//  BPEPreTokenizeBenchmarkTests.swift
//  swift-transformers
//
//  Benchmarks for the byte-level pre-tokenization hot path
//  (`ByteLevelPreTokenizer.preTokenize`, `BPETokenizer.byteEncode` /
//  `hexaEncode`). Run with `RUN_BENCHMARKS=1`.
//

import Dispatch
import Foundation
import Hub
import Testing

@testable import Tokenizers

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct BPEPreTokenizeBenchmarkTests {
    /// Realistic Qwen3 BPE (~152K merges); same model used by the BPE merge
    /// inner-loop benchmark, so `encode` numbers are directly comparable.
    static let modelId = "mlx-community/Qwen3-0.6B-Base-DQ5"

    let tokenizer: Tokenizer
    let shortText: String
    let mediumText: String
    let longText: String
    let codeText: String

    init() async throws {
        let hubApi = HubApi()
        let repo = Hub.Repo(id: Self.modelId)
        let modelFolder = try await hubApi.snapshot(from: repo, matching: ["tokenizer.json", "tokenizer_config.json"])
        let offlineHubApi = HubApi(useOfflineMode: true)
        tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder, hubApi: offlineHubApi)

        // Short prompt — typical chat turn (~70 chars). This is the regime
        // where regex re-compilation dominates encode time.
        shortText = "Explain the BPE tokenization algorithm in three short bullet points."

        let para = """
        Byte-pair encoding (BPE) is a tokenization algorithm originally proposed for data \
        compression by Philip Gage in 1994. It was later adapted for use in neural machine \
        translation by Sennrich, Haddow, and Birch in 2015, and is now the dominant \
        sub-word tokenization scheme for modern large language models including the GPT, \
        Llama, Qwen, and Mistral families. The algorithm operates by iteratively replacing \
        the most frequent adjacent pair of bytes in a corpus with a new symbol, building up \
        a vocabulary of merges that compactly represents both common words and rare strings.
        """
        mediumText = String(repeating: para + "\n\n", count: 2)
        longText = String(repeating: para + "\n\n", count: 20)

        codeText = """
        public final class GPT2BytePairEncoderConfiguration: Codable, Sendable {
            public let vocabularyIdentifierToTokenStringMap: [Int: String]
            public let bytePairMergeRanksByPairOfStrings: [BytePair: Int]
            public let unknownTokenIdentifierForOutOfVocabularyByteSequences: Int?
            public let beginningOfSequenceSpecialTokenIdentifier: Int?
            public let endOfSequenceSpecialTokenIdentifier: Int?
            public let shouldFuseConsecutiveUnknownTokenSequencesIntoASingleUnknownToken: Bool
        }
        """
    }

    // MARK: - Measurement helpers

    struct Stats {
        let mean: Double
        let stdDev: Double
        let p50: Double
        let p95: Double
        let min: Double
        let max: Double

        var formatted: String {
            String(format: "%7.3f ms (± %5.3f, p50 %6.3f, p95 %6.3f)", mean, stdDev, p50, p95)
        }
    }

    private static func stats(_ times: [Double]) -> Stats {
        let sorted = times.sorted()
        let mean = sorted.reduce(0, +) / Double(sorted.count)
        let variance = sorted.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(sorted.count)
        let stdDev = sqrt(variance)
        let p50 = sorted[sorted.count / 2]
        let p95 = sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.95))]
        return Stats(mean: mean, stdDev: stdDev, p50: p50, p95: p95, min: sorted.first ?? 0, max: sorted.last ?? 0)
    }

    private static func measure(label: String, iterations: Int, warmup: Int = 3, _ block: () -> Void) -> Stats {
        for _ in 0..<warmup { block() }
        var times: [Double] = []
        times.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let start = DispatchTime.now()
            block()
            let end = DispatchTime.now()
            times.append(Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
        }
        let s = stats(times)
        print("  \(label.padding(toLength: 26, withPad: " ", startingAt: 0)) \(s.formatted)")
        return s
    }

    // MARK: - Benchmarks

    /// Direct micro-benchmark of `ByteLevelPreTokenizer.preTokenize`. This
    /// isolates the regex + byte-encoder cost from BPE merging and from
    /// tokenizer-config loading, so before/after numbers are stable across
    /// models that share the GPT-2 byte-level pre-tokenizer.
    @Test("ByteLevelPreTokenizer micro-benchmark")
    func byteLevelPreTokenizerMicro() {
        let preTokenizer = ByteLevelPreTokenizer(config: Config([String: Config]()))
        let preTokenizerWithPrefix = ByteLevelPreTokenizer(config: Config(["addPrefixSpace": true]))
        let preTokenizerNoRegex = ByteLevelPreTokenizer(config: Config(["useRegex": false]))

        print("\n=== ByteLevelPreTokenizer.preTokenize micro-benchmark ===")
        let cases: [(String, String, Int)] = [
            ("short (~70 B)", shortText, 5_000),
            ("code (~600 B)", codeText, 1_000),
            ("medium (~3 KB)", mediumText, 200),
            ("long (~30 KB)", longText, 20),
        ]
        for (label, text, iterations) in cases {
            _ = Self.measure(label: "default \(label)", iterations: iterations) {
                _ = preTokenizer.preTokenize(text: text)
            }
        }
        _ = Self.measure(label: "addPrefixSpace short", iterations: 5_000) {
            _ = preTokenizerWithPrefix.preTokenize(text: shortText)
        }
        _ = Self.measure(label: "useRegex=false short", iterations: 5_000) {
            _ = preTokenizerNoRegex.preTokenize(text: shortText)
        }
    }

    /// Direct micro-benchmark of `BPETokenizer.byteEncode` / `hexaEncode`. These
    /// two helpers share the byte-level regex + byte encoder lookups; this test
    /// covers them on the same model used by the BPE merge benchmark so the
    /// numbers can be added together to reason about end-to-end encode cost.
    @Test("BPETokenizer.byteEncode / hexaEncode micro-benchmark")
    func bpeTokenizerHelperMicro() throws {
        guard let bpe = tokenizer as? PreTrainedTokenizer else {
            Issue.record("Expected PreTrainedTokenizer")
            return
        }
        guard let model = Mirror(reflecting: bpe).descendant("model") as? BPETokenizer else {
            Issue.record("Expected BPETokenizer model")
            return
        }

        print("\n=== BPETokenizer.byteEncode micro-benchmark ===")
        _ = Self.measure(label: "byteEncode short", iterations: 5_000) {
            _ = model.byteEncode(text: shortText)
        }
        _ = Self.measure(label: "byteEncode code", iterations: 1_000) {
            _ = model.byteEncode(text: codeText)
        }
        _ = Self.measure(label: "byteEncode medium", iterations: 200) {
            _ = model.byteEncode(text: mediumText)
        }

        print("\n=== BPETokenizer.hexaEncode micro-benchmark ===")
        // hexaEncode is called once per unknown BPE chunk; benchmark per-call
        // cost on a short identifier-style input.
        let identifier = "supercalifragilisticexpialidocious"
        _ = Self.measure(label: "hexaEncode word", iterations: 10_000) {
            _ = model.hexaEncode(text: identifier)
        }
        _ = Self.measure(label: "hexaEncode short", iterations: 5_000) {
            _ = model.hexaEncode(text: shortText)
        }
    }

    /// End-to-end encode throughput. The pre-tokenize fast path benefits
    /// short-input encoding the most (where regex compilation is the dominant
    /// term), so short cases are weighted heavier.
    @Test("BPE encode end-to-end throughput")
    func encodeThroughput() {
        print("\n=== BPE encode end-to-end (\(Self.modelId)) ===")
        let cases: [(String, String, Int)] = [
            ("short (~70 B)", shortText, 1_000),
            ("code (~600 B)", codeText, 200),
            ("medium (~3 KB)", mediumText, 50),
            ("long (~30 KB)", longText, 10),
        ]
        for (label, text, iterations) in cases {
            let bytes = text.utf8.count
            let stats = Self.measure(label: label, iterations: iterations) {
                _ = tokenizer.encode(text: text, addSpecialTokens: false)
            }
            let mbPerSec = (Double(bytes) / 1_048_576.0) / (stats.mean / 1_000.0)
            print(String(format: "    → %.2f MB/s", mbPerSec))
        }
    }
}
