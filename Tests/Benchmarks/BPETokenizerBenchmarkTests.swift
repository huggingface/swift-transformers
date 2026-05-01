//
//  BPETokenizerBenchmarkTests.swift
//  swift-transformers
//
//  Benchmarks for the BPE tokenization hot path. Run with `RUN_BENCHMARKS=1`.
//

import Dispatch
import Foundation
import Hub
import Testing

@testable import Tokenizers

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct BPETokenizerBenchmarkTests {
    /// A small but realistic BPE-based tokenizer with a non-trivial merge table.
    /// Qwen3 0.6B uses a Qwen2-style BPE with ~152K merges.
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

        // Short prompt — typical chat turn (~100 chars, ~20 tokens)
        shortText = "Explain the BPE tokenization algorithm in three short bullet points."

        // Medium document — Wikipedia-style paragraph (~1.5 KB, ~300 tokens)
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

        // Long document — many paragraphs (~15 KB, ~3K tokens). Stress test for the
        // per-pretokenized-chunk BPE merge inner loop.
        longText = String(repeating: para + "\n\n", count: 20)

        // Code — long identifiers, lots of camelCase / snake_case fragments where each
        // pre-tokenized chunk forces many merges.
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
            String(format: "%7.2f ms (± %5.2f, p50 %6.2f, p95 %6.2f)", mean, stdDev, p50, p95)
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
        print("  \(label.padding(toLength: 20, withPad: " ", startingAt: 0)) \(s.formatted)")
        return s
    }

    // MARK: - Benchmarks

    @Test("BPE encode throughput across input sizes")
    func encodeThroughput() {
        print("\n=== BPE encode throughput (\(Self.modelId)) ===")
        let cases: [(String, String, Int)] = [
            ("short (~100 B)", shortText, 200),
            ("medium (~3 KB)", mediumText, 50),
            ("long (~30 KB)", longText, 10),
            ("code (~600 B)", codeText, 100),
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

    /// Direct micro-benchmark of `BPETokenizer.bpe(token:)`. Exercises the merge
    /// inner loop without any pre/post-tokenization overhead.
    @Test("BPE merge inner loop on synthetic long words")
    func bpeMergeInnerLoop() throws {
        guard let bpe = tokenizer as? PreTrainedTokenizer else {
            Issue.record("Expected PreTrainedTokenizer")
            return
        }
        guard let model = Mirror(reflecting: bpe).descendant("model") as? BPETokenizer else {
            Issue.record("Expected BPETokenizer model")
            return
        }

        // Pre-tokenize once, then time the BPE merges directly.
        let words: [String] = [
            "internationalization",
            "supercalifragilisticexpialidocious",
            "GPT2BytePairEncoderConfiguration",
            "vocabularyIdentifierToTokenStringMap",
            "shouldFuseConsecutiveUnknownTokenSequencesIntoASingleUnknownToken",
        ]

        print("\n=== BPE merge inner loop (per-word, 1000 iterations) ===")
        for word in words {
            // Match what BPETokenizer.tokenize does: byte-encode first, then BPE.
            let encoded = model.byteEncode(text: word).first ?? word
            let stats = Self.measure(label: "len=\(encoded.count)", iterations: 1000) {
                _ = model.bpe(token: encoded)
            }
            print(String(format: "    word=\"%@\" mean %.3f ms", word, stats.mean))
        }
    }
}
