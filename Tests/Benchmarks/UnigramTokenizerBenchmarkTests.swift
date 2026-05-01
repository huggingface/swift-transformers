//
//  UnigramTokenizerBenchmarkTests.swift
//  swift-transformers
//
//  Benchmarks for the SentencePiece / Unigram tokenization hot path
//  (`UnigramTokenizer.tokenize`, `Trie.commonPrefixSearch`,
//  `TokenLattice.viterbi`). Run with `RUN_BENCHMARKS=1`.
//
//  The end-to-end and `tokenize` benchmarks are also designed to expose
//  super-linear scaling: the same paragraph is repeated 1x / 4x / 16x and
//  ms-per-byte is reported, so a flat curve indicates linear behavior and
//  a rising curve indicates O(N^2) String-API misuse.
//

import Dispatch
import Foundation
import Hub
import Testing

@testable import Tokenizers

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct UnigramTokenizerBenchmarkTests {
    /// XLM-RoBERTa base — Unigram SentencePiece, ~250K vocab, the largest
    /// publicly hosted Unigram tokenizer that the existing tests already use.
    static let modelId = "FacebookAI/xlm-roberta-base"

    let tokenizer: Tokenizer
    let unigramModel: UnigramTokenizer
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

        guard let pretrained = tokenizer as? PreTrainedTokenizer,
            let model = Mirror(reflecting: pretrained).descendant("model") as? UnigramTokenizer
        else {
            fatalError("Expected XLM-R PreTrainedTokenizer wrapping UnigramTokenizer")
        }
        unigramModel = model

        // Short prompt — typical chat turn (~70 chars).
        shortText = "Explain the SentencePiece tokenization algorithm in three short bullet points."

        let para = """
            SentencePiece is an unsupervised text tokenizer that treats the input as a raw \
            Unicode stream and learns sub-word units directly, without language-specific \
            pre-tokenization. It is widely used in modern multilingual encoders such as \
            XLM-RoBERTa and decoder-only language models such as T5 and the original Llama, \
            where the Unigram language model variant scores each candidate piece and the \
            Viterbi algorithm picks the most probable segmentation of the sentence.
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

    // MARK: - Measurement helpers (matches BPEPreTokenizeBenchmarkTests)

    struct Stats {
        let mean: Double
        let stdDev: Double
        let p50: Double
        let p95: Double
        let min: Double
        let max: Double

        var formatted: String {
            String(format: "%8.3f ms (± %5.3f, p50 %7.3f, p95 %7.3f)", mean, stdDev, p50, p95)
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
        print("  \(label.padding(toLength: 30, withPad: " ", startingAt: 0)) \(s.formatted)")
        return s
    }

    // MARK: - Benchmarks

    /// Direct micro-benchmark of `UnigramTokenizer.tokenize`. Bypasses
    /// pre-tokenization and post-processing so before/after numbers reflect
    /// only the lattice / trie hot path.
    @Test("UnigramTokenizer.tokenize micro-benchmark")
    func unigramTokenizeMicro() {
        print("\n=== UnigramTokenizer.tokenize micro-benchmark (\(Self.modelId)) ===")
        let cases: [(String, String, Int)] = [
            ("short (~80 B)", shortText, 200),
            ("code (~600 B)", codeText, 100),
            ("medium (~1.3 KB)", mediumText, 50),
            ("long (~13 KB)", longText, 5),
        ]
        for (label, text, iterations) in cases {
            let bytes = text.utf8.count
            let stats = Self.measure(label: label, iterations: iterations) {
                _ = unigramModel.tokenize(text: text)
            }
            let usPerByte = stats.mean * 1_000.0 / Double(bytes)
            print(String(format: "    → %.3f us/byte (mean)", usPerByte))
        }
    }

    /// Scaling benchmark: tokenize the same paragraph repeated 1x / 4x / 16x.
    /// If the implementation is linear, ms-per-byte stays roughly constant;
    /// if it is quadratic the rate doubles each time the input grows ~4x.
    @Test("UnigramTokenizer.tokenize scaling")
    func unigramTokenizeScaling() {
        print("\n=== UnigramTokenizer.tokenize scaling (\(Self.modelId)) ===")
        let para = """
            SentencePiece is an unsupervised text tokenizer that treats the input as a raw \
            Unicode stream and learns sub-word units directly, without language-specific \
            pre-tokenization. The Viterbi algorithm picks the most probable segmentation.
            """
        let factors = [1, 4, 16, 64]
        let iterationsByFactor = [1: 100, 4: 30, 16: 8, 64: 2]
        for f in factors {
            let text = String(repeating: para + "\n", count: f)
            let bytes = text.utf8.count
            let iters = iterationsByFactor[f] ?? 5
            let stats = Self.measure(label: "x\(f) (\(bytes) B)", iterations: iters) {
                _ = unigramModel.tokenize(text: text)
            }
            let usPerByte = stats.mean * 1_000.0 / Double(bytes)
            print(String(format: "    → %.3f us/byte (mean)", usPerByte))
        }
    }

    /// `Trie.commonPrefixSearch` direct micro-benchmark using the live XLM-R
    /// trie — this isolates the Trie traversal cost from lattice / Viterbi.
    @Test("Trie.commonPrefixSearch micro-benchmark")
    func trieCommonPrefixSearchMicro() {
        guard let trie = Mirror(reflecting: unigramModel).descendant("trie") as? Trie<Character> else {
            Issue.record("Expected UnigramTokenizer to expose 'trie'")
            return
        }
        print("\n=== Trie.commonPrefixSearch micro-benchmark (\(Self.modelId)) ===")
        let queries: [(String, String)] = [
            ("ascii", "the algorithm picks the most probable segmentation of the sentence"),
            ("multilingual", "SentencePieceは多言語対応のサブワード分割器です。"),
            ("long ascii", String(repeating: "the quick brown fox jumps over the lazy dog ", count: 4)),
        ]
        for (label, q) in queries {
            _ = Self.measure(label: "search \(label)", iterations: 10_000) {
                _ = trie.commonPrefixSearch(q)
            }
        }
    }

    /// End-to-end encode throughput. Includes pre-tokenization, the Unigram
    /// lattice, and post-processing. Useful as a sanity check that Unigram
    /// fast-path improvements actually surface at the public API level.
    @Test("XLM-R Unigram encode end-to-end throughput")
    func encodeThroughput() {
        print("\n=== XLM-R Unigram encode end-to-end (\(Self.modelId)) ===")
        let cases: [(String, String, Int)] = [
            ("short (~80 B)", shortText, 200),
            ("code (~600 B)", codeText, 100),
            ("medium (~1.3 KB)", mediumText, 50),
            ("long (~13 KB)", longText, 5),
        ]
        for (label, text, iterations) in cases {
            let bytes = text.utf8.count
            let stats = Self.measure(label: label, iterations: iterations) {
                _ = tokenizer.encode(text: text, addSpecialTokens: false)
            }
            let mbPerSec = (Double(bytes) / 1_048_576.0) / (stats.mean / 1_000.0)
            print(String(format: "    → %.3f MB/s", mbPerSec))
        }
    }
}
