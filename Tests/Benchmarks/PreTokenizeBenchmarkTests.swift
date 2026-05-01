//
//  PreTokenizeBenchmarkTests.swift
//  swift-transformers
//
//  Benchmarks for the punctuation/whitespace/digits/Bert pre-tokenizers,
//  which all sit on the same `text.ranges(of: re)` pattern as
//  `ByteLevelPreTokenizer` did before the byte-level fast path landed.
//  Run with `RUN_BENCHMARKS=1`.
//

import Dispatch
import Foundation
import Hub
import Testing

@testable import Tokenizers

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct PreTokenizeBenchmarkTests {
    let shortText: String
    let codeText: String
    let mediumText: String
    let longText: String
    let numericText: String

    init() {
        // Short prompt — typical chat turn (~70 B). Regex re-compilation
        // dominates per-call cost in this regime.
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

        // Numeric-heavy text exercises the digit-splitting branch.
        numericText = """
            Order #2024-001234 totals $1,299.99 plus 8.875% tax for a final amount of \
            $1,415.36, paid 2026-04-30 at 14:32:07 UTC. Reference codes: A1B2-C3D4-E5F6, \
            SKU 9876543210, account 4111-1111-1111-1111, phone +1-555-867-5309.
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

    @Test("BertPreTokenizer micro-benchmark")
    func bertPreTokenizer() {
        let preTokenizer = BertPreTokenizer(config: Config([String: Config]()))
        print("\n=== BertPreTokenizer.preTokenize ===")
        let cases: [(String, String, Int)] = [
            ("short (~70 B)", shortText, 5_000),
            ("code (~600 B)", codeText, 1_000),
            ("medium (~3 KB)", mediumText, 200),
            ("long (~30 KB)", longText, 20),
        ]
        for (label, text, iterations) in cases {
            _ = Self.measure(label: label, iterations: iterations) {
                _ = preTokenizer.preTokenize(text: text)
            }
        }
    }

    @Test("WhitespacePreTokenizer micro-benchmark")
    func whitespacePreTokenizer() {
        let preTokenizer = WhitespacePreTokenizer(config: Config([String: Config]()))
        print("\n=== WhitespacePreTokenizer.preTokenize ===")
        let cases: [(String, String, Int)] = [
            ("short (~70 B)", shortText, 5_000),
            ("code (~600 B)", codeText, 1_000),
            ("medium (~3 KB)", mediumText, 200),
            ("long (~30 KB)", longText, 20),
        ]
        for (label, text, iterations) in cases {
            _ = Self.measure(label: label, iterations: iterations) {
                _ = preTokenizer.preTokenize(text: text)
            }
        }
    }

    @Test("PunctuationPreTokenizer micro-benchmark")
    func punctuationPreTokenizer() {
        let preTokenizer = PunctuationPreTokenizer(config: Config([String: Config]()))
        print("\n=== PunctuationPreTokenizer.preTokenize ===")
        let cases: [(String, String, Int)] = [
            ("short (~70 B)", shortText, 5_000),
            ("code (~600 B)", codeText, 1_000),
            ("medium (~3 KB)", mediumText, 200),
            ("long (~30 KB)", longText, 20),
        ]
        for (label, text, iterations) in cases {
            _ = Self.measure(label: label, iterations: iterations) {
                _ = preTokenizer.preTokenize(text: text)
            }
        }
    }

    @Test("DigitsPreTokenizer micro-benchmark")
    func digitsPreTokenizer() {
        let groupedTokenizer = DigitsPreTokenizer(config: Config([String: Config]()))
        let individualTokenizer = DigitsPreTokenizer(config: Config(["individualDigits": true]))

        print("\n=== DigitsPreTokenizer.preTokenize (grouped) ===")
        let cases: [(String, String, Int)] = [
            ("numeric (~250 B)", numericText, 5_000),
            ("short (~70 B)", shortText, 5_000),
            ("code (~600 B)", codeText, 1_000),
            ("medium (~3 KB)", mediumText, 200),
            ("long (~30 KB)", longText, 20),
        ]
        for (label, text, iterations) in cases {
            _ = Self.measure(label: label, iterations: iterations) {
                _ = groupedTokenizer.preTokenize(text: text)
            }
        }

        print("\n=== DigitsPreTokenizer.preTokenize (individual) ===")
        for (label, text, iterations) in cases {
            _ = Self.measure(label: label, iterations: iterations) {
                _ = individualTokenizer.preTokenize(text: text)
            }
        }
    }
}
