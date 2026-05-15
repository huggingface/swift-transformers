//
//  MultilingualConformanceTests.swift
//
//  Byte-identical parity tests against HuggingFace Python `transformers`.
//
//  Baselines under `Resources/MultilingualConformance/baselines/` are produced
//  by `Tools/generate_tokenizer_baselines.py` and treated as the authoritative
//  reference. Each Swift tokenizer kernel is expected to produce identical
//  `input_ids` for every input in the corpus.
//
//  Inputs known to diverge today because of bugs being tracked upstream are
//  enumerated in `expectedDivergences` below with a reference to the relevant
//  issue or PR, so the target lands green while the work is in flight. Any
//  divergence that isn't in that list is a hard failure (regression catch).
//  Any input listed there that now matches Python emits a printed hint inviting
//  removal of the entry — but doesn't fail the test, so the green CI signal
//  isn't broken by an upstream improvement.
//
//  Adding a model: append to `kernels`, append to `MODELS` in the Python
//  script, and re-run it. Adding an input: append to `inputs.json` and re-run
//  the script.
//

import Foundation
import Testing

@testable import Hub
@testable import Models
@testable import Tokenizers

private let downloadDestination: URL = {
    let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    return base.appending(component: "huggingface-tests")
}()

private let hubApiForTests = HubApi(downloadBase: downloadDestination)

// MARK: - Fixtures

private struct CorpusInput: Decodable {
    let id: String
    let category: String
    let text: String
}

private struct Corpus: Decodable {
    let schema_version: Int
    let description: String
    let inputs: [CorpusInput]
}

private struct BaselineEntry: Decodable {
    let input_ids: [Int]
    let tokens: [String]
    // The Python generator also emits `decoded_with_special` / `decoded_skip_special`
    // for future use; they are intentionally not decoded here because decoder-side
    // parity has its own failure modes that deserve a dedicated test (and at least
    // one known-buggy path — see WordPieceDecoder's empty-tokens `tokens.first!`).
}

private struct Baseline: Decodable {
    let model_id: String
    let transformers_version: String
    let entries: [String: BaselineEntry]
}

private enum FixtureError: Error, CustomStringConvertible {
    case missingResource(String)

    var description: String {
        switch self {
        case .missingResource(let name): "missing resource: \(name)"
        }
    }
}

// Resource lookup deliberately doesn't use the `subdirectory:` parameter of
// `Bundle.module.url(forResource:withExtension:subdirectory:)`. SPM's
// `.process("Resources")` does not always preserve the directory layout in a way
// that subdirectory lookup can rely on, but flat lookup by basename works
// because every fixture filename below is unique within the bundle (the corpus
// is named `inputs.json` and every baseline uses a slugified model id).
private func loadCorpus() throws -> Corpus {
    guard let url = Bundle.module.url(forResource: "inputs", withExtension: "json") else {
        throw FixtureError.missingResource("inputs.json")
    }
    return try JSONDecoder().decode(Corpus.self, from: try Data(contentsOf: url))
}

private func loadBaseline(_ slug: String) throws -> Baseline {
    // Slugified model ids replace `/` with `__` so they're valid as filesystem and bundle names.
    guard let url = Bundle.module.url(forResource: slug, withExtension: "json") else {
        throw FixtureError.missingResource("\(slug).json")
    }
    return try JSONDecoder().decode(Baseline.self, from: try Data(contentsOf: url))
}

private func makeTokenizer(_ modelId: String) async throws -> Tokenizer {
    let config = LanguageModelConfigurationFromHub(modelName: modelId, hubApi: hubApiForTests)
    guard let tokenizerConfig = try await config.tokenizerConfig else {
        Issue.record("Missing tokenizer config for \(modelId)")
        throw FixtureError.missingResource("tokenizer_config.json for \(modelId)")
    }
    let tokenizerData = try await config.tokenizerData
    return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
}

// MARK: - Diff formatting

private func formatTokenDiff(
    expected: [Int],
    actual: [Int],
    expectedTokens: [String],
    actualTokens: [String]
) -> String {
    let common = min(expected.count, actual.count)
    var firstDiff = common
    for i in 0..<common where expected[i] != actual[i] {
        firstDiff = i
        break
    }

    var lines: [String] = []
    lines.append("expected \(expected.count) ids, got \(actual.count) ids; first divergence at index \(firstDiff)")
    let window = 3
    let start = max(0, firstDiff - window)
    let endExpected = min(expected.count, firstDiff + window + 1)
    let endActual = min(actual.count, firstDiff + window + 1)
    lines.append("  expected[\(start)..<\(endExpected)]: \(Array(expected[start..<endExpected])) \(Array(expectedTokens[start..<min(expectedTokens.count, endExpected)]))")
    lines.append("    actual[\(start)..<\(endActual)]:   \(Array(actual[start..<endActual])) \(Array(actualTokens[start..<min(actualTokens.count, endActual)]))")
    return lines.joined(separator: "\n")
}

// MARK: - Kernel matrix

private struct Kernel: Sendable, CustomStringConvertible {
    let modelId: String
    let baselineSlug: String

    var description: String { modelId }

    init(_ modelId: String) {
        self.modelId = modelId
        self.baselineSlug = modelId.replacingOccurrences(of: "/", with: "__")
    }
}

private let kernels: [Kernel] = [
    Kernel("BAAI/bge-small-en-v1.5"), // WordPiece (Bert family)
    Kernel("google-t5/t5-small"), // Unigram / SentencePiece
    Kernel("openai-community/gpt2"), // Byte-level BPE (legacy)
    Kernel("Qwen/Qwen2.5-0.5B"), // Byte-level BPE (modern)
    Kernel("TinyLlama/TinyLlama-1.1B-Chat-v1.0"), // SentencePiece BPE with byte-fallback
]

/// (modelId, inputId) pairs that are known to diverge from the Python reference
/// today, paired with a reason string that links the divergence to an upstream
/// issue or PR. Entries should be removed as upstream fixes land.
///
/// The test fails for any (model, input) pair NOT listed here that diverges,
/// and also surfaces a non-fatal warning if a pair listed here now matches —
/// that's the signal to drop the entry from this table.
private let expectedDivergences: [String: [String: String]] = [
    "BAAI/bge-small-en-v1.5": [
        // BasicTokenizer should strip combining marks (NFD then drop Mn) before
        // WordPiece lookup — see #352 Bug 2 / #354.
        "ja_dakuten": "swift-transformers#352 Bug 2 (BasicTokenizer voiced-kana)",
        "ja_handakuten": "swift-transformers#352 Bug 2 (BasicTokenizer voiced-kana)",
        "ja_kanji_mixed": "swift-transformers#352 Bug 2 (BasicTokenizer voiced-kana)",
        "ja_romaji_mixed": "swift-transformers#352 Bug 2 (BasicTokenizer voiced-kana)",
        "ja_long_sentence": "swift-transformers#352 Bug 2 (BasicTokenizer voiced-kana)",
        "ar_diacritics": "swift-transformers#352 Bug 1 / #353 (BasicTokenizer combining mark stripping)",
        "hi_devanagari": "swift-transformers#352 Bug 1 / #353 (BasicTokenizer combining mark stripping)",
        "mixed_polyglot": "swift-transformers#352 Bug 1 / #353 (BasicTokenizer combining mark stripping)",
    ],
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": [
        // SentencePiece BPE with byte-fallback — merges happen at grapheme
        // cluster boundaries in Swift but at Unicode scalar boundaries in HF.
        // See #352 Bug 4 / #355.
        "ascii_code": "swift-transformers#352 (whitespace prefix handling in SentencePiece)",
        "ar_diacritics": "swift-transformers#352 Bug 4 / #355 (BPE merge by grapheme vs scalar)",
        "hi_devanagari": "swift-transformers#352 Bug 4 / #355 (BPE merge by grapheme vs scalar)",
        "th_basic": "swift-transformers#352 Bug 4 / #355 (BPE merge by grapheme vs scalar)",
        "emoji_zwj_family": "swift-transformers#352 Bug 4 / #355 (BPE merge by grapheme vs scalar)",
        "mixed_polyglot": "swift-transformers#352 Bug 4 / #355 (BPE merge by grapheme vs scalar)",
        "edge_combining": "swift-transformers#352 Bug 4 / #355 (BPE merge by grapheme vs scalar)",
    ],
    "Qwen/Qwen2.5-0.5B": [
        // Byte-level BPE merge differences in Arabic / Thai / Devanagari
        // contexts — relates to Unicode scalar handling in BPE merge loop.
        "ar_diacritics": "swift-transformers#352 Bug 4 / #355 (byte-level BPE merge ordering)",
        "th_basic": "swift-transformers#352 Bug 4 / #355 (byte-level BPE merge ordering)",
        "mixed_polyglot": "swift-transformers#352 Bug 4 / #355 (byte-level BPE merge ordering)",
    ],
]

// MARK: - Tests

@Suite("Multilingual Conformance Tests")
struct MultilingualConformanceTests {

    /// Each kernel is checked against every input in the corpus. Failures are reported
    /// per (kernel, input) pair with a windowed diff around the first divergence so the
    /// root cause is visible without re-running the test against a single id.
    @Test("Byte-identical token ids", arguments: kernels)
    fileprivate func tokenIdsMatchPython(kernel: Kernel) async throws {
        let corpus = try loadCorpus()
        let baseline = try loadBaseline(kernel.baselineSlug)
        let tokenizer = try await makeTokenizer(kernel.modelId)
        let expectedDivergent = expectedDivergences[kernel.modelId] ?? [:]

        var unexpectedDivergences: [(input: CorpusInput, message: String)] = []
        var unexpectedMatches: [String] = [] // entries in expected list that now pass — invite removal

        for input in corpus.inputs {
            guard let expected = baseline.entries[input.id] else {
                Issue.record("Baseline for \(kernel.modelId) is missing entry \(input.id) — regenerate with Tools/generate_tokenizer_baselines.py")
                continue
            }
            let actualIds = tokenizer.encode(text: input.text)
            let isMatch = actualIds == expected.input_ids
            let isListedAsDivergent = expectedDivergent[input.id] != nil

            if isMatch {
                if isListedAsDivergent {
                    unexpectedMatches.append(input.id)
                }
                continue
            }
            if isListedAsDivergent { continue }

            let actualTokens = actualIds.map { tokenizer.convertIdToToken($0) ?? "<\($0)>" }
            let message = formatTokenDiff(
                expected: expected.input_ids,
                actual: actualIds,
                expectedTokens: expected.tokens,
                actualTokens: actualTokens
            )
            unexpectedDivergences.append((input, "[\(input.category)] id=\(input.id) text=\(input.text.debugDescription)\n\(message)"))
        }

        // Unexpected divergence is a hard failure: either swift-transformers regressed,
        // or the corpus / baseline added a case that wasn't classified yet.
        for failure in unexpectedDivergences {
            Issue.record("\(failure.message)")
        }
        if !unexpectedDivergences.isEmpty {
            Issue.record("\(kernel.modelId): \(unexpectedDivergences.count) unexpected divergence(s) from Python `transformers` \(baseline.transformers_version). Either swift-transformers regressed or `expectedDivergences` needs a new entry.")
        }

        // Unexpected match is informational: an upstream fix has landed and the entry
        // should be dropped from `expectedDivergences`. Printed but does NOT fail the
        // test, so freshly merged improvements don't break CI; the message surfaces
        // when running locally and is the trigger to clean up the expected list.
        if !unexpectedMatches.isEmpty {
            print(
                "[MultilingualConformance] \(kernel.modelId): \(unexpectedMatches.count) input(s) now match Python — "
                    + "please remove from `expectedDivergences` in MultilingualConformanceTests.swift: "
                    + unexpectedMatches.sorted().joined(separator: ", ")
            )
        }
    }

    /// Sanity check: the corpus itself should not regress in shape or schema between
    /// edits. Caught here so a malformed inputs.json fails fast rather than silently
    /// skipping cases inside the parity test.
    @Test("Corpus is well-formed")
    func corpusIsWellFormed() throws {
        let corpus = try loadCorpus()
        #expect(corpus.schema_version == 1)
        #expect(!corpus.inputs.isEmpty)

        var seen = Set<String>()
        for input in corpus.inputs {
            #expect(!input.id.isEmpty, "input id must not be empty")
            #expect(!seen.contains(input.id), "duplicate input id: \(input.id)")
            seen.insert(input.id)
            #expect(!input.text.isEmpty, "input text must not be empty (id: \(input.id))")
        }
    }

    /// Sanity check: every kernel must have a baseline file covering every input id.
    @Test("Baselines cover the corpus", arguments: kernels)
    fileprivate func baselinesCoverCorpus(kernel: Kernel) throws {
        let corpus = try loadCorpus()
        let baseline = try loadBaseline(kernel.baselineSlug)
        let corpusIds = Set(corpus.inputs.map(\.id))
        let baselineIds = Set(baseline.entries.keys)
        let missing = corpusIds.subtracting(baselineIds)
        let extra = baselineIds.subtracting(corpusIds)
        #expect(missing.isEmpty, "baseline missing entries: \(missing.sorted())")
        #expect(extra.isEmpty, "baseline has stale entries not in corpus: \(extra.sorted())")
    }
}
