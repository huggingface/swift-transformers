import Benchmark
import Foundation
import Hub
import Tokenizers

private let shortText = "Explain the BPE tokenization algorithm in three short bullet points."
private let paragraph = """
    Byte-pair encoding (BPE) is a tokenization algorithm originally proposed for data \
    compression by Philip Gage in 1994. It was later adapted for use in neural machine \
    translation by Sennrich, Haddow, and Birch in 2015, and is now the dominant \
    sub-word tokenization scheme for modern large language models including the GPT, \
    Llama, Qwen, and Mistral families. The algorithm operates by iteratively replacing \
    the most frequent adjacent pair of bytes in a corpus with a new symbol, building up \
    a vocabulary of merges that compactly represents both common words and rare strings.
    """
private let mediumText = String(repeating: paragraph + "\n\n", count: 2)
private let longText = String(repeating: paragraph + "\n\n", count: 20)
private let codeText = """
    public final class GPT2BytePairEncoderConfiguration: Codable, Sendable {
        public let vocabularyIdentifierToTokenStringMap: [Int: String]
        public let bytePairMergeRanksByPairOfStrings: [BytePair: Int]
        public let unknownTokenIdentifierForOutOfVocabularyByteSequences: Int?
        public let beginningOfSequenceSpecialTokenIdentifier: Int?
        public let endOfSequenceSpecialTokenIdentifier: Int?
        public let shouldFuseConsecutiveUnknownTokenSequencesIntoASingleUnknownToken: Bool
    }
    """

private func downloadTokenizer(_ model: String, revision: String) async throws -> URL {
    try await HubApi.shared.snapshot(
        from: Hub.Repo(id: model),
        revision: revision,
        matching: ["config.json", "tokenizer.json", "tokenizer_config.json"]
    )
}

private func loadTokenizer(_ model: String, revision: String) async throws -> Tokenizer {
    let folder = try await downloadTokenizer(model, revision: revision)
    return try await AutoTokenizer.from(modelFolder: folder)
}

let benchmarks: @Sendable () -> Void = {
    Benchmark.defaultConfiguration = .init(
        metrics: [.wallClock, .throughput, .peakMemoryResident, .mallocCountTotal],
        maxIterations: 100
    )

    let models: [(name: String, id: String, revision: String)] = [
        ("WordPiece", "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
        ("BPE", "mlx-community/Qwen3-0.6B-Base-DQ5", "9a160ab0c112dda560cbaee3c9c24fbac2f98549"),
        ("Unigram", "FacebookAI/xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"),
    ]

    let inputs: [(name: String, text: String)] = [
        ("short text", shortText),
        ("medium text", mediumText),
        ("long text", longText),
        ("code", codeText),
    ]

    for model in models {
        Benchmark("Load tokenizer (\(model.name))") { benchmark, folder in
            for _ in benchmark.scaledIterations {
                blackHole(try await AutoTokenizer.from(modelFolder: folder))
            }
        } setup: {
            try await downloadTokenizer(model.id, revision: model.revision)
        }

        for input in inputs {
            Benchmark("Encode \(input.name) (\(model.name))") { benchmark, tokenizer in
                for _ in benchmark.scaledIterations {
                    blackHole(tokenizer.encode(text: input.text, addSpecialTokens: false))
                }
            } setup: {
                try await loadTokenizer(model.id, revision: model.revision)
            }
        }

        Benchmark("Decode (\(model.name))") { (benchmark: Benchmark, state: (Tokenizer, [Int])) in
            let (tokenizer, tokens) = state
            for _ in benchmark.scaledIterations {
                blackHole(tokenizer.decode(tokens: tokens))
            }
        } setup: {
            let tokenizer = try await loadTokenizer(model.id, revision: model.revision)
            return (tokenizer, tokenizer.encode(text: longText, addSpecialTokens: false))
        }
    }
}
