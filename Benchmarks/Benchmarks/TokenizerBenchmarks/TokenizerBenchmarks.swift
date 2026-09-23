import Benchmark
import Foundation
import Hub
import Tokenizers

private let sampleText = "The quick brown fox jumps over the lazy dog."
private let longText = String(repeating: sampleText + "\n", count: 100)

private func downloadTokenizer(_ model: String) async throws -> URL {
    try await HubApi.shared.snapshot(
        from: Hub.Repo(id: model),
        matching: ["tokenizer.json", "tokenizer_config.json"]
    )
}

private func loadTokenizer(_ model: String) async throws -> Tokenizer {
    let folder = try await downloadTokenizer(model)
    return try await AutoTokenizer.from(modelFolder: folder)
}

let benchmarks: @Sendable () -> Void = {
    Benchmark.defaultConfiguration = .init(
        metrics: [.wallClock, .throughput, .peakMemoryResident, .mallocCountTotal],
        maxIterations: 100
    )

    let models = [
        ("WordPiece", "sentence-transformers/all-MiniLM-L6-v2"),
        ("BPE", "mlx-community/Qwen3-0.6B-Base-DQ5"),
        ("Unigram", "FacebookAI/xlm-roberta-base"),
    ]

    for (name, model) in models {
        Benchmark("Load tokenizer (\(name))") { benchmark, folder in
            for _ in benchmark.scaledIterations {
                blackHole(try await AutoTokenizer.from(modelFolder: folder))
            }
        } setup: {
            try await downloadTokenizer(model)
        }

        Benchmark("Encode short text (\(name))") { benchmark, tokenizer in
            for _ in benchmark.scaledIterations {
                blackHole(tokenizer.encode(text: sampleText, addSpecialTokens: false))
            }
        } setup: {
            try await loadTokenizer(model)
        }

        Benchmark("Encode long text (\(name))") { benchmark, tokenizer in
            for _ in benchmark.scaledIterations {
                blackHole(tokenizer.encode(text: longText, addSpecialTokens: false))
            }
        } setup: {
            try await loadTokenizer(model)
        }

        Benchmark("Decode (\(name))") { (benchmark: Benchmark, state: (Tokenizer, [Int])) in
            let (tokenizer, tokens) = state
            for _ in benchmark.scaledIterations {
                blackHole(tokenizer.decode(tokens: tokens))
            }
        } setup: {
            let tokenizer = try await loadTokenizer(model)
            return (tokenizer, tokenizer.encode(text: longText, addSpecialTokens: false))
        }
    }
}
