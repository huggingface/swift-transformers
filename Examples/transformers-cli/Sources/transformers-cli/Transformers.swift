import ArgumentParser
import CoreML
import Foundation
import Generation
import Models
import Tokenizers

@available(macOS 15.0, iOS 18.0, *)
@main
struct TransformersCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run text generation on a Core ML language model",
        version: "0.0.1"
    )

    @Argument(help: "Input text")
    var prompt: String

    @Argument(help: "Path to Core ML mlpackage model")
    var modelPath: String = "./model.mlpackage"

    @Option(help: "Maximum amount of tokens the model should generate")
    var maxLength: Int = 100

    @Option(help: "Compute units to load model with {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine}")
    var computeUnits: ComputeUnits = .cpuAndGPU

    @Option(
        help: """
            When enabled, two generation passes are ran, one to 'warm up' and another to collect \
            benchmark metrics.
            """)
    var warmup: Bool = false

    @Option(help: "Enable sampling mode (true) or use greedy decoding (false)")
    var doSample: Bool = false

    @Option(help: "Temperature for sampling (lower = more deterministic, typical: 0.1-2.0)")
    var temperature: Float?

    @Option(help: "Top-k filtering - only consider k most likely tokens (typical: 5-50)")
    var topK: Int?

    @Option(help: "Top-p (nucleus) sampling - cumulative probability threshold (typical: 0.9-0.95)")
    var topP: Float?

    @Option(help: "Min-p sampling - minimum probability threshold scaled by top token (typical: 0.01-0.2)")
    var minP: Float?

    @Option(help: "Repetition penalty to discourage repeating tokens (typical: 1.0-2.0, 1.0 = no penalty)")
    var repetitionPenalty: Float?

    @Option(help: "Path to a local folder containing tokenizer_config.json and tokenizer.json")
    var tokenizerPath: String?

    func generate(
        model: LanguageModel,
        config: GenerationConfig,
        prompt: String,
        printOutput: Bool = true
    ) async throws {
        var tokensReceived = 0
        var previousIndex: String.Index? = nil
        var startTime = Date()
        var promptProcessingTime: Double = 0 // seconds
        try await model.generate(config: config, prompt: prompt) { inProgressGeneration in
            if previousIndex == nil { // Prompt pre-filling
                promptProcessingTime = Date().timeIntervalSince(startTime)
                // Reset start time to more accurately compute the average tps.
                startTime = Date()
            } else { // Extend
                // Only start counting tokens once the prompt has been processed.
                tokensReceived += 1
            }
            let response = formatResponse(inProgressGeneration)
            if printOutput {
                print(response[(previousIndex ?? response.startIndex)...], terminator: "")
                fflush(stdout)
            }
            previousIndex = response.endIndex
        }
        // Current time - start time + elapsed time to process the prompt
        let endTime = Date()
        let completionTime = endTime.timeIntervalSince(startTime) + promptProcessingTime
        let tps = Double(tokensReceived) / endTime.timeIntervalSince(startTime)
        if printOutput {
            print("")
            print(
                """
                \(tps.formatted("%.2f")) tokens/s, \
                prompt pre-filling time: \(promptProcessingTime.formatted("%.2f"))s, \
                total time: \(completionTime.formatted("%.2f"))s
                """)
        }
    }

    func compile(at url: URL) throws -> URL {
        #if os(watchOS)
        fatalError("Model compilation is not supported on watchOS")
        #else
        if url.pathExtension == "mlmodelc" { return url }
        print("Compiling model \(url)")
        return try MLModel.compileModel(at: url)
        #endif
    }

    func run() async throws {
        let url = URL(filePath: modelPath)
        let compiledURL = try compile(at: url)
        print("Loading model \(compiledURL)")
        let model: LanguageModel
        if let tokenizerPath {
            let tokenizerURL = URL(filePath: tokenizerPath, directoryHint: .isDirectory)
            let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
            model = try LanguageModel.loadCompiled(
                url: compiledURL,
                computeUnits: computeUnits.asMLComputeUnits,
                tokenizer: tokenizer
            )
        } else {
            model = try LanguageModel.loadCompiled(url: compiledURL, computeUnits: computeUnits.asMLComputeUnits)
        }

        var config = model.defaultGenerationConfig
        config.doSample = doSample
        config.maxNewTokens = maxLength

        if let temperature = temperature {
            config.temperature = temperature
        }
        if let topK = topK {
            config.topK = topK
        }
        if let topP = topP {
            config.topP = topP
        }
        if let minP = minP {
            config.minP = minP
        }
        if let repetitionPenalty = repetitionPenalty {
            config.repetitionPenalty = repetitionPenalty
        }

        // Given the size of the out-of-model computation, dispatch all
        // tensor operations to the CPU.

        if warmup {
            print("Warming up...")
            try await withMLTensorComputePolicy(.cpuOnly) {
                try await generate(model: model, config: config, prompt: prompt, printOutput: false)
            }
        }

        print("Generating")
        try await withMLTensorComputePolicy(.cpuOnly) {
            try await generate(model: model, config: config, prompt: prompt)
        }
    }
}

@available(macOS 15.0, iOS 18.0, *)
enum ComputeUnits: String, ExpressibleByArgument, CaseIterable {
    case all, cpuAndGPU, cpuOnly, cpuAndNeuralEngine
    var asMLComputeUnits: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuOnly: return .cpuOnly
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        }
    }
}

/// Returns a cleaned and formatted version of the response.
///
/// - Parameter response: The response to clean and format.
/// - Returns: A 'user friendly' representation of the generated response.
private func formatResponse(_ response: String) -> String {
    response
        .replacingOccurrences(of: "\\n", with: "\n")
        .replacingOccurrences(of: "<s>", with: "")
        .replacingOccurrences(of: "</s>", with: "")
}

extension Double {
    func formatted(_ format: String) -> String {
        return String(format: "\(format)", self)
    }
}
