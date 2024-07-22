import ArgumentParser
import CoreML
import Foundation

import Models
import Generation

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

    @Option(help: """
        When enabled, two generation passes are ran, one to 'warm up' and another to collect \
        benchmark metrics. 
        """)
    var warmup: Bool = false

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
            print("""
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
        let model = try LanguageModel.loadCompiled(url: compiledURL, computeUnits: computeUnits.asMLComputeUnits)
        
        // Using greedy generation for now
        var config = model.defaultGenerationConfig
        config.doSample = false
        config.maxNewTokens = maxLength

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
/// - Parameter respone: The response to clean and format.
/// - Returns: A 'user friendly' representation of the generated response.
fileprivate func formatResponse(_ response: String) -> String {
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
