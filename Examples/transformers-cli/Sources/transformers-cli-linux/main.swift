import ArgumentParser
import Foundation
import Hub
import Tokenizers

/// Returns a HubApi configured to use persistent storage on WendyOS (/mnt/app),
/// or the default location otherwise.
func createHubApi() -> HubApi {
    let wendyPersistentPath = "/mnt/app"
    if FileManager.default.fileExists(atPath: wendyPersistentPath) {
        let downloadBase = URL(filePath: wendyPersistentPath).appending(component: "huggingface")
        return HubApi(downloadBase: downloadBase)
    }
    return HubApi()
}

@main
struct TransformersLinuxCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transformers-cli-linux",
        abstract: "Cross-platform CLI for HuggingFace Transformers (tokenization & Hub)",
        version: "0.0.1",
        subcommands: [Demo.self, Tokenize.self, Decode.self, Download.self, ChatTemplate.self],
        defaultSubcommand: Demo.self
    )
}

struct Demo: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run a demo showcasing tokenizer capabilities"
    )

    @Option(name: .shortAndLong, help: "HuggingFace model ID")
    var model: String = "bert-base-uncased"

    func run() async throws {
        print("Swift Transformers Demo")
        print("=======================\n")

        let hub = createHubApi()
        print("Loading tokenizer for '\(model)'...")
        print("Cache location: \(hub.downloadBase.path)")
        let tokenizer = try await AutoTokenizer.from(pretrained: model, hubApi: hub)
        print("Tokenizer loaded successfully!\n")

        // Demo 1: Basic tokenization
        let text1 = "Hello, world! Welcome to Swift Transformers."
        print("Demo 1: Basic Tokenization")
        print("---------------------------")
        print("Input: \"\(text1)\"")
        let tokens1 = tokenizer.encode(text: text1)
        print("Tokens: \(tokens1)")
        print("Token count: \(tokens1.count)")
        print("Decoded: \"\(tokenizer.decode(tokens: tokens1))\"\n")

        // Demo 2: Subword tokenization
        let text2 = "Tokenization handles unknownwords and subwords nicely."
        print("Demo 2: Subword Tokenization")
        print("-----------------------------")
        print("Input: \"\(text2)\"")
        let tokens2 = tokenizer.encode(text: text2)
        print("Tokens: \(tokens2)")
        print("Token count: \(tokens2.count)\n")

        // Demo 3: Special tokens
        let text3 = "Testing special tokens"
        print("Demo 3: Encoding with Special Tokens")
        print("-------------------------------------")
        print("Input: \"\(text3)\"")
        let tokens3 = tokenizer.encode(text: text3)
        print("Tokens (with special tokens): \(tokens3)")
        print("Token count: \(tokens3.count)\n")

        print("Demo complete! Try other commands:")
        print("  tokenize <text>     - Tokenize custom text")
        print("  decode <ids>        - Decode token IDs")
        print("  download <model>    - Download a tokenizer")
        print("  chat-template <msg> - Apply chat template")
    }
}

struct Tokenize: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Tokenize text using a HuggingFace tokenizer"
    )

    @Argument(help: "Text to tokenize")
    var text: String

    @Option(name: .shortAndLong, help: "HuggingFace model ID (e.g., 'bert-base-uncased')")
    var model: String = "bert-base-uncased"

    @Option(name: .shortAndLong, help: "Path to local tokenizer folder")
    var localPath: String?

    @Flag(name: .shortAndLong, help: "Show token strings alongside IDs")
    var verbose: Bool = false

    func run() async throws {
        let hub = createHubApi()
        let tokenizer: Tokenizer
        if let localPath {
            let url = URL(filePath: localPath, directoryHint: .isDirectory)
            tokenizer = try await AutoTokenizer.from(modelFolder: url, hubApi: hub)
        } else {
            tokenizer = try await AutoTokenizer.from(pretrained: model, hubApi: hub)
        }

        let tokens = tokenizer.encode(text: text)

        print("Input: \"\(text)\"")
        print("Token count: \(tokens.count)")
        print("Token IDs: \(tokens)")

        if verbose {
            let tokenStrings = tokenizer.decode(tokens: tokens)
            print("Decoded: \"\(tokenStrings)\"")
        }
    }
}

struct Decode: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Decode token IDs back to text"
    )

    @Argument(help: "Token IDs to decode (comma-separated)")
    var tokenIds: String

    @Option(name: .shortAndLong, help: "HuggingFace model ID")
    var model: String = "bert-base-uncased"

    @Option(name: .shortAndLong, help: "Path to local tokenizer folder")
    var localPath: String?

    func run() async throws {
        let hub = createHubApi()
        let tokenizer: Tokenizer
        if let localPath {
            let url = URL(filePath: localPath, directoryHint: .isDirectory)
            tokenizer = try await AutoTokenizer.from(modelFolder: url, hubApi: hub)
        } else {
            tokenizer = try await AutoTokenizer.from(pretrained: model, hubApi: hub)
        }

        let ids = tokenIds.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let decoded = tokenizer.decode(tokens: ids)

        print("Token IDs: \(ids)")
        print("Decoded: \"\(decoded)\"")
    }
}

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Download a tokenizer from HuggingFace Hub"
    )

    @Argument(help: "HuggingFace model ID to download")
    var model: String

    @Option(name: .shortAndLong, help: "Output directory (defaults to HF cache)")
    var output: String?

    func run() async throws {
        let hub = createHubApi()
        print("Downloading tokenizer for '\(model)'...")
        print("Cache location: \(hub.downloadBase.path)")

        let repo = Hub.Repo(id: model)

        // Download tokenizer files
        let files = ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "vocab.json", "merges.txt"]

        for file in files {
            do {
                let url = try await hub.snapshot(from: repo, matching: [file])
                print("  Downloaded: \(file) -> \(url.path)")
            } catch {
                // File might not exist for this tokenizer type
            }
        }

        print("Done! Tokenizer cached locally.")

        // Verify it works
        let tokenizer = try await AutoTokenizer.from(pretrained: model, hubApi: hub)
        let testTokens = tokenizer.encode(text: "Hello, world!")
        print("Verification: \"Hello, world!\" -> \(testTokens.count) tokens")
    }
}

struct ChatTemplate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Apply a chat template to messages"
    )

    @Option(name: .shortAndLong, help: "HuggingFace model ID")
    var model: String = "microsoft/Phi-3-mini-4k-instruct"

    @Option(name: .shortAndLong, help: "System message")
    var system: String?

    @Argument(help: "User message")
    var message: String

    func run() async throws {
        let hub = createHubApi()
        let tokenizer = try await AutoTokenizer.from(pretrained: model, hubApi: hub)

        var messages: [[String: String]] = []

        if let system {
            messages.append(["role": "system", "content": system])
        }
        messages.append(["role": "user", "content": message])

        let tokens = try tokenizer.applyChatTemplate(messages: messages)
        let formatted = tokenizer.decode(tokens: tokens)

        print("Formatted prompt:")
        print("---")
        print(formatted)
        print("---")
        print("\nToken count: \(tokens.count)")
    }
}
