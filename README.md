<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="media/swift-t-banner.png">
    <source media="(prefers-color-scheme: light)" srcset="media/swift-t-banner.png">
    <img alt="Swift + Transformers" src="media/swift-t-banner.png" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

[![Unit Tests](https://github.com/huggingface/swift-transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/huggingface/swift-transformers/actions/workflows/ci.yml)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fhuggingface%2Fswift-transformers%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/huggingface/swift-transformers)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fhuggingface%2Fswift-transformers%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/huggingface/swift-transformers)

`swift-transformers` provides tokenizers, chat templates, Hub downloads, and Core ML utilities for language models in Swift apps.

Those familiar with the [`transformers`](https://github.com/huggingface/transformers) Python library will find a familiar yet idiomatic Swift API.

## Overview

Choose the modules you need:

- **Tokenizers** loads Hugging Face tokenizer configurations for BPE, WordPiece, and Unigram models, and renders chat templates with [swift-jinja](https://github.com/huggingface/swift-jinja), including tool specifications.
- **Hub** downloads model files and tokenizer assets, with caching, progress reporting, and optional Xet transport. It uses [swift-huggingface](https://github.com/huggingface/swift-huggingface) for Hub access.
- **Models and Generation** provide Core ML model loading and text-generation utilities.

Tokenization and Hub APIs support macOS, iOS, and Linux. The package targets macOS 13+ and iOS 16+;
Core ML `LanguageModel` APIs require macOS 15+ or iOS 18+. Linux does not provide Core ML inference.

Check out [our v1.0 release post](https://huggingface.co/blog/swift-transformers) and our [original announcement](https://huggingface.co/blog/swift-coreml-llm) for more context on why we built this library.

## Installation

To use `swift-transformers` with SwiftPM, you can add this to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.5")
]
```

For tokenization and Hub downloads, add the corresponding products to your target:

```swift
targets: [
    .target(
        name: "YourTargetName",
        dependencies: [
            .product(name: "Tokenizers", package: "swift-transformers"),
            .product(name: "Hub", package: "swift-transformers"),
        ]
    )
]
```

Use the `Transformers` product instead if you also need the `Models` and `Generation` modules.
Import the modules individually, such as `import Tokenizers` or `import Models`.

## Examples

The most commonly used modules from `swift-transformers` are `Tokenizers` and `Hub`, which allow fast tokenization and
model downloads from the Hugging Face Hub.

### Tokenizing text

Run the examples below from an asynchronous context. Loading a pretrained tokenizer downloads and caches its
configuration and tokenizer assets.

```swift
import Tokenizers

let tokenizer = try await AutoTokenizer.from(pretrained: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
let tokens = tokenizer.encode(text: "Hello, Swift!", addSpecialTokens: false)
let text = tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
```

By default, `encode` adds the tokenizer's configured special tokens, and `decode` retains them.
Use the flags above when you need plain text without those tokens.

### Chat templates

Use the model's chat template to format messages and tokenize the result:

```swift
import Tokenizers

let tokenizer = try await AutoTokenizer.from(pretrained: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
let messages: [Message] = [["role": "user", "content": "Describe the Swift programming language."]]
let tokens = try tokenizer.applyChatTemplate(messages: messages)
```

This overload requests an assistant generation prompt, as expected by the model to continue generation. The exact formatting depends on the model's
template.

### Tool calling

For models whose chat templates support tools, pass their JSON-schema specifications alongside the messages.
Your application runs the requested tools and supplies their results in subsequent messages.

```swift
import Tokenizers

let tokenizer = try await AutoTokenizer.from(pretrained: "mlx-community/Qwen2.5-7B-Instruct-4bit")

let weatherTool: ToolSpec = [
    "type": "function",
    "function": [
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": [
            "type": "object",
            "properties": ["location": ["type": "string", "description": "City and state"]],
            "required": ["location"],
        ] as [String: any Sendable],
    ] as [String: any Sendable],
]

let tokens = try tokenizer.applyChatTemplate(
    messages: [["role": "user", "content": "What's the weather in Paris?"]],
    tools: [weatherTool]
)
```

### Hub downloads

Download selected files from a Hub repository with progress reporting:

```swift
import Foundation
import Hub

let repo = Hub.Repo(id: "mlx-community/Qwen2.5-0.5B-Instruct-2bit-mlx")
let modelDirectory: URL = try await Hub.snapshot(
    from: repo,
    matching: ["config.json", "*.safetensors"],
    progressHandler: { progress in
        print("Download progress: \(progress.fractionCompleted * 100)%")
    }
)
print("Files downloaded to: \(modelDirectory.path)")
```

For private or gated repositories, provide a token through `HF_TOKEN` or a token saved by
[`hf auth login`](https://huggingface.co/docs/huggingface_hub/en/guides/cli#hf-auth-login).
You can also configure a `HubApi` instance with `hfToken` and `downloadBase` to set credentials
and the directory where snapshots are written.

### Local and offline tokenizers

Load bundled or previously downloaded tokenizer files directly:

```swift
import Foundation
import Tokenizers

let tokenizerFolder = URL(fileURLWithPath: "/path/to/tokenizer")
let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder)
```

Keep `tokenizer.json` and `tokenizer_config.json` together, along with `config.json` and a separate
`chat_template.jinja` or `chat_template.json` if the repository uses them. Local loading reads these
files without downloading from the Hub. Use assets from the same checkpoint as your model.

### Core ML integration

The `Models` and `Generation` modules provide handy utilities when working with language models in Core ML. Check out our
example converting and running Mistral 7B using Core ML in [Examples/Mistral7B](Examples/Mistral7B).

The [modernization of Core ML](https://github.com/huggingface/swift-transformers/pull/257) and corresponding examples were primarily contributed by @joshnewnham, @1duo, @alejandro-isaza, @aseemw. Thank you 🙏

### Offline Core ML models

When you bundle a compiled Core ML model and tokenizer files with your app, you can skip any network requests by injecting
the tokenizer when constructing `LanguageModel`:

```swift
import Foundation
import Models
import Tokenizers

// Construct the tokenizer from local files
let compiledURL = URL(fileURLWithPath: "/path/to/model.mlmodelc")
let tokenizerFolder = URL(fileURLWithPath: "/path/to/tokenizer")

// Requires macOS 15+ or iOS 18+.
let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder)
let model = try LanguageModel.loadCompiled(
    url: compiledURL,
    tokenizer: tokenizer
)
```

Make sure the tokenizer assets come from the same Hugging Face repo as the original checkpoint or are compatible with the model you use. For the
Mistral example in `Examples/Mistral7B/`, you can fetch the tokenizer like this:

```bash
hf download \
  mistralai/Mistral-7B-Instruct-v0.3 \
  tokenizer.json tokenizer_config.json \
  --local-dir Examples/Mistral7B/local-tokenizer
```

If the repo is gated, authenticate with `hf auth login` first. Passing the local tokenizer to
`LanguageModel.loadCompiled` avoids a Hub lookup when the model needs its tokenizer.

## Optional Xet downloads

`swift-transformers` includes a `Xet` [package trait](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0450-swiftpm-package-traits.md)
that enables fast, parallel downloads from the Hugging Face Hub
via [swift-xet](https://github.com/huggingface/swift-xet).
Because Xet introduces additional transitive dependencies
(including [AsyncHTTPClient](https://github.com/swift-server/async-http-client)),
it is opt-in.

On Swift 6.1+, enable it in your `Package.swift`:

```swift
dependencies: [
    .package(
        url: "https://github.com/huggingface/swift-transformers",
        from: "1.3.5",
        traits: ["Xet"]
    )
]
```

Or from the command line:

```bash
swift build --traits Xet
swift test --traits Xet
```

When the trait is not enabled,
Hub downloads use the default `URLSession`-based transport.

On Swift versions earlier than 6.1,
package traits are unavailable and Xet transport cannot be enabled.

> [!NOTE]
> If your Xcode project does not expose trait selection, declare the dependency with `traits: ["Xet"]`
> in a local Swift package and add that package to your project.

## Projects that use swift-transformers ❤️

- [WhisperKit](https://github.com/argmaxinc/WhisperKit): A Swift Package for state-of-the-art speech-to-text systems from [Argmax](https://github.com/argmaxinc)
- [MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm): LLM and vision-language model inference using MLX Swift.
- [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples): Apps and examples built with MLX Swift.

Using `swift-transformers` in your project? Let us know and we'll add you to the list!

## Other Tools

- [`swift-chat`](https://github.com/huggingface/swift-chat) (archived), the original demo app for this package.
- [`exporters`](https://github.com/huggingface/exporters), a Core ML conversion package for transformers models, based on Apple's [`coremltools`](https://github.com/apple/coremltools).

## Contributing

Swift Transformers is a community project and we welcome contributions. Please
check out [Issues](https://github.com/huggingface/swift-transformers/issues)
tagged with `good first issue` if you are looking for a place to start!

Before submitting a pull request, please ensure your code:

- Passes the test suite (`swift test`)
- Passes linting checks (`swift format lint --strict --recursive .`)

To format your code, run `swift format -i --recursive .`.

### Offline tokenizer tests

`TokenizersTests` uses in-memory configurations and bundled fixtures. Put tests
that download models or other Hub files in `TokenizersIntegrationTests`, along
with any reference fixtures they need. `swift test` runs both targets; integration
tests require network access.

Run just the offline tokenizer tests with:

```sh
swift test --filter TokenizersTests
```

Linux CI runs the offline tests with and without the `Xet` trait. It skips
`HubTests` and `TokenizersIntegrationTests` because of the existing
[FoundationNetworking teardown race](https://github.com/swiftlang/swift-corelibs-foundation/issues/3675).
To reproduce the Linux selection locally using Docker:

```sh
docker run --rm -t \
  -v "$PWD":/src -w /src \
  -v swift-transformers-tests-spm-cache:/root/.cache/org.swift.swiftpm \
  swift:6.2.3 \
  swift test --scratch-path .build/linux \
    --skip HubTests --skip TokenizersIntegrationTests
```

## Benchmarks

See [Benchmarks](Benchmarks/README.md) for standalone tokenizer benchmarks and
commands to record and compare performance baselines.
The existing `Tests/Benchmarks` suite covers internal hot paths and input scaling;
run it with `RUN_BENCHMARKS=1 swift test -c release --filter Benchmarks`. It prints timing
statistics for comparisons between changes.

## License

[Apache 2](LICENSE).
