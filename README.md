<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="media/swift-t-banner.png">
    <source media="(prefers-color-scheme: light)" srcset="media/swift-t-banner.png">
    <img alt="Swift + Transformers" src="media/swift-t-banner.png" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

[![Unit Tests](https://github.com/huggingface/swift-transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/huggingface/swift-transformers/actions/workflows/unit-tests.yml)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fhuggingface%2Fswift-transformers%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/huggingface/swift-transformers)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fhuggingface%2Fswift-transformers%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/huggingface/swift-transformers)

`swift-transformers` is a collection of utilities to help adopt language models in Swift apps. 

Those familiar with the [`transformers`](https://github.com/huggingface/transformers) python library will find a familiar but idiomatic Swift API.

## Rationale & Overview

Check out [our announcement post](https://huggingface.co/blog/swift-coreml-llm).

## Modules

- `Tokenizers`: Utilities to convert text to tokens and back, with support for Chat Templates and Tools. Follows the abstractions in [`tokenizers`](https://github.com/huggingface/tokenizers). 

Usage example:
```swift
import Tokenizers
func testTokenizer() async throws {
    let tokenizer = try await AutoTokenizer.from(pretrained: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    let messages = [["role": "user", "content": "Describe the Swift programming language."]]
    let encoded = try tokenizer.applyChatTemplate(messages: messages)
    let decoded = tokenizer.decode(tokens: encoded)
}
```
- `Hub`: Utilities for interacting with the Hugging Face Hub. Download models, tokenizers and other config files. 

Usage example:

```swift
import Hub
func testHub() async throws {
    let repo = Hub.Repo(id: "mlx-community/Qwen2.5-0.5B-Instruct-2bit-mlx")
    let modelDirectory: URL = try await Hub.snapshot(
        from: repo,
        matching: ["config.json", "*.safetensors"],
        progressHandler: { progress in
            print("Download progress: \(progress.fractionCompleted * 100)%")
        }
    )
    print("Files downloaded to: \(modelDirectory.path)")
}
```

- `Generation`: Utilities for text generation, handling tokenization for you. Currently supported sampling methods: greedy search, top-k sampling, and top-p sampling.
- `Models`: Language model abstraction over a Core ML package.

## Usage via SwiftPM

To use `swift-transformers` with SwiftPM, you can add this to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.17")
]
```

And then, add the Transformers library as a dependency to your target:

```swift
targets: [
    .target(
        name: "YourTargetName",
        dependencies: [
            .product(name: "Transformers", package: "swift-transformers")
        ]
    )
]
```

## Projects that use swift-transformers ❤️ 

- [WhisperKit](https://github.com/argmaxinc/WhisperKit): A Swift Package for state-of-the-art speech-to-text systems from [Argmax](https://github.com/argmaxinc)
- [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples): A Swift Package for integrating MLX models in Swift apps.

Using `swift-transformers` in your project? Let us know and we'll add you to the list!

## Other Tools

- [`swift-chat`](https://github.com/huggingface/swift-chat), a simple app demonstrating how to use this package.
- [`exporters`](https://github.com/huggingface/exporters), a Core ML conversion package for transformers models, based on Apple's [`coremltools`](https://github.com/apple/coremltools).

## Contributing 

Swift Transformers is a community project and we welcome contributions. Please
check out [Issues](https://github.com/huggingface/swift-transformers/issues)
tagged with `good first issue` if you are looking for a place to start!

Before submitting a pull request, please ensure your code:

- Passes the test suite (`swift test`)
- Passes linting checks (`swift format lint --recursive .`)

To format your code, run `swift format -i --recursive .`.

## License

[Apache 2](LICENSE).


