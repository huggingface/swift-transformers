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

Those familiar with the [`transformers`](https://github.com/huggingface/transformers) Python library will find a familiar yet idiomatic Swift API.

## Rationale & Overview

Check out [our announcement post](https://huggingface.co/blog/swift-coreml-llm).

## Examples

The most commonly used modules from `swift-transformers` are `Tokenizers` and `Hub`, which allow fast tokenization and
model downloads from the Hugging Face Hub.

### Tokenizing text + chat templating

Tokenizing text should feel very familiar to those who have used the Python `transformers` library:

```swift
let tokenizer = try await AutoTokenizer.from(pretrained: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
let messages = [["role": "user", "content": "Describe the Swift programming language."]]
let encoded = try tokenizer.applyChatTemplate(messages: messages)
let decoded = tokenizer.decode(tokens: encoded)
```


### Tool calling

`swift-transformers` natively supports formatting inputs for tool calling, allowing for complex interactions with language models:

```swift
let tokenizer = try await AutoTokenizer.from(pretrained: "mlx-community/Qwen2.5-7B-Instruct-4bit")

let weatherTool = [
    "type": "function",
    "function": [
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": [
            "type": "object",
            "properties": ["location": ["type": "string", "description": "City and state"]],
            "required": ["location"]
        ]
    ]
]

let tokens = try tokenizer.applyChatTemplate(
    messages: [["role": "user", "content": "What's the weather in Paris?"]],
    tools: [weatherTool]
)
```


### Hub downloads

Downloading models to a user device _fast_ and _reliably_ is a core requirement of on-device ML. `swift-transformers` provides a simple API to
download models from the Hugging Face Hub, with progress reporting, flaky connection handling, and more:

```swift
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


