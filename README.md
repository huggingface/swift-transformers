# `swift-transformers`
[![Unit Tests](https://github.com/huggingface/swift-transformers/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/huggingface/swift-transformers/actions/workflows/unit-tests.yml)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fhuggingface%2Fswift-transformers%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/huggingface/swift-transformers)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fhuggingface%2Fswift-transformers%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/huggingface/swift-transformers)

This is a collection of utilities to help adopt language models in Swift apps. 

It tries to follow the Python `transformers` API and abstractions whenever possible, but it also aims to provide an idiomatic Swift interface and does not assume prior familiarity with [`transformers`](https://github.com/huggingface/transformers) or [`tokenizers`](https://github.com/huggingface/tokenizers).


## Rationale & Overview

Check out [our announcement post](https://huggingface.co/blog/swift-coreml-llm).

## Modules

- `Tokenizers`: Utilities to convert text to tokens and back, with support for Chat Templates and Tools. Follows the abstractions in [`tokenizers`](https://github.com/huggingface/tokenizers). Usage example:
```swift
import Tokenizers
func testTokenizer() async throws {
    let tokenizer = try await AutoTokenizer.from(pretrained: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    let messages = [["role": "user", "content": "Describe the Swift programming language."]]
    let encoded = try tokenizer.applyChatTemplate(messages: messages)
    let decoded = tokenizer.decode(tokens: encoded)
}
```

You don't usually need to tokenize the input text yourself - the [`Generation` code](https://github.com/huggingface/swift-transformers/blob/17d4bfae3598482fc7ecf1a621aa77ab586d379a/Sources/Generation/Generation.swift#L82) will take care of it.

- `Hub`: Utilities to download config files from the Hub, used to instantiate tokenizers and learn about language model characteristics.
- `Generation`: Algorithms for text generation. Currently supported ones are greedy search and top-k sampling.
- `Models`: Language model abstraction over a Core ML package.

## Supported Models

This package has been tested with autoregressive language models such as:

- GPT, GPT-Neox, GPT-J.
- SantaCoder.
- StarCoder.
- Falcon.
- Llama 2.

Encoder-decoder models such as T5 and Flan are currently _not supported_. They are high up in our [priority list](#roadmap).

## Usage via SwiftPM

To use `swift-transformers` with SwiftPM, you can add this to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.17")
]
```

And then, add the Transformers library as a dependency to your target:

```
targets: [
    .target(
        name: "YourTargetName",
        dependencies: [
            .product(name: "Transformers", package: "swift-transformers")
        ]
    )
]
```

## Other Tools

- [`swift-chat`](https://github.com/huggingface/swift-chat), a simple app demonstrating how to use this package.
- [`exporters`](https://github.com/huggingface/exporters), a Core ML conversion package for transformers models, based on Apple's [`coremltools`](https://github.com/apple/coremltools).
- [`transformers-to-coreml`](https://huggingface.co/spaces/coreml-projects/transformers-to-coreml), a no-code Core ML conversion tool built on `exporters`.

## License

[Apache 2](LICENSE).
