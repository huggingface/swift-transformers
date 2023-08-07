# `swift-transformers`

This is a collection of utilities to help adopt language models in Swift apps. It tries to follow the Python `transformers` API and abstractions whenever possible, but it also aims to provide an idiomatic Swift interface and does not assume prior familiarity with [`transformers`](https://github.com/huggingface/transformers) or [`tokenizers`](https://github.com/huggingface/tokenizers).


## Modules

- `Tokenizers`. Utilities to convert text to tokens and back. Follows the abstractions in [`tokenizers`](https://github.com/huggingface/tokenizers) and [`transformers.js`](https://github.com/xenova/transformers.js). Usage example:

```swift
import Tokenizers

func testTokenizer() async throws {
    let tokenizer = try await AutoTokenizer.from(pretrained: "pcuenq/Llama-2-7b-chat-coreml")
    let inputIds = tokenizer("Today she took a train to the West")
    assert(inputIds == [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122])
}
```

However, you don't usually need to tokenize the input text yourself - the [`Generation` code](https://github.com/huggingface/swift-transformers/blob/17d4bfae3598482fc7ecf1a621aa77ab586d379a/Sources/Generation/Generation.swift#L82) will take care of it.

- `Hub`. Utilities to download configuration files from the Hub, used to instantiate tokenizers and learn about language model characteristics.

- `Generation`. Algorithms for text generation. Currently supported ones are greedy search and top-k sampling.

- `Models`. Language model abstraction over a Core ML package.


## Supported Models

This package has been tested with autoregressive language models such as:

- GPT, GPT-Neox, GPT-J.
- SantaCoder.
- StarCoder.
- Falcon.
- Llama 2.

Encoder-decoder models such as T5 and Flan are currently _not supported_. They are high up in our [priority list](#roadmap).

## Other Tools

- [`swift-chat`](https://github.com/huggingface/swift-chat), a simple app demonstrating how to use this package.
- [`exporters`](https://github.com/huggingface/exporters), a Core ML conversion package for transformers models, based on Apple`s [`coremltools`](https://github.com/apple/coremltools).
- [`transformers-to-coreml`](https://huggingface.co/spaces/coreml-projects/transformers-to-coreml), a no-code Core ML conversion tool built on `exporters`.

## <a name="roadmap"></a> Roadmap / To Do

- [ ] Tokenizers: download from the Hub, port from [`tokenizers`](https://github.com/huggingface/tokenizers)
  - [x] BPE family
  - [x] Fix Falcon, broken while porting BPE
  - [x] Improve tests, add edge cases, see https://github.com/xenova/transformers.js/blob/27920d84831e323275b38f0b5186644b7936e1a2/tests/generate_tests.py#L24
  - [x] Include fallback `tokenizer_config.json` for known architectures whose models don't have a configuration in the Hub (GPT2)
  - [ ] Port other tokenizer types: Unigram, WordPiece
- [ ] [`exporters`](https://github.com/huggingface/exporters) – Core ML conversion tool.
  - [x] Allow max sequence length to be specified.
  - [ ] Allow discrete shapes
  - [x] Return `logits` from converted Core ML model
  - [x] Use `coremltools` @ `main` for latest fixes. In particular, [this merged PR](https://github.com/apple/coremltools/pull/1915) makes it easier to use recent versions of transformers.
- [ ] Generation
  - [ ] Nucleus sampling (we currently have greedy and top-k sampling)
  - [ ] Use [new `top-k` implementation in `Accelerate`](https://developer.apple.com/documentation/accelerate/bnns#4164142).
  - [ ] Support discrete shapes in the underlying Core ML model by selecting the smallest sequence length larger than the input.
- [ ] Optimization: cache past key-values.
- [ ] Encoder-decoder models (T5)
- [ ] [Demo app](https://github.com/pcuenca/swift-chat)
  - [ ] Allow system prompt to be specified.
  - [ ] How to define a system prompt template?
  - [ ] Test a code model (to stretch system prompt definition)

## License

[Apache 2](LICENSE).
