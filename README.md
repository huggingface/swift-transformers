# `swift-transformers`

This is a collection of utilities to help adopt language models in Swift apps.

## Roadmap / To Do

- [ ] Tokenizers: download from the Hub, port from [`tokenizers`](https://github.com/huggingface/tokenizers)
  - [x] BPE family ([in progress](https://github.com/pcuenca/swift-transformers/pull/2))
  - [x] Fix Falcon, broken while porting BPE
  - [x] Improve tests, add edge cases, see https://github.com/xenova/transformers.js/blob/27920d84831e323275b38f0b5186644b7936e1a2/tests/generate_tests.py#L24
  - [x] Include fallback `tokenizer_config.json` for known architectures whose models don't have a configuration in the Hub (GPT2)
  - [ ] Port more tokenizer types
- [ ] [`exporters`](https://github.com/huggingface/exporters) – Core ML conversion tool.
  - [ ] Allow max sequence length to be specified.
  - [ ] Allow discrete shapes
  - [ ] Return `logits` from converted Core ML model
  - [ ] Use `coremltools` @ `main` for latest fixes. In particular, [this merged PR](https://github.com/apple/coremltools/pull/1915) makes it easier to use recent versions of transformers.
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
