# Benchmarks

Simple tokenizer benchmarks built with
[benchmark](https://github.com/ordo-one/benchmark).

Requires Swift 6.1 or later and macOS 15 or later.

## Running

From this directory:

```bash
swift package --disable-sandbox benchmark
```

The suite measures local tokenizer loading, encoding a short sentence, encoding
100 repetitions of that sentence, and decoding the longer input. It covers
WordPiece (`sentence-transformers/all-MiniLM-L6-v2`), BPE
(`mlx-community/Qwen3-0.6B-Base-DQ5`), and Unigram (`FacebookAI/xlm-roberta-base`).

Setup downloads only `tokenizer.json` and `tokenizer_config.json` from the
Hugging Face Hub and caches them locally. Downloads happen outside measurement;
the load benchmarks read the cached files, and encode/decode benchmarks reuse a
tokenizer loaded during setup. Decode input tokens are also prepared in setup.
No model weights are downloaded. The first run requires network access.

Results include wall-clock time, throughput, peak resident memory, and allocation
counts, with at most 100 iterations per benchmark.

## Useful commands

```bash
# List available benchmarks
swift package benchmark list

# Run and store a baseline
swift package --disable-sandbox benchmark baseline update main

# Compare against a stored baseline
swift package --disable-sandbox benchmark baseline compare main
```
