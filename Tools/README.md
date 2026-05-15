# Tools

Repository-side scripts that produce or maintain fixtures used by the Swift test
suite. They run on macOS/Linux with a CPython interpreter and do not touch the
Swift build.

## `generate_tokenizer_baselines.py`

Regenerates the byte-identical reference values consumed by
`Tests/TokenizersTests/MultilingualConformanceTests.swift`. The Python
`transformers` library is treated as the authoritative reference; whenever
this script changes its output, the Swift parity tests are expected to be
re-validated against the new baselines.

### Setup

```sh
python3 -m venv .venv-tokenizer-baselines
.venv-tokenizer-baselines/bin/pip install -r Tools/requirements.txt
```

### Regenerate all kernels

```sh
.venv-tokenizer-baselines/bin/python Tools/generate_tokenizer_baselines.py
```

This rewrites every `Tests/TokenizersTests/Resources/MultilingualConformance/baselines/*.json`
file in place. Commit the diffs together with the upstream `transformers`
version pinned in `Tools/requirements.txt` so the references are reproducible.

### Regenerate a single kernel

```sh
.venv-tokenizer-baselines/bin/python Tools/generate_tokenizer_baselines.py \
    --models BAAI/bge-small-en-v1.5
```

### Adding a new kernel or input

1. Append the model id to the `MODELS` list in
   `generate_tokenizer_baselines.py`, or add an entry to `inputs.json`.
2. Rerun the script. The new baseline file appears under
   `Tests/TokenizersTests/Resources/MultilingualConformance/baselines/`.
3. Mirror the kernel in `MultilingualConformanceTests.swift`'s `kernels`
   array.
4. Run `swift test --filter MultilingualConformanceTests`.

If the Swift tokenizer diverges from the new reference, add an entry to
`expectedDivergences` linking to the relevant upstream issue or PR. The test
target stays green while the divergence remains documented, and the test
prints a hint when an upstream fix lands and the entry can be removed.
