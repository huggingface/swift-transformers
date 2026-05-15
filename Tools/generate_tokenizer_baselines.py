#!/usr/bin/env python3
"""Regenerate multilingual conformance baselines from HuggingFace Python `transformers`.

This script is the single source of truth for the byte-identical reference values
that `MultilingualConformanceTests` compares Swift output against. To regenerate:

    pip install -r Tools/requirements.txt
    python Tools/generate_tokenizer_baselines.py

Each baseline file is a JSON dictionary keyed by input id, containing the
`input_ids`, the convert_ids_to_tokens result, and the decoded form (both with
and without special tokens). The values are produced by Python's
`AutoTokenizer.from_pretrained(model_id)`, which is treated as the authoritative
reference for byte-identical parity.

When a Swift test fails against a baseline, regenerate locally with the same
`transformers` version listed in `requirements.txt` to confirm the divergence
isn't an upstream change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import transformers
    from transformers import AutoTokenizer
except ImportError:
    sys.stderr.write(
        "transformers is required. Install with: pip install -r Tools/requirements.txt\n"
    )
    sys.exit(1)


# Model matrix is intentionally small and covers four distinct tokenizer kernels
# observed in production HuggingFace text models. Adding a new kernel is
# preferable to adding a near-duplicate of an existing one.
MODELS = [
    # WordPiece (Bert family) — exercises BasicTokenizer pre-tokenization on
    # CJK / dakuten / diacritics. BGE-small-en is the encoder most embedding
    # pipelines on Apple Silicon use today.
    "BAAI/bge-small-en-v1.5",

    # Unigram / SentencePiece — exercises Unigram lattice + Metaspace decoder
    # on multi-codepoint graphemes. T5-small is the canonical Unigram model and
    # ships the tokenizer.json required by swift-transformers.
    "google-t5/t5-small",

    # Byte-level BPE (GPT-2 family) — exercises ByteLevelPreTokenizer regex +
    # byte encoding. Expected to be byte-identical with the Python reference
    # across the entire corpus.
    "openai-community/gpt2",

    # Modern Byte-level BPE (Qwen family) — exercises a more recent vocabulary
    # and merge table while sharing the GPT-2 kernel.
    "Qwen/Qwen2.5-0.5B",

    # SentencePiece BPE with byte-fallback (Llama family) — exercises BPE merge
    # on multi-codepoint graphemes. TinyLlama uses the standard Llama tokenizer
    # without an auth gate.
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


def slugify(model_id: str) -> str:
    """Filesystem-safe representation of a HuggingFace model id."""
    return model_id.replace("/", "__")


def encode_input(tokenizer: Any, text: str) -> dict[str, Any]:
    """Produce a stable JSON-serializable view of how `tokenizer` handles `text`."""
    # `add_special_tokens=True` matches what `tokenizer.encode(text)` and the
    # default `AutoTokenizer(text)` callable produce.
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    decoded_with_special = tokenizer.decode(input_ids, skip_special_tokens=False)
    decoded_skip_special = tokenizer.decode(input_ids, skip_special_tokens=True)
    return {
        "input_ids": list(input_ids),
        "tokens": list(tokens),
        "decoded_with_special": decoded_with_special,
        "decoded_skip_special": decoded_skip_special,
    }


def generate(model_id: str, corpus: list[dict[str, Any]]) -> dict[str, Any]:
    print(f"  loading tokenizer for {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # swift-transformers loads tokenizers via `tokenizer.json` (the Rust-backed
    # fast format), so the Python reference has to be the matching fast
    # tokenizer for parity to be meaningful. Slow tokenizers can silently
    # produce different ids on multi-codepoint inputs.
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"{model_id} resolved to a slow tokenizer; swift-transformers requires a "
            "tokenizer.json (fast) reference. Either pick a model with tokenizer.json "
            "published, or pre-convert one with `AutoTokenizer.save_pretrained` and "
            "point this script at the local path."
        )
    entries: dict[str, Any] = {}
    for item in corpus:
        entries[item["id"]] = encode_input(tokenizer, item["text"])
    return {
        "model_id": model_id,
        "transformers_version": transformers.__version__,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to inputs.json (defaults to Tests/TokenizersTests/Resources/MultilingualConformance/inputs.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the per-model baseline JSON files",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Override the model matrix (default: all 5 kernels)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    base_dir = repo_root / "Tests" / "TokenizersTests" / "Resources" / "MultilingualConformance"
    corpus_path = args.corpus or (base_dir / "inputs.json")
    output_dir = args.output_dir or (base_dir / "baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))["inputs"]
    print(f"loaded {len(corpus)} inputs from {_display_path(corpus_path, repo_root)}")

    for model_id in args.models:
        baseline = generate(model_id, corpus)
        path = output_dir / f"{slugify(model_id)}.json"
        path.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {_display_path(path, repo_root)}")

    return 0


def _display_path(path: Path, repo_root: Path) -> str:
    """Return `path` as repo-relative when it lives inside the repo, otherwise absolute."""
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
