import argparse
from typing import Dict, Generator, List, Tuple

import numpy as np
from coremltools.models import MLModel
from transformers import AutoTokenizer

from export import METADATA_TOKENIZER


def load(model_path: str) -> Tuple[MLModel, AutoTokenizer]:
    """Load a Core ML model and corresponding tokenizer."""
    model: MLModel = MLModel(model_path)
    description = model.get_spec().description
    if METADATA_TOKENIZER not in description.metadata.userDefined:
        raise ValueError("Model metadata does not contain tokenizer path.")
    tokenizer_path: str = description.metadata.userDefined[METADATA_TOKENIZER]
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def get_next_token(model: MLModel, prompt_tokens: np.ndarray) -> Generator[int, None, None]:
    """Generate a sequence of tokens with naive greedy decoding."""

    def sample(logits: np.ndarray) -> int:
        """Perform greedy decoding on the logits array to get the next token."""
        return int(np.argmax(logits[0][-1], axis=-1))

    def inference(model: MLModel, input_ids: np.ndarray, num_past_tokens: int) -> np.ndarray:
        """Perform inference with the given model and input data."""
        causal_mask: np.ndarray = np.triu(
            np.full(
                (1, 1, input_ids.shape[-1], num_past_tokens + input_ids.shape[-1]),
                fill_value=-np.inf if num_past_tokens == 0 else 0,
            ),
            k=1,
        ).astype(np.float16)
        outputs: Dict[str, np.ndarray] = model.predict(
            data={"inputIds": input_ids, "causalMask": causal_mask},
            state=kv_cache_state,
        )
        return outputs["logits"]

    kv_cache_state = model.make_state()
    logits: np.ndarray = inference(model, input_ids=prompt_tokens, num_past_tokens=0)
    token: int = sample(logits=logits)
    num_past_tokens: int = prompt_tokens.shape[-1]

    while True:
        yield token
        logits: np.ndarray = inference(
            model,
            input_ids=np.array([[token]], dtype=np.int32),
            num_past_tokens=num_past_tokens,
        )
        token: int = sample(logits=logits)
        num_past_tokens += 1


def generate(
    model: MLModel,
    prompt: str,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> str:
    prompt_tokens: np.ndarray = tokenizer(prompt, return_tensors="np").input_ids
    extend_tokens: List[int] = []
    for i, token in enumerate(get_next_token(model, prompt_tokens=prompt_tokens.astype(np.int32))):
        if token == tokenizer.eos_token_id or i == max_new_tokens:
            break
        extend_tokens.append(token)
    return tokenizer.decode(prompt_tokens[0].tolist() + extend_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    model, tokenizer = load(args.model_path)
    extend_text: str = generate(
        model,
        prompt=args.prompt,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )
    print(extend_text)
