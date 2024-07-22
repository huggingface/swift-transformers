import logging
import os
import warnings
from typing import List, Optional, Tuple

import coremltools as ct
import numpy as np
import torch
from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import (
    MISTRAL_ATTENTION_CLASSES,
    MistralAttention,
    MistralConfig,
    MistralForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

warnings.filterwarnings("ignore")
logging.getLogger("coremltools").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
MODEL_ID: str = "mistralai/Mistral-7B-Instruct-v0.3"
METADATA_TOKENIZER: str = "co.huggingface.exporters.name"


class SliceUpdateKeyValueCache(Cache):
    def __init__(
        self,
        shape: Tuple[int, ...],
        device="cpu",
        dtype=torch.float32,
    ) -> None:
        """KV cache of shape (#layers, batch_size, #kv_heads, context_size, head_dim)."""
        super().__init__()
        self.past_seen_tokens: int = 0
        self.k_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        slice_indices: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update key/value cache tensors for slice [slice_indices[0], slice_indices[1]).
        Return slice of key/value cache tensors from [0, slice_indices[1]).
        """
        if len(slice_indices) != 2:
            raise ValueError(f"Expect tuple of integers [start, end), got {slice_indices=}.")
        begin, end = slice_indices
        self.k_cache[layer_idx, :, : k_state.shape[1], begin:end, :] = k_state
        self.v_cache[layer_idx, :, : v_state.shape[1], begin:end, :] = v_state
        k_cache: torch.Tensor = self.k_cache[layer_idx, :, :, :end, :]
        v_cache: torch.Tensor = self.v_cache[layer_idx, :, :, :end, :]
        return k_cache, v_cache

    def get_seq_length(self, _: int | None = 0) -> int:
        """Get the sequence length of the cache."""
        return self.past_seen_tokens


class SliceUpdateMistralAttention(MistralAttention):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor | None, ...]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Slice update key/value cache
        end_step = attention_mask.shape[-1]
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            slice_indices=(end_step - q_len, end_step),
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None


class StatefulMistralForCausalLM(torch.nn.Module):
    def __init__(self, model_path: str, max_context_size: int = 2048, batch_size: int = 1) -> None:
        super().__init__()

        # Custom attention implementation for stateful slice update key/value cache, override
        # "sdpa" to compliance with transformers.modeling_utils._autoset_attn_implementation
        MISTRAL_ATTENTION_CLASSES["sdpa"] = SliceUpdateMistralAttention
        self.model = MistralForCausalLM.from_pretrained(model_path)

        # Register KV cache buffers to be recognized as Core ML states
        config: MistralConfig = self.model.config
        self.kv_cache_shape: Tuple[int, ...] = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_context_size,
            config.hidden_size // config.num_attention_heads,
        )
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)
        self.register_buffer("keyCache", self.kv_cache.k_cache)
        self.register_buffer("valueCache", self.kv_cache.v_cache)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Compute past seen tokens used for updating key/value cache slices
        self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits


def export() -> None:
    # Construct model from transformers and trace to TorchScript
    max_context_size: int = 2048
    torch_model = StatefulMistralForCausalLM(MODEL_ID, max_context_size=max_context_size)
    torch_model.eval()
    input_ids: torch.Tensor = torch.zeros((1, 2), dtype=torch.int32)
    causal_mask: torch.Tensor = torch.zeros((1, 1, 2, 5), dtype=torch.float32)
    traced_model = torch.jit.trace(torch_model, [input_ids, causal_mask])

    # Convert traced TorchScript to Core ML format
    query_length = ct.RangeDim(lower_bound=1, upper_bound=max_context_size, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=max_context_size, default=1)
    inputs: List[ct.TensorType] = [
        ct.TensorType(shape=(1, query_length), dtype=np.int32, name="inputIds"),
        ct.TensorType(
            shape=(1, 1, query_length, end_step_dim),
            dtype=np.float16,
            name="causalMask",
        ),
    ]
    outputs: List[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
    states: List[ct.StateType] = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
            name="valueCache",
        ),
    ]

    # Convert model with FP16 precision
    mlmodel_fp16: ct.MLModel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=True,
    )

    # Block-wise quantize model weights to int4
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=32,
    )
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(mlmodel_fp16, config=config)
    mlmodel_int4._spec.description.metadata.userDefined.update({METADATA_TOKENIZER: MODEL_ID})
    mlmodel_int4.save("StatefulMistral7BInstructInt4.mlpackage")


if __name__ == "__main__":
    export()
