import functools
from typing import List, Optional

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float

from refusal_direction.pipeline.model_utils.model_base import (
    ModelBase,
    get_preferred_device,
    get_preferred_dtype,
)
from refusal_direction.pipeline.utils.utils import get_orthogonalized_matrix

# Gemma 2 chat template documented at https://ai.google.dev/gemma/docs/formatting
# (identical to Gemma 1 instruct formatting).
GEMMA2_CHAT_TEMPLATE = (
    """<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"""
)


# Gemma 2 instruct models follow the same constraints as Gemma 1: they ignore system prompts.
def format_instruction_gemma2_chat(
    instruction: str,
    output: Optional[str] = None,
    system: Optional[str] = None,
    include_trailing_whitespace: bool = True,
) -> str:
    if system is not None:
        raise ValueError("System prompts are not supported for Gemma 2 models.")

    formatted_instruction = GEMMA2_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_gemma2_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: Optional[List[str]] = None,
    system: Optional[str] = None,
    include_trailing_whitespace: bool = True,
):
    if outputs is not None:
        prompts = [
            format_instruction_gemma2_chat(
                instruction=instruction,
                output=output,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_gemma2_chat(
                instruction=instruction,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in instructions
        ]

    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


def _orthogonalize_linear_weight(
    linear_module: torch.nn.Linear, direction: Float[Tensor, "d_model"]
):
    linear_module.weight.data = get_orthogonalized_matrix(
        linear_module.weight.data.T, direction
    ).T


def orthogonalize_gemma2_weights(
    model: AutoModelForCausalLM, direction: Float[Tensor, "d_model"]
):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction
    )

    for block in model.model.layers:
        _orthogonalize_linear_weight(block.self_attn.o_proj, direction)
        _orthogonalize_linear_weight(block.mlp.down_proj, direction)


def act_add_gemma2_weights(
    model, direction: Float[Tensor, "d_model"], coeff: float, layer: int
):
    target_layer = model.model.layers[layer - 1]
    dtype = target_layer.mlp.down_proj.weight.dtype
    device = target_layer.mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)
    target_layer.mlp.down_proj.bias = torch.nn.Parameter(bias)


class Gemma2Model(ModelBase):
    def _load_model(self, model_path: str, dtype: Optional[torch.dtype] = None):
        target_device = get_preferred_device()
        resolved_dtype = dtype or get_preferred_dtype(target_device)
        device_map = "auto" if target_device.type == "cuda" else None

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=resolved_dtype,
            device_map=device_map,
        ).eval()

        if device_map is None:
            model.to(target_device)

        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_gemma2_chat,
            tokenizer=self.tokenizer,
            system=None,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        suffix = GEMMA2_CHAT_TEMPLATE.split("{instruction}")[-1]
        return self.tokenizer.encode(suffix, add_special_tokens=False)

    def _get_refusal_toks(self):
        # Most safety benchmarks look for the assistant beginning responses with "I".
        return self.tokenizer.encode("I", add_special_tokens=False)

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            block.self_attn for block in self.model_block_modules
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(block.mlp for block in self.model_block_modules)

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_gemma2_weights, direction=direction)

    def _get_act_add_mod_fn(
        self, direction: Float[Tensor, "d_model"], coeff: float, layer: int
    ):
        return functools.partial(
            act_add_gemma2_weights, direction=direction, coeff=coeff, layer=layer
        )
