from abc import ABC, abstractmethod
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int, Float

from refusal_direction.pipeline.utils.hook_utils import add_hooks


def get_preferred_device() -> torch.device:
    env_device = os.environ.get("REFUSAL_DEVICE")
    if env_device is not None:
        requested = torch.device(env_device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "REFUSAL_DEVICE requested CUDA but torch.cuda.is_available() is False"
            )
        if requested.type == "mps":
            if (
                not hasattr(torch.backends, "mps")
                or not torch.backends.mps.is_available()
            ):
                raise RuntimeError(
                    "REFUSAL_DEVICE requested MPS but it is not available in this PyTorch build"
                )
        return requested

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_preferred_dtype(device: torch.device) -> torch.dtype:
    env_dtype = os.environ.get("REFUSAL_DTYPE")
    if env_dtype is not None:
        if not hasattr(torch, env_dtype):
            raise RuntimeError(
                f"Unknown dtype requested via REFUSAL_DTYPE: {env_dtype}"
            )
        return getattr(torch, env_dtype)

    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        # MPS is numerically unstable in fp16 for large models; prefer fp32 despite higher cost.
        return torch.float32
    return torch.float32


def get_high_precision_dtype(device: torch.device) -> torch.dtype:
    if device.type == "mps":
        return torch.float32
    return torch.float64


class ModelBase(ABC):
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.model: AutoModelForCausalLM = self._load_model(model_name_or_path)
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)

        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.eoi_toks = self._get_eoi_toks()
        self.refusal_toks = self._get_refusal_toks()

        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()
        first_param = next(self.model.parameters())
        self.device = first_param.device
        self.dtype = first_param.dtype
        print(
            f"[ModelBase] Loaded {self.model_name_or_path} on {self.device}"
            f" (dtype={self.dtype})"
        )

    def del_model(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        pass

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        pass

    @abstractmethod
    def _get_refusal_toks(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass

    @abstractmethod
    def _get_attn_modules(self):
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        pass

    @abstractmethod
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        pass

    @abstractmethod
    def _get_act_add_mod_fn(
        self, direction: Float[Tensor, "d_model"], coeff: float, layer: int
    ):
        pass

    def generate_completions(
        self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64
    ):
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens, do_sample=False
        )
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        instructions = [x["instruction"] for x in dataset]
        categories = [x["category"] for x in dataset]

        for i in tqdm(range(0, len(dataset), batch_size)):
            tokenized_instructions = self.tokenize_instructions_fn(
                instructions=instructions[i : i + batch_size]
            )

            with add_hooks(
                module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
            ):
                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(self.model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(
                        self.model.device
                    ),
                    generation_config=generation_config,
                )

                generation_toks = generation_toks[
                    :, tokenized_instructions.input_ids.shape[-1] :
                ]

                for generation_idx, generation in enumerate(generation_toks):
                    completions.append(
                        {
                            "category": categories[i + generation_idx],
                            "prompt": instructions[i + generation_idx],
                            "response": self.tokenizer.decode(
                                generation, skip_special_tokens=True
                            ).strip(),
                        }
                    )

        return completions
