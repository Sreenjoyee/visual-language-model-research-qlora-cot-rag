"""LLM loader — LLaMA-3.2-3B-Instruct, 4-bit NF4, bf16 compute.

SRS §5 & §9: "no CPU offload bugs." We pass `max_memory` explicitly so that
if the model can't fit, loading fails LOUDLY rather than silently offloading
layers to CPU RAM.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import Config


@dataclass
class LoadedLLM:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    hidden_dim: int
    device: torch.device


def _build_max_memory(config: Config) -> dict:
    """Map device index -> memory budget string (e.g. {0: '6GiB'})."""
    if not config.device.startswith("cuda"):
        # CPU-only path is allowed for smoke-testing shape logic, but real
        # inference will be painfully slow. Caller should know this.
        return {"cpu": f"{int(config.max_vram_gb * 2)}GiB"}
    idx = 0
    if ":" in config.device:
        idx = int(config.device.split(":")[1])
    return {idx: f"{config.max_vram_gb:.2f}GiB"}


def load_llm(config: Config) -> LoadedLLM:
    """Load the 4-bit LLM + tokenizer.

    Failure modes this guards against:
        - Silent CPU offload: prevented by explicit max_memory.
        - Missing pad token: set to eos_token.
        - Compute dtype mismatch: explicitly bf16.
        - CUDA incompatibility: falls back to CPU loading.
    """
    config.validate()

    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_model_id,
        token=config.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
        print("[load_llm] No CUDA GPU detected — loading in float32 on CPU (slow but functional).")

    _CPU_FALLBACK_TRIGGERS = ("CUDA", "no kernel image", "accelerator device", "available devices are []")

    try:
        if cuda_available:
            model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_id,
                token=config.hf_token,
                quantization_config=bnb_config,
                dtype=torch.bfloat16,
                device_map="auto",
                max_memory=_build_max_memory(config),
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_id,
                token=config.hf_token,
                dtype=torch.float32,
                device_map="cpu",
            )
        model.eval()

        # Verify no layers silently ended up on CPU (only meaningful with a real GPU).
        if cuda_available:
            for name, param in model.named_parameters():
                if param.device.type == "cpu":
                    raise RuntimeError(
                        f"Parameter '{name}' is on CPU — silent offload detected. "
                        f"Raise MEDDIAG_MAX_VRAM_GB or use a bigger GPU. "
                        f"Current budget: {config.max_vram_gb} GiB."
                    )
    except RuntimeError as e:
        if any(trigger in str(e) for trigger in _CPU_FALLBACK_TRIGGERS):
            print(
                f"[load_llm] GPU error: {e}\n"
                "Falling back to CPU (float32, no quantization)."
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_id,
                token=config.hf_token,
                dtype=torch.float32,
                device_map="cpu",
            )
            model.eval()
        else:
            raise

    hidden_dim = model.config.hidden_size
    device = next(model.parameters()).device
    return LoadedLLM(model=model, tokenizer=tokenizer, hidden_dim=hidden_dim, device=device)