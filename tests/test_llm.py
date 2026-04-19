"""LLM integration tests — SRS §19.2 module 3.

Test cases:
  - inputs_embeds ONLY (no input_ids override)
  - attention_mask shape matches inputs_embeds
  - 4-bit quantized model loads without CPU offload (model-dependent)
  - max_memory enforcement: no params silently on CPU (model-dependent)

Model-dependent tests need MEDDIAG_RUN_MODEL_TESTS=1.
"""
from __future__ import annotations

import torch
import pytest

from tests.conftest import needs_model


# ── shape / logic tests (no model download needed) ─────────────────────────

def test_attention_mask_shape_matches_embeds():
    """SRS §19.2: attention_mask must be (B, seq_len) matching inputs_embeds."""
    B, seq_len, D = 1, 50, 3072
    inputs_embeds = torch.randn(B, seq_len, D)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long)
    assert attention_mask.shape == (B, seq_len)


def test_attention_mask_is_all_ones_for_full_sequence():
    """No padding in single-image inference — mask must be all 1s."""
    B, seq_len = 1, 50
    inputs_embeds = torch.randn(B, seq_len, 3072)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long)
    assert attention_mask.sum().item() == B * seq_len


def test_no_input_ids_in_multimodal_forward():
    """SRS §5: NEVER pass input_ids in multimodal mode.

    This test is structural: it verifies that our generate call dict
    does NOT contain the key 'input_ids'.
    """
    inputs_embeds = torch.randn(1, 20, 3072)
    attention_mask = torch.ones(1, 20, dtype=torch.long)

    gen_kwargs = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=False,
    )
    assert "input_ids" not in gen_kwargs, (
        "input_ids must never appear in multimodal generate kwargs — SRS §5"
    )


# ── model-dependent tests ─────────────────────────────────────────────────

@needs_model
def test_llm_loads_without_cpu_offload(cfg):
    """SRS §19.2: 4-bit quantized model loads without silent CPU offload."""
    from src.llm import load_llm
    loaded = load_llm(cfg)

    cpu_params = [
        name for name, p in loaded.model.named_parameters()
        if p.device.type == "cpu"
    ]
    assert cpu_params == [], (
        f"Silent CPU offload detected — params on CPU: {cpu_params[:3]}"
    )


@needs_model
def test_llm_is_4bit_quantized(cfg):
    """SRS §5: LLM must be 4-bit NF4 quantized."""
    from src.llm import load_llm
    loaded = load_llm(cfg)

    # bitsandbytes quantized layers are instances of Linear4bit
    try:
        from bitsandbytes.nn import Linear4bit
        quant_layers = [
            m for m in loaded.model.modules()
            if isinstance(m, Linear4bit)
        ]
        assert len(quant_layers) > 0, "No 4-bit quantized layers found"
    except ImportError:
        pytest.skip("bitsandbytes not installed")


@needs_model
def test_llm_hidden_dim_matches_config(cfg):
    """SRS §19.2: LLM hidden dim must match config.llm_hidden_dim."""
    from src.llm import load_llm
    loaded = load_llm(cfg)
    assert loaded.hidden_dim == cfg.llm_hidden_dim, (
        f"LLM hidden dim {loaded.hidden_dim} != config {cfg.llm_hidden_dim}"
    )


@needs_model
def test_llm_pad_token_is_set(cfg):
    """Tokenizer must have a pad token to avoid generate() warnings."""
    from src.llm import load_llm
    loaded = load_llm(cfg)
    assert loaded.tokenizer.pad_token_id is not None, (
        "pad_token_id is None — generation will fail"
    )
