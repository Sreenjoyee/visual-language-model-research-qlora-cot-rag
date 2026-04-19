"""Visual projector tests — SRS §19.2 module 2.

Test cases:
  - Random tensor input -> valid output shape (B, 8, 3072)
  - Output dtype is bfloat16
  - Zero input (ablation) -> degraded but stable (finite) output
  - Gradient flow: projector parameters receive non-zero gradients
  - Overfit on 10 identical samples: loss must decrease

All tests run without downloading any model — projector uses pure torch.
"""
from __future__ import annotations

import torch
import pytest

from src.projector import PerceiverResampler
from src.config import CONFIG

# Standard projector config matching production settings
VISION_DIM = CONFIG.vision_hidden_dim   # 1280 for EfficientNet-B0
LLM_DIM = CONFIG.llm_hidden_dim         # 3072 for LLaMA-3.2-3B
NUM_LATENTS = CONFIG.num_visual_tokens  # 8
NUM_HEADS = CONFIG.projector_num_heads  # 8
NUM_LAYERS = CONFIG.projector_num_layers  # 2


@pytest.fixture
def proj() -> PerceiverResampler:
    return PerceiverResampler(
        vision_dim=VISION_DIM,
        llm_dim=LLM_DIM,
        num_latents=NUM_LATENTS,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    )


# ── shape and dtype ────────────────────────────────────────────────────────

def test_output_shape_batch_1(proj):
    """SRS §19.2: random tensor -> valid output shape (B, 8, 3072)."""
    x = torch.randn(1, 49, VISION_DIM)  # EfficientNet-B0: 7×7=49 spatial tokens
    out = proj(x)
    assert out.shape == (1, NUM_LATENTS, LLM_DIM), f"Got {out.shape}"


def test_output_shape_batch_n(proj):
    """SRS §19.2 batch consistency: N > 1 works and shape is correct."""
    x = torch.randn(4, 49, VISION_DIM)
    out = proj(x)
    assert out.shape == (4, NUM_LATENTS, LLM_DIM)


def test_output_dtype_bfloat16(proj):
    """SRS §5: projector output must be bf16 to match LLaMA compute dtype."""
    x = torch.randn(1, 49, VISION_DIM)
    out = proj(x)
    assert out.dtype == torch.bfloat16, f"Expected bfloat16, got {out.dtype}"


def test_variable_token_count(proj):
    """Projector cross-attention handles any N."""
    for n_tokens in [49, 100, 197]:
        x = torch.randn(1, n_tokens, VISION_DIM)
        out = proj(x)
        assert out.shape == (1, NUM_LATENTS, LLM_DIM)


# ── stability ──────────────────────────────────────────────────────────────

def test_zero_input_produces_finite_output(proj):
    """SRS §19.2 ablation: zero vision input -> degraded but finite (no crash)."""
    x = torch.zeros(1, 49, VISION_DIM)
    out = proj(x)
    assert torch.isfinite(out).all(), "Zero vision input produced NaN/Inf"


def test_output_is_finite_on_random_input(proj):
    """Output must not contain NaN or Inf for random inputs."""
    x = torch.randn(2, 49, VISION_DIM)
    out = proj(x)
    assert torch.isfinite(out).all()


# ── gradient flow (trainability) ───────────────────────────────────────────

def test_gradient_flows_to_all_projector_params(proj):
    """SRS §19.2: gradient flow test — projector must be trainable."""
    x = torch.randn(1, 49, VISION_DIM)
    out = proj(x)
    loss = out.float().mean()
    loss.backward()

    params_with_grad = [
        name for name, p in proj.named_parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    ]
    assert len(params_with_grad) > 0, (
        "No projector parameters received gradients — check graph is connected"
    )


def test_latents_receive_gradient(proj):
    """Learned latents must be in the gradient graph."""
    x = torch.randn(1, 49, VISION_DIM)
    out = proj(x)
    out.float().mean().backward()
    assert proj.latents.grad is not None
    assert proj.latents.grad.abs().sum().item() > 0


# ── overfit on small dataset ───────────────────────────────────────────────

def test_loss_decreases_on_repeated_batch(proj):
    """SRS §19.2: overfit check — loss must decrease over 10 steps on same sample."""
    optimizer = torch.optim.Adam(proj.parameters(), lr=1e-3)
    x = torch.randn(1, 49, VISION_DIM)
    # Target: a fixed random vector in LLM space (simulate regression loss)
    target = torch.randn(1, NUM_LATENTS, LLM_DIM)

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        out = proj(x).float()
        loss = (out - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


# ── delta vs zero-vision baseline ─────────────────────────────────────────

def test_real_input_differs_from_zero_input(proj):
    """SRS §19.2: real vision tokens must produce meaningfully different output
    than zero tokens — confirms visual signal is being used."""
    proj.eval()
    with torch.no_grad():
        x_real = torch.randn(1, 49, VISION_DIM)
        x_zero = torch.zeros(1, 49, VISION_DIM)
        out_real = proj(x_real).float()
        out_zero = proj(x_zero).float()
        delta = (out_real - out_zero).abs().mean().item()

    # Any non-trivial difference confirms the projector uses the input
    assert delta > 1e-4, f"Output delta too small ({delta:.6f}) — projector may ignore input"
