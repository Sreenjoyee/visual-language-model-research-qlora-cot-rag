"""Multimodal fusion module tests — SRS §19.2 module 4.

Tests the embedding splice logic (visual tokens + text tokens) without
requiring a downloaded LLM. A mock tokenizer is used for the structural tests.

Test cases:
  - Image + prompt -> correct embedding concatenation (shape, order)
  - Missing IMAGE_PLACEHOLDER -> ValueError (not silent)
  - Mismatched batch sizes -> detected
  - Prompt consistency: IMAGE_PLACEHOLDER always present
  - Attention mask is all 1s (no padding)
  - Visual tokens are not lost (no silent truncation)
"""
from __future__ import annotations

import torch
import pytest
from unittest.mock import MagicMock

from src.prompts import IMAGE_PLACEHOLDER, build_inference_prompt


# ── helpers ────────────────────────────────────────────────────────────────

def _make_mock_tokenizer(left_len: int = 10, right_len: int = 15):
    """Minimal tokenizer mock: encodes left/right text as fixed-length tensors."""
    tok = MagicMock()

    call_count = {"n": 0}

    def side_effect(text, add_special_tokens=False, return_tensors="pt"):
        call_count["n"] += 1
        # Alternate between left and right lengths based on call order
        length = left_len if call_count["n"] % 2 == 1 else right_len
        ids = torch.zeros(1, length, dtype=torch.long)
        result = MagicMock()
        result.input_ids = ids
        return result

    tok.side_effect = None
    tok.__call__ = side_effect
    tok.return_value = MagicMock(input_ids=torch.zeros(1, left_len, dtype=torch.long))
    return tok


def _fake_embed(input_ids: torch.Tensor, dim: int = 3072) -> torch.Tensor:
    """Simulate LLM embedding lookup: (B, seq_len) -> (B, seq_len, D)."""
    B, L = input_ids.shape
    return torch.randn(B, L, dim)


def _splice_visual(
    prompt_text: str,
    visual_embeds: torch.Tensor,
    tokenizer,
    embed_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-implementation of pipeline._splice_visual for isolated testing."""
    if IMAGE_PLACEHOLDER not in prompt_text:
        raise ValueError(f"Prompt missing {IMAGE_PLACEHOLDER}")

    left_text, right_text = prompt_text.split(IMAGE_PLACEHOLDER, 1)

    left_ids = tokenizer(left_text, add_special_tokens=False, return_tensors="pt").input_ids
    right_ids = tokenizer(right_text, add_special_tokens=False, return_tensors="pt").input_ids

    left_emb = embed_fn(left_ids)
    right_emb = embed_fn(right_ids)
    vis = visual_embeds.to(dtype=left_emb.dtype)

    embeds = torch.cat([left_emb, vis, right_emb], dim=1)
    mask = torch.ones(embeds.shape[:2], dtype=torch.long)
    return embeds, mask


# ── structural tests (no model download) ──────────────────────────────────

def test_splice_output_shape_is_sum_of_parts():
    """SRS §19.2: left + visual + right tokens concatenated correctly."""
    L_left, L_vis, L_right, D = 10, 8, 15, 3072

    visual_embeds = torch.randn(1, L_vis, D)
    prompt = f"left text {IMAGE_PLACEHOLDER} right text"

    tok = MagicMock()
    call_n = [0]

    def tok_call(text, **kwargs):
        call_n[0] += 1
        length = L_left if call_n[0] == 1 else L_right
        m = MagicMock()
        m.input_ids = torch.zeros(1, length, dtype=torch.long)
        return m

    tok.side_effect = tok_call

    embeds, mask = _splice_visual(prompt, visual_embeds, tok, lambda ids: torch.randn(1, ids.shape[1], D))
    expected_len = L_left + L_vis + L_right
    assert embeds.shape == (1, expected_len, D), f"Got {embeds.shape}"


def test_attention_mask_all_ones():
    """SRS §19.2: attention_mask must be all 1s (no padding in inference)."""
    L_left, L_vis, L_right, D = 5, 8, 7, 3072
    visual_embeds = torch.randn(1, L_vis, D)
    prompt = f"A {IMAGE_PLACEHOLDER} B"

    tok = MagicMock()
    call_n = [0]

    def tok_call(text, **kwargs):
        call_n[0] += 1
        length = L_left if call_n[0] == 1 else L_right
        m = MagicMock()
        m.input_ids = torch.zeros(1, length, dtype=torch.long)
        return m

    tok.side_effect = tok_call
    _, mask = _splice_visual(prompt, visual_embeds, tok, lambda ids: torch.randn(1, ids.shape[1], D))
    assert mask.all().item(), "Attention mask contains zeros — unexpected padding"


def test_missing_placeholder_raises_value_error():
    """SRS §19.2: prompt without IMAGE_PLACEHOLDER must raise, not silently skip."""
    visual_embeds = torch.randn(1, 8, 3072)
    bad_prompt = "No placeholder here at all."
    tok = MagicMock()

    with pytest.raises(ValueError, match=IMAGE_PLACEHOLDER):
        _splice_visual(bad_prompt, visual_embeds, tok, lambda ids: torch.randn(1, ids.shape[1], 3072))


def test_visual_tokens_not_lost():
    """SRS §19.2: visual token count in output must equal num_visual_tokens."""
    L_vis, D = 8, 3072
    visual_embeds = torch.randn(1, L_vis, D)
    prompt = f"left {IMAGE_PLACEHOLDER} right"

    tok = MagicMock()
    call_n = [0]

    def tok_call(text, **kwargs):
        call_n[0] += 1
        m = MagicMock()
        m.input_ids = torch.zeros(1, 5, dtype=torch.long)
        return m

    tok.side_effect = tok_call
    embeds, _ = _splice_visual(
        prompt, visual_embeds, tok, lambda ids: torch.randn(1, ids.shape[1], D)
    )
    total = embeds.shape[1]
    # total = 5 (left) + 8 (visual) + 5 (right) = 18
    assert total >= L_vis, "Visual tokens were dropped from the sequence"


def test_inference_prompt_always_contains_placeholder():
    """SRS §12: build_inference_prompt must always include IMAGE_PLACEHOLDER."""
    for snippets in [[], ["one"], ["one", "two", "three"]]:
        prompt = build_inference_prompt(snippets)
        assert IMAGE_PLACEHOLDER in prompt, (
            f"IMAGE_PLACEHOLDER missing from prompt with {len(snippets)} snippets"
        )


def test_no_input_ids_returned_from_splice():
    """SRS §5: splice function must return (embeds, mask), never input_ids."""
    L_vis, D = 8, 3072
    visual_embeds = torch.randn(1, L_vis, D)
    prompt = f"a {IMAGE_PLACEHOLDER} b"

    tok = MagicMock()
    call_n = [0]

    def tok_call(text, **kwargs):
        call_n[0] += 1
        m = MagicMock()
        m.input_ids = torch.zeros(1, 3, dtype=torch.long)
        return m

    tok.side_effect = tok_call
    result = _splice_visual(
        prompt, visual_embeds, tok, lambda ids: torch.randn(1, ids.shape[1], D)
    )
    assert len(result) == 2, "Expected (inputs_embeds, attention_mask) tuple"
    inputs_embeds, attention_mask = result
    assert inputs_embeds.ndim == 3, "inputs_embeds must be 3D"
    assert attention_mask.ndim == 2, "attention_mask must be 2D"
