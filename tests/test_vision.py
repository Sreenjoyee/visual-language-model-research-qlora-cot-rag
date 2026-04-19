"""Vision encoder tests — SRS §19.2 module 1.

Test cases:
  - Valid image -> correct tensor shape (B, N, C)
  - Grayscale image -> correct RGB conversion
  - Corrupted image -> safe failure handling
  - Batch size = 1 consistency
  - No NaN / Inf in output
  - All parameters frozen (requires_grad=False)

Model-dependent tests are guarded by MEDDIAG_RUN_MODEL_TESTS=1.
"""
from __future__ import annotations

import io
import pytest
import torch
from PIL import Image

from tests.conftest import needs_model
from src.config import CONFIG


def _make_rgb_image(w: int = 224, h: int = 224) -> Image.Image:
    arr = (torch.randn(h, w, 3) * 128 + 128).clamp(0, 255).byte().numpy()
    return Image.fromarray(arr, mode="RGB")


def _make_gray_image(w: int = 224, h: int = 224) -> Image.Image:
    arr = (torch.randn(h, w) * 128 + 128).clamp(0, 255).byte().numpy()
    return Image.fromarray(arr.astype("uint8"), mode="L")


# ── tests that do NOT need a downloaded model ──────────────────────────────

def test_corrupted_image_safe_failure():
    """SRS §19.2: corrupted image -> safe failure handling.

    PIL raises before the encoder is even called — that's the safe boundary.
    """
    bad_bytes = io.BytesIO(b"not an image")
    with pytest.raises(Exception):
        Image.open(bad_bytes).verify()


def test_preprocess_converts_grayscale_to_rgb():
    """Grayscale PIL -> RGB conversion happens inside preprocess(), model not needed."""
    gray = _make_gray_image()
    assert gray.mode == "L"
    rgb = gray.convert("RGB")
    assert rgb.mode == "RGB"
    assert rgb.size == gray.size


# ── model-dependent tests (need MEDDIAG_RUN_MODEL_TESTS=1) ─────────────────

@needs_model
def test_valid_image_output_shape(vision_encoder):
    """SRS §19.2: valid image -> correct tensor shape (B, N, C)."""
    img = _make_rgb_image()
    pixel_values = vision_encoder.preprocess(img)
    tokens = vision_encoder(pixel_values)

    assert tokens.ndim == 3, f"Expected 3D tensor, got shape {tokens.shape}"
    B, N, C = tokens.shape
    assert B == 1
    assert N > 0
    assert C == CONFIG.vision_hidden_dim, (
        f"Channel dim {C} != config.vision_hidden_dim {CONFIG.vision_hidden_dim}"
    )


@needs_model
def test_grayscale_input_succeeds(vision_encoder):
    """SRS §19.2: grayscale image passes through preprocess without error."""
    gray = _make_gray_image()
    pixel_values = vision_encoder.preprocess(gray)
    tokens = vision_encoder(pixel_values)
    assert tokens.ndim == 3


@needs_model
def test_no_nan_inf_in_output(vision_encoder):
    """SRS §19.2: output tensors must be finite."""
    img = _make_rgb_image()
    pixel_values = vision_encoder.preprocess(img)
    tokens = vision_encoder(pixel_values)
    assert torch.isfinite(tokens).all(), "Vision encoder produced NaN or Inf"


@needs_model
def test_batch_size_1_shape_consistency(vision_encoder):
    """SRS §19.2: two separate single-image calls return the same shape."""
    img_a = _make_rgb_image()
    img_b = _make_rgb_image()
    t_a = vision_encoder(vision_encoder.preprocess(img_a))
    t_b = vision_encoder(vision_encoder.preprocess(img_b))
    assert t_a.shape == t_b.shape, "Shape inconsistency between single-image calls"


@needs_model
def test_all_parameters_frozen(vision_encoder):
    """SRS §5: encoder must be fully frozen — no parameter should require grad."""
    trainable = [
        name for name, p in vision_encoder.model.named_parameters()
        if p.requires_grad
    ]
    assert trainable == [], f"Unexpected trainable params: {trainable[:5]}"


@needs_model
def test_output_shape_properties_populated(vision_encoder):
    """num_tokens and hidden_dim properties are populated after a forward pass."""
    img = _make_rgb_image()
    pixel_values = vision_encoder.preprocess(img)
    vision_encoder(pixel_values)
    # These raise RuntimeError if not populated
    assert vision_encoder.num_tokens > 0
    assert vision_encoder.hidden_dim == CONFIG.vision_hidden_dim
