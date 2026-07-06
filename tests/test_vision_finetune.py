"""Tests for Run-4 vision fine-tuning + train-time augmentation."""
import random

import pytest
from PIL import Image


# ── _augment_image (pure, no model) ───────────────────────────────────────────

def test_augment_image_preserves_size_and_returns_rgb():
    from experiments.stage2_classification import _augment_image
    img = Image.new("L", (64, 48), color=120)          # grayscale input
    out = _augment_image(img, random.Random(0))
    assert out.mode == "RGB"
    assert out.size == (64, 48)


def test_augment_image_deterministic_with_seed():
    from experiments.stage2_classification import _augment_image
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    a = _augment_image(img, random.Random(7))
    b = _augment_image(img, random.Random(7))
    assert list(a.getdata()) == list(b.getdata())


def test_augment_image_changes_pixels():
    from experiments.stage2_classification import _augment_image
    img = Image.new("RGB", (32, 32))
    img.putdata([(i % 256, (i * 2) % 256, (i * 3) % 256) for i in range(32 * 32)])
    out = _augment_image(img, random.Random(3))
    assert list(out.getdata()) != list(img.getdata())


# ── config flags ──────────────────────────────────────────────────────────────

def test_vision_config_defaults_are_run3(monkeypatch):
    for k in ("MEDDIAG_VISION_FINETUNE", "MEDDIAG_VISION_AUGMENT",
              "MEDDIAG_VISION_LR", "MEDDIAG_VISION_FT_BLOCKS"):
        monkeypatch.delenv(k, raising=False)
    from src.config import Config
    c = Config()
    assert c.vision_finetune is False and c.vision_augment is False
    assert c.vision_lr == 1e-5 and c.vision_finetune_blocks == 2


def test_vision_config_env_override(monkeypatch):
    monkeypatch.setenv("MEDDIAG_VISION_FINETUNE", "1")
    monkeypatch.setenv("MEDDIAG_VISION_AUGMENT", "true")
    monkeypatch.setenv("MEDDIAG_VISION_LR", "2e-5")
    monkeypatch.setenv("MEDDIAG_VISION_FT_BLOCKS", "3")
    from src.config import Config
    c = Config()
    assert c.vision_finetune is True and c.vision_augment is True
    assert c.vision_lr == 2e-5 and c.vision_finetune_blocks == 3


# ── VisionEncoder.enable_finetune (loads the real backbone; skips if unavailable) ─

@pytest.fixture(scope="module")
def vision_encoder():
    from src.config import CONFIG
    from src.vision import VisionEncoder
    try:
        return VisionEncoder(CONFIG)
    except Exception as e:  # noqa: BLE001 — no network / model cache in this env
        pytest.skip(f"vision backbone unavailable: {e}")


def test_vision_finetune_lifecycle(vision_encoder):
    import torch
    px = torch.randn(1, 3, 224, 224)

    # frozen by default: forward output carries no grad graph
    assert vision_encoder._finetune is False
    assert vision_encoder(px).requires_grad is False

    # enable fine-tuning -> params unfrozen, forward builds a grad graph
    params = vision_encoder.enable_finetune(last_n_blocks=2)
    assert vision_encoder._finetune is True
    assert len(params) > 0 and all(p.requires_grad for p in params)

    out = vision_encoder(px)
    assert out.requires_grad
    out.sum().backward()
    assert any(p.grad is not None for p in params)   # grads actually reach the encoder
