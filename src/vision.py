"""Vision encoder — frozen Google ViT-B/16.

SRS §5: "frozen; no pooling-layer hacks; outputs spatial tokens."
We pull last_hidden_state from the base model, which gives spatial features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, ViTModel

from .config import Config


class VisionEncoder(nn.Module):
    """Frozen ViT wrapper producing spatial tokens.

    Output shape for 224x224 input: (B, N, C) where N=197 (14x14 + 1 cls) and C=768.
    The actual N and C are derived at runtime and exposed as attributes — never
    assume shapes; pipeline.py uses these to wire the projector.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.processor = AutoImageProcessor.from_pretrained(
            config.vision_model_id,
            use_fast=True,
        )
        self.model = ViTModel.from_pretrained(config.vision_model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Derived dims — filled on first forward
        self._num_tokens: int | None = None
        self._hidden_dim: int | None = None

    @property
    def num_tokens(self) -> int:
        if self._num_tokens is None:
            raise RuntimeError("Run a forward pass first to populate shape metadata.")
        return self._num_tokens

    @property
    def hidden_dim(self) -> int:
        if self._hidden_dim is None:
            raise RuntimeError("Run a forward pass first to populate shape metadata.")
        return self._hidden_dim

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> normalized tensor (1, 3, H, W).

        Handles grayscale by converting to RGB. Does not cache or write to disk.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        batch = self.processor(images=image, return_tensors="pt")
        return batch["pixel_values"]

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, 3, H, W) -> (B, N, C)

        Uses last_hidden_state, which for CLIP ViT is already (B, N, C).
        """
        out = self.model(pixel_values=pixel_values, return_dict=True)
        # last_hidden_state shape: (B, N, C)
        tokens = out.last_hidden_state
        if tokens.ndim != 3:
            raise RuntimeError(
                f"Unexpected vision model output shape {tuple(tokens.shape)}; "
                "expected (B, N, C)."
            )

        # Sanity: no NaN/Inf — §19.2 module-1 validation requirement
        if not torch.isfinite(tokens).all():
            raise RuntimeError("Vision encoder produced non-finite values.")

        self._num_tokens = tokens.shape[1]
        self._hidden_dim = tokens.shape[2]
        return tokens