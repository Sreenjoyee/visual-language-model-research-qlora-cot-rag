"""Vision encoder — frozen lightweight CNN feature extractor.

SRS §5: "frozen; no pooling-layer hacks; outputs spatial tokens."
Uses AutoModel so the backbone is swappable via config.vision_model_id without
code changes. Current default: google/efficientnet-b0 (5.3M params, ImageNet)
→ last_hidden_state (B, 1280, 7, 7) reshaped to (B, 49, 1280) spatial tokens.

CNN models return 4D output (B, C, H, W); ViT models return 3D (B, N, C).
The forward() method handles both via an ndim check before the projector sees it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from .config import CONFIG, Config


class VisionEncoder(nn.Module):
    """Frozen vision encoder producing spatial tokens.

    EfficientNet-B0 at 224×224: last_hidden_state is (B, 1280, 7, 7).
    forward() reshapes CNN 4D output → (B, 49, 1280) before returning.
    ViT-like models that already return (B, N, C) pass through unchanged.
    Actual N and C are derived at runtime via properties.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.processor = AutoImageProcessor.from_pretrained(
            config.vision_model_id,
            use_fast=True,
        )
        self.model = AutoModel.from_pretrained(config.vision_model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Derived dims — filled on first forward pass
        self._num_tokens: int | None = None
        self._hidden_dim: int | None = None
        # Cached after first forward: True = CNN 4D output, False = ViT 3D output
        self._output_is_4d: bool | None = None

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

        Handles both CNN output (B, C, H, W) and ViT output (B, N, C).
        CNN 4D tensors are permuted and flattened to (B, H*W, C) so the
        projector always receives a consistent 3D sequence.
        """
        out = self.model(pixel_values=pixel_values, return_dict=True)
        tokens = out.last_hidden_state

        # Resolve and cache the ndim branch on first call.
        if self._output_is_4d is None:
            self._output_is_4d = tokens.ndim == 4
            if not self._output_is_4d and tokens.ndim != 3:
                raise RuntimeError(
                    f"Unexpected vision model output shape {tuple(tokens.shape)}; "
                    "expected (B, N, C) or (B, C, H, W)."
                )

        if self._output_is_4d:
            # CNN spatial map: (B, C, H, W) -> (B, H*W, C)
            B, C, H, W = tokens.shape
            tokens = tokens.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

        # Finiteness check is expensive on every forward — only run in debug mode.
        if CONFIG.debug_vision and not torch.isfinite(tokens).all():
            raise RuntimeError("Vision encoder produced non-finite values.")

        self._num_tokens = tokens.shape[1]
        self._hidden_dim = tokens.shape[2]
        return tokens