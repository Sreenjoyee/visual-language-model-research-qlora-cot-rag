"""Vision encoder — frozen lightweight CXR feature extractor.

SRS §5: "frozen; no pooling-layer hacks; outputs spatial tokens."
Default backbone: TorchXRayVision DenseNet-121 ("xrv:densenet121-res224-all",
~8M params, pretrained on the union of NIH/CheXpert/MIMIC/PadChest) →
features (B, 1024, 7, 7) reshaped to (B, 49, 1024) spatial tokens. Chosen for
cross-dataset (OOD) generalisation on chest X-rays.

An "xrv:<weights>" config.vision_model_id loads a TorchXRayVision backbone; any
other value loads a HuggingFace AutoModel (CNN 4D or ViT 3D output — forward()
handles both via an ndim check before the projector sees it).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from .config import CONFIG, Config


class VisionEncoder(nn.Module):
    """Frozen vision encoder producing spatial tokens.

    DenseNet-121 (TorchXRayVision) at 224×224: features() is (B, 1024, 7, 7),
    reshaped to (B, 49, 1024) before returning. A HuggingFace AutoModel backbone
    (CNN 4D or ViT 3D output) is also supported; actual N and C are derived at
    runtime via properties.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # "xrv:<weights>" selects a TorchXRayVision CXR-pretrained backbone (default:
        # DenseNet-121); any other id is loaded as a generic HuggingFace AutoModel.
        self._is_xrv = str(config.vision_model_id).startswith("xrv:")
        if self._is_xrv:
            import torchxrayvision as xrv
            weights = config.vision_model_id.split(":", 1)[1]
            self.processor = None
            self.model = (
                xrv.models.ResNet(weights=weights)
                if weights.startswith("resnet")
                else xrv.models.DenseNet(weights=weights)
            )
        else:
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
        # Run-4: when True the encoder is unfrozen and forward() builds a grad graph.
        self._finetune: bool = False

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
        """PIL image -> normalized tensor.

        TorchXRayVision backbones expect single-channel input normalised to the
        [-1024, 1024] range → (1, 1, H, W). HuggingFace backbones use their own
        image processor on an RGB image → (1, 3, H, W).
        """
        if self._is_xrv:
            import numpy as np
            import torchxrayvision as xrv
            size = self.config.image_size
            img = image.convert("L").resize((size, size))
            arr = np.asarray(img, dtype=np.float32)          # [0, 255]
            arr = xrv.datasets.normalize(arr, 255)           # -> [-1024, 1024]
            return torch.from_numpy(arr)[None, None, :, :]   # (1, 1, H, W)
        if image.mode != "RGB":
            image = image.convert("RGB")
        batch = self.processor(images=image, return_tensors="pt")
        return batch["pixel_values"]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, 3, H, W) -> (B, N, C)

        Runs under no_grad (frozen encoder) unless enable_finetune() has been called
        (Run-4 vision fine-tuning), in which case a gradient graph is built so the
        encoder weights can be updated.
        """
        if self._finetune:
            return self._encode(pixel_values)
        with torch.no_grad():
            return self._encode(pixel_values)

    def _encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Backbone forward + reshape to (B, N, C). Handles CNN 4D and ViT 3D output."""
        if self._is_xrv:
            # TorchXRayVision DenseNet: features() -> (B, 1024, 7, 7) spatial map.
            feats = torch.relu(self.model.features(pixel_values))
            B, C, H, W = feats.shape
            tokens = feats.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
            self._num_tokens = tokens.shape[1]
            self._hidden_dim = tokens.shape[2]
            return tokens

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

    def enable_finetune(self, last_n_blocks: int | None = None) -> list[nn.Parameter]:
        """Unfreeze encoder weights for fine-tuning (Run-4 AUROC lever).

        Keeps the module in eval() so BatchNorm running stats stay fixed — safe for
        batch-size-1 training; only conv/linear weights receive gradients. Unfreezes
        the last dense block for a TorchXRayVision DenseNet, or the whole backbone if
        the block structure can't be located. Returns the trainable parameters so the
        caller can add them to the optimizer.
        """
        self._finetune = True
        if self._is_xrv:
            feats = getattr(self.model, "features", None)
            targets = []
            if feats is not None and hasattr(feats, "denseblock4"):
                targets = [feats.denseblock4]
                if hasattr(feats, "norm5"):
                    targets.append(feats.norm5)
            params = ([p for t in targets for p in t.parameters()]
                      if targets else list(self.model.parameters()))
        else:
            blocks = getattr(getattr(self.model, "encoder", None), "blocks", None)
            if last_n_blocks and blocks is not None and len(blocks) >= last_n_blocks:
                params = [p for blk in list(blocks)[-last_n_blocks:] for p in blk.parameters()]
            else:
                params = list(self.model.parameters())
        for p in params:
            p.requires_grad = True
        self.model.eval()   # keep BatchNorm running stats fixed (batch size 1)
        return [p for p in params if p.requires_grad]