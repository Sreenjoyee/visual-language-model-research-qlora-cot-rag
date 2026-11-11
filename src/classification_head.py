"""Binary classification head — visual features + RAG embeddings → NORMAL/ABNORMAL.

Sits parallel to LLaMA: takes Perceiver output and retrieved snippet embeddings,
produces a binary logit directly from visual evidence rather than from generated text.

This fixes the core architectural gap identified in experiments 1/4a/4b:
classification was previously derived by parsing LLaMA's text, which ignored
vision features and was trivially overridden by RAG prompts (sycophancy).

Design — arch="paper" (default; MEDDIAG paper §3.8, exactly):
    visual_pool = mean_pool(visual_tokens)          # (B, 3072)
    rag_pool    = mean_pool(RAG_embeddings)         # (B, 384)
    fused       = concat(visual_pool, rag_pool)     # (B, 3456)
    h           = GELU(Linear(3456 → 512)(fused))
    logits      = Linear(512 → 2)(h)
    plus a scalar temperature (init 1 = identity) that the post-hoc
    temperature-scaling step (§4.7, experiments/calibrate_temperature.py) writes
    back into the saved weights.

arch="projected" (legacy, pre-paper runs): RAG projected 384→3072, LayerNorm on
both branches, 6144→512→GELU→LayerNorm→Dropout→2. Kept only so old checkpoints
still load; ClassificationHead.from_state_dict() detects the layout automatically.

Checkpoint: saved separately to models/cls_head.pt (SRS §15 checkpoint separation).
Trained in Stage 2 alongside LoRA; input_norm in projector is also unfrozen then.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Visual + RAG binary classifier.

    Args:
        llm_dim:    Dimension of Perceiver output tokens (matches LLM hidden dim).
        rag_dim:    Dimension of retrieved snippet embeddings (MiniLM = 384).
        hidden_dim: Hidden dimension of the classification MLP.
    """

    def __init__(
        self,
        llm_dim: int = 3072,
        rag_dim: int = 384,
        hidden_dim: int = 512,
        pool: str = "mean",
        arch: str = "paper",
    ):
        super().__init__()
        if arch not in ("paper", "projected"):
            raise ValueError(f"unknown ClassificationHead arch {arch!r} (use 'paper' or 'projected')")
        self.arch = arch

        # Visual pooling over the K Perceiver tokens. "mean" is the paper's design;
        # "attn" (learned per-token weights) is an optional extension. The mode is
        # inferable from the state_dict (vis_attn.*), so checkpoints self-describe.
        self.pool = pool
        if pool == "attn":
            self.vis_attn = nn.Linear(llm_dim, 1)

        if arch == "paper":
            # §3.8: raw pooled features, concat (llm_dim + rag_dim) → 512 → GELU → 2
            self.mlp = nn.Sequential(
                nn.Linear(llm_dim + rag_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            # legacy layout (see module docstring)
            self.vis_norm = nn.LayerNorm(llm_dim)
            self.rag_proj = nn.Linear(rag_dim, llm_dim)
            self.rag_norm = nn.LayerNorm(llm_dim)
            self.mlp = nn.Sequential(
                nn.Linear(llm_dim * 2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 2),
            )

        # Learnable temperature — divides logits before softmax.
        # Initialises to 1 (identity). Trained end-to-end with the rest,
        # so calibration improves without a separate post-hoc step.
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        perceiver_out: torch.Tensor,   # (B, K, D_llm)
        rag_embeddings: torch.Tensor,  # (B, k, rag_dim)
    ) -> torch.Tensor:
        """Return raw logits of shape (B, 2). Index 0 = NORMAL, 1 = ABNORMAL."""
        # Pool visual latents across K tokens (mean, or learned attention)
        po = perceiver_out.float()
        if getattr(self, "pool", "mean") == "attn":
            w = torch.softmax(self.vis_attn(po), dim=1)              # (B, K, 1)
            pooled = (w * po).sum(dim=1)                             # (B, D_llm)
        else:
            pooled = po.mean(dim=1)                                  # (B, D_llm)

        if getattr(self, "arch", "projected") == "paper":
            rag = rag_embeddings.float().mean(dim=1)                 # (B, rag_dim)
            x = torch.cat([pooled, rag], dim=1)                      # (B, D_llm + rag_dim) = 3456
        else:
            vis = self.vis_norm(pooled)                              # (B, D_llm)
            rag = self.rag_norm(self.rag_proj(rag_embeddings.float()).mean(dim=1))  # (B, D_llm)
            x = torch.cat([vis, rag], dim=1)                         # (B, D_llm * 2)
        logits = self.mlp(x)                                         # (B, 2)

        # Clamp temperature away from zero to prevent division instability
        return logits / self.temperature.clamp(min=0.1)

    @classmethod
    def from_state_dict(cls, state: dict, llm_dim: int, rag_dim: int, hidden_dim: int) -> "ClassificationHead":
        """Build a head whose pool mode and layout match a saved state_dict, then load it."""
        head = cls(
            llm_dim=llm_dim, rag_dim=rag_dim, hidden_dim=hidden_dim,
            pool="attn" if any("vis_attn" in k for k in state) else "mean",
            arch="projected" if any(k.startswith("rag_proj") for k in state) else "paper",
        )
        head.load_state_dict(state)
        return head
