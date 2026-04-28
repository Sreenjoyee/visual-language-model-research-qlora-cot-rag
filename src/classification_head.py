"""Binary classification head — visual features + RAG embeddings → NORMAL/ABNORMAL.

Sits parallel to LLaMA: takes Perceiver output and retrieved snippet embeddings,
produces a binary logit directly from visual evidence rather than from generated text.

This fixes the core architectural gap identified in experiments 1/4a/4b:
classification was previously derived by parsing LLaMA's text, which ignored
vision features and was trivially overridden by RAG prompts (sycophancy).

Design:
    - Pool Perceiver latents (mean across K=8 tokens) → (B, D_llm)
    - Project + pool RAG embeddings (mean across k snippets) → (B, D_llm)
    - LayerNorm each branch independently before fusion
    - Concatenate → MLP → 2 logits
    - Learnable temperature scalar for calibration (fixes ECE without retraining)

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
    ):
        super().__init__()

        # Visual branch: normalise pooled Perceiver output
        self.vis_norm = nn.LayerNorm(llm_dim)

        # RAG branch: project snippet embeddings to LLM dim, then normalise
        self.rag_proj = nn.Linear(rag_dim, llm_dim)
        self.rag_norm = nn.LayerNorm(llm_dim)

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(llm_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
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
        # Pool visual latents across K tokens
        vis = self.vis_norm(perceiver_out.float().mean(dim=1))       # (B, D_llm)

        # Project RAG embeddings to LLM dim, pool across k snippets
        rag = self.rag_norm(self.rag_proj(rag_embeddings.float()).mean(dim=1))  # (B, D_llm)

        x = torch.cat([vis, rag], dim=1)                             # (B, D_llm * 2)
        logits = self.mlp(x)                                         # (B, 2)

        # Clamp temperature away from zero to prevent division instability
        return logits / self.temperature.clamp(min=0.1)
