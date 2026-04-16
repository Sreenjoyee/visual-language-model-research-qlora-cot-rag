"""Visual projector — Perceiver Resampler.

SRS §5: "compress -> expand to LLaMA embedding space; bf16 output."

This module is TRAINABLE in Stage 1. It's the only path by which visual
information reaches the LLM, so Stage-1 loss curves and a zero-vision
ablation (§19.2 module-2) are the primary signals that it's actually doing
something.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    """Cross-attention from learned latent queries to vision tokens.

    Design:
        - K learned latent queries of dim D_llm
        - L stacked blocks of (cross-attn from latents to vision) + (FFN)
        - Final LayerNorm
        - Output cast to bf16 to match LLaMA compute dtype

    No gated/tanh tricks — keeping it boring on purpose. If Stage-1 doesn't
    converge, upgrade this; don't preemptively complicate.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_latents: int = 8,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.llm_dim = llm_dim

        # Project vision features to LLM dim so cross-attn k/v live in the same space
        self.vision_proj = nn.Linear(vision_dim, llm_dim)
        self.vision_norm = nn.LayerNorm(llm_dim)

        # Learned latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, llm_dim) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm_q": nn.LayerNorm(llm_dim),
                        "norm_kv": nn.LayerNorm(llm_dim),
                        "attn": nn.MultiheadAttention(
                            embed_dim=llm_dim,
                            num_heads=num_heads,
                            batch_first=True,
                        ),
                        "norm_ffn": nn.LayerNorm(llm_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(llm_dim, llm_dim * ffn_mult),
                            nn.GELU(),
                            nn.Linear(llm_dim * ffn_mult, llm_dim),
                        ),
                    }
                )
            )
        self.out_norm = nn.LayerNorm(llm_dim)

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_tokens: (B, N, C_vision) from the vision encoder.

        Returns:
            (B, K, D_llm) in bf16, ready to splice into inputs_embeds.
        """
        B = vision_tokens.shape[0]
        # Compute in fp32 for numerical stability, cast at the end
        v = self.vision_norm(self.vision_proj(vision_tokens.float()))  # (B, N, D)
        q = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()   # (B, K, D)

        for block in self.layers:
            q_norm = block["norm_q"](q)
            kv_norm = block["norm_kv"](v)
            attn_out, _ = block["attn"](q_norm, kv_norm, kv_norm, need_weights=False)
            q = q + attn_out
            q = q + block["ffn"](block["norm_ffn"](q))

        q = self.out_norm(q)
        return q.to(torch.bfloat16)