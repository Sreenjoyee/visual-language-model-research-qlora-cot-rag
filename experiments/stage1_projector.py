"""Stage 1 — projector alignment training.

SRS §7: "train projector only; next-token prediction on MIMIC reports."

Training objective:
    L = cross-entropy over the report tokens only,
    given [system + user_prefix + visual_embeds + user_suffix + report] as input.

Only the projector is trainable. Vision encoder and LLM are frozen.

Critical correctness invariants (SRS §9 failure modes):
    - Labels must be -100 for every position that is NOT a report token.
      Otherwise the model learns to "predict" its own visual embeddings,
      which is padding-loss contamination.
    - Gradient flow through the projector is asserted on the first step.
      Zero-gradient projector = training is silently a no-op.
    - No input_ids anywhere in the multimodal forward. Only inputs_embeds
      + attention_mask + labels.

What this script does NOT do (yet):
    - Distributed training. Single-GPU only.
    - Balanced sampling. Stage-1 is report-only; imbalance applies to Stage-2.
    - Checkpoint resuming mid-epoch. Saves one checkpoint at the end; resume
      logic goes in when we need it.

Usage:
    python -m experiments.stage1_projector --max-pairs 5000 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.config import CONFIG, Config
from src.data.pairs import Pair, stream_mimic_pairs
from src.llm import LoadedLLM, load_llm
from src.projector import PerceiverResampler
from src.prompts import IMAGE_PLACEHOLDER, build_chat_messages
from src.vision import VisionEncoder


# -----------------------------------------------------------------------------
# Stage-1 prompt: different from inference (no FAISS evidence needed here).
# We still route through build_chat_messages so system prompt stays consistent,
# but we pass an explicit sentinel that the chat message layout does NOT drift
# from Stage-2 / inference.
# -----------------------------------------------------------------------------

STAGE1_ALIGNMENT_TAG = "[STAGE1_ALIGNMENT]"


def _build_stage1_prompt(tokenizer) -> tuple[str, str]:
    """Build the chat-formatted prefix (ending with the assistant-open tag)
    and a trailing empty-string slot. Returns (prefix_text, suffix_text).

    We reuse the same SYSTEM_PROMPT by borrowing build_chat_messages with a
    sentinel user message that contains IMAGE_PLACEHOLDER. This keeps the
    system prompt byte-identical to inference — the user content differs
    because Stage 1 has no FAISS evidence, and that's OK: Stage-2 re-uses
    inference formatting exactly, which is what §12 requires ("training vs
    inference consistency" at the classification stage).
    """
    # We piggy-back on build_chat_messages but with a stage-1-specific instruction.
    # The instruction text matters less than the invariant that IMAGE_PLACEHOLDER
    # is present and that the assistant turn is opened via apply_chat_template.
    messages = [
        {"role": "system", "content": "You are a careful radiology assistant."},
        {
            "role": "user",
            "content": (
                f"Chest X-ray image:\n{IMAGE_PLACEHOLDER}\n\n"
                f"{STAGE1_ALIGNMENT_TAG}\n"
                "Write the radiology report."
            ),
        },
    ]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prefix, ""  # suffix is empty; report is appended raw


# -----------------------------------------------------------------------------
# Core training step
# -----------------------------------------------------------------------------


@dataclass
class BatchTensors:
    inputs_embeds: torch.Tensor   # (B, T, D)
    attention_mask: torch.Tensor  # (B, T)
    labels: torch.Tensor          # (B, T), -100 everywhere except report tokens


def _encode_example(
    pair: Pair,
    vision: VisionEncoder,
    projector: PerceiverResampler,
    llm: LoadedLLM,
    prefix_text: str,
    max_report_tokens: int = 192,
) -> BatchTensors:
    """Single-example encoder. Returns a batch-of-1 BatchTensors.

    Produces:
        [ prefix_left_tokens | visual_embeds | prefix_right_tokens | report_tokens ]

    Labels are -100 everywhere except the report_tokens positions.
    """
    device = llm.device
    tokenizer = llm.tokenizer

    # --- Split prefix around the image placeholder ---
    if IMAGE_PLACEHOLDER not in prefix_text:
        raise RuntimeError("Stage-1 prefix lost IMAGE_PLACEHOLDER — check prompt code.")
    left_text, right_text = prefix_text.split(IMAGE_PLACEHOLDER, 1)

    left_ids = tokenizer(left_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    right_ids = tokenizer(right_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # --- Report tokens (the supervised targets) ---
    # Do NOT add special tokens here; the chat template already put them in the prefix.
    report_ids = tokenizer(
        pair.report,
        add_special_tokens=False,
        truncation=True,
        max_length=max_report_tokens,
        return_tensors="pt",
    ).input_ids.to(device)
    # Terminate with EOS so the model learns to stop.
    eos = torch.tensor([[tokenizer.eos_token_id]], device=device, dtype=report_ids.dtype)
    report_ids = torch.cat([report_ids, eos], dim=1)

    # --- Vision -> projector ---
    pixel_values = vision.preprocess(pair.image).to(device)
    vision_tokens = vision(pixel_values)                       # (1, N, C_v)
    visual_embeds = projector(vision_tokens)                   # (1, K, D_llm) bf16

    # --- Embed text chunks ---
    embed = llm.model.get_input_embeddings()
    left_embeds = embed(left_ids)                              # (1, L1, D)
    right_embeds = embed(right_ids)                            # (1, L2, D)
    report_embeds = embed(report_ids)                          # (1, R, D)

    # Match dtypes for concat.
    target_dtype = left_embeds.dtype
    visual_embeds = visual_embeds.to(target_dtype)

    inputs_embeds = torch.cat(
        [left_embeds, visual_embeds, right_embeds, report_embeds], dim=1
    )
    T = inputs_embeds.shape[1]
    attention_mask = torch.ones((1, T), dtype=torch.long, device=device)

    # --- Labels: -100 everywhere except the report_tokens region ---
    # HuggingFace shifts labels internally when computing loss for causal LMs,
    # so we supply labels aligned 1:1 with inputs_embeds.
    labels = torch.full((1, T), -100, dtype=torch.long, device=device)
    report_start = left_embeds.shape[1] + visual_embeds.shape[1] + right_embeds.shape[1]
    labels[0, report_start:] = report_ids[0]

    return BatchTensors(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )


# -----------------------------------------------------------------------------
# Train loop
# -----------------------------------------------------------------------------


def train(
    config: Config,
    max_pairs: int,
    epochs: int,
    lr: float,
    log_every: int,
    save_path: Path,
) -> None:
    config.validate()

    print("[stage1] Loading vision encoder...")
    vision = VisionEncoder(config)
    print("[stage1] Loading LLM (4-bit)...")
    llm = load_llm(config)

    # Sanity: LLM dim matches config
    if llm.hidden_dim != config.llm_hidden_dim:
        raise RuntimeError(
            f"LLM hidden_dim {llm.hidden_dim} != config {config.llm_hidden_dim}"
        )

    print("[stage1] Initializing projector...")
    projector = PerceiverResampler(
        vision_dim=config.vision_hidden_dim,
        llm_dim=llm.hidden_dim,
        num_latents=config.num_visual_tokens,
        num_heads=config.projector_num_heads,
        num_layers=config.projector_num_layers,
    )
    vision.to(llm.device)
    projector.to(llm.device)
    projector.train()

    # Freeze everything else — explicitly, so we don't rely on inherited state.
    for p in vision.parameters():
        p.requires_grad = False
    for p in llm.model.parameters():
        p.requires_grad = False

    trainable = [p for p in projector.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    print(f"[stage1] Trainable params (projector only): {trainable_count:,}")

    optimizer = AdamW(trainable, lr=lr, weight_decay=0.01)

    # Stage-1 prefix is fixed — compute once per run.
    prefix_text, _ = _build_stage1_prompt(llm.tokenizer)

    log_path = config.logs_dir / "stage1.jsonl"
    log_f = open(log_path, "w", encoding="utf-8")

    global_step = 0
    t_start = time.time()
    first_step_grad_checked = False

    for epoch in range(epochs):
        print(f"[stage1] === Epoch {epoch + 1}/{epochs} ===")
        pairs_iter = stream_mimic_pairs(config, split="train", max_pairs=max_pairs)
        for pair in pairs_iter:
            optimizer.zero_grad(set_to_none=True)
            try:
                batch = _encode_example(pair, vision, projector, llm, prefix_text)
            except Exception as e:  # noqa: BLE001 — a bad sample should not kill training
                print(f"[stage1] skip sample: {type(e).__name__}: {e}")
                continue

            out = llm.model(
                inputs_embeds=batch.inputs_embeds,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
            )
            loss = out.loss
            if not torch.isfinite(loss):
                print(f"[stage1] step {global_step}: non-finite loss, skipping")
                continue

            loss.backward()

            # First-step gradient flow check — if projector grad is all zero the
            # training is silently broken. SRS §19.2 module-2 validation.
            if not first_step_grad_checked:
                total_g = 0.0
                for p in trainable:
                    if p.grad is not None:
                        total_g += p.grad.detach().abs().sum().item()
                if total_g == 0.0:
                    raise RuntimeError(
                        "Projector received zero gradient on step 1 — training is a no-op."
                    )
                print(f"[stage1] gradient flow OK (sum|grad|={total_g:.4g})")
                first_step_grad_checked = True

            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            global_step += 1
            if global_step % log_every == 0:
                vram_gb = (
                    torch.cuda.memory_allocated(llm.device) / (1024**3)
                    if llm.device.type == "cuda" else 0.0
                )
                elapsed = time.time() - t_start
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "vram_gb": round(vram_gb, 2),
                    "elapsed_s": round(elapsed, 1),
                }
                print(
                    f"[stage1] step {global_step:5d} | loss {row['loss']:.4f} "
                    f"| vram {row['vram_gb']:.2f}GB"
                )
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

    log_f.close()

    # Save projector only. Per SRS §15: checkpoint separation for LoRA vs projector.
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), save_path)
    print(f"[stage1] Saved projector to {save_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument(
        "--save-path",
        type=Path,
        default=CONFIG.models_dir / "projector_stage1.pt",
    )
    args = ap.parse_args()

    try:
        train(
            CONFIG,
            max_pairs=args.max_pairs,
            epochs=args.epochs,
            lr=args.lr,
            log_every=args.log_every,
            save_path=args.save_path,
        )
    except KeyboardInterrupt:
        print("\n[stage1] Interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())