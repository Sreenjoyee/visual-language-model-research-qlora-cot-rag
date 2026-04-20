"""Stage 2 — LoRA classification training (NORMAL / ABNORMAL).

SRS §7:
  - LoRA on q_proj, v_proj only
  - Classify: NORMAL / ABNORMAL
  - Strict prompt consistency with inference (SRS §12)
  - Balanced dataset mandatory (SRS §4, §13)

SRS §15: checkpoint separation.
  - Projector: loaded from Stage-1, FROZEN, not updated here.
  - LoRA adapter: saved separately to models/lora_stage2/.

Training flow per sample:
  1. Stream balanced pair (image, report, label) from balanced_mimic_stream
  2. Retrieve FAISS evidence (same generic query as inference — SRS §12)
  3. Build input prompt via build_chat_messages (byte-identical to inference)
  4. Build target via build_classification_target (training-only template)
  5. Forward: inputs_embeds (prompt) + target_ids; labels=-100 for prompt tokens
  6. LoRA + projector gradients only (LLM base weights frozen via 4-bit quant)

Failure modes prevented (SRS §9):
  - Label leakage: prompt built from build_chat_messages — no label parameter
  - CPU offload: load_llm enforces max_memory
  - Padding loss: labels=-100 for all non-target positions
  - Prompt drift: uses same build_chat_messages as pipeline.py
  - LoRA OOM: gradient checkpointing enabled + max_new_tokens not needed for training

Usage:
    python -m experiments.stage2_classification \\
        --projector-path models/projector_stage1.pt \\
        --max-pairs 4000 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from src.config import CONFIG, Config
from src.data.balanced_stream import LabeledPair, balanced_mimic_stream, check_label_distribution
from src.llm import LoadedLLM, load_llm
from src.projector import PerceiverResampler
from src.prompts import IMAGE_PLACEHOLDER, build_chat_messages, build_classification_target
from src.retrieval import Retriever
from src.vision import VisionEncoder


# ── LoRA config (SRS §7) ───────────────────────────────────────────────────

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]   # SRS §7 — exactly these two


def _apply_lora(model, config: Config):
    """Apply LoRA adapter to the LLM. Returns the peft-wrapped model."""
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError:
        raise RuntimeError(
            "peft is required for Stage 2. Install with: pip install peft"
        )

    # Required for 4-bit quantized + LoRA: enables gradients through frozen quant layers
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── Batch encoding ─────────────────────────────────────────────────────────

@dataclass
class BatchTensors:
    inputs_embeds: torch.Tensor   # (1, T_prompt + T_target, D)
    attention_mask: torch.Tensor  # (1, T_prompt + T_target)
    labels: torch.Tensor          # (1, T_prompt + T_target); -100 for prompt tokens


def _encode_example(
    pair: LabeledPair,
    vision: VisionEncoder,
    projector: PerceiverResampler,
    llm: LoadedLLM,
    retriever: Retriever,
    config: Config,
    max_target_tokens: int = 128,
) -> BatchTensors:
    """Encode one labeled pair into training tensors.

    Input sequence:  [left_prompt | visual_embeds | right_prompt]
    Target sequence: [classification_target_tokens]
    Labels:          [-100 ... -100 | target_ids]

    Retrieval uses the same generic query as inference (SRS §12 consistency).
    """
    device = llm.device
    tokenizer = llm.tokenizer

    # 1. FAISS retrieval — same query as inference (not report text, to match SRS §12)
    retrieved = retriever.query("Chest X-ray findings and clinical impression.", k=config.retrieval_top_k)
    snippets = [r.text for r in retrieved]

    # 2. Build prompt — identical to inference (SRS §12)
    messages = build_chat_messages(snippets)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if IMAGE_PLACEHOLDER not in prompt_text:
        raise RuntimeError("IMAGE_PLACEHOLDER missing from prompt — prompt code changed.")

    left_text, right_text = prompt_text.split(IMAGE_PLACEHOLDER, 1)

    left_ids = tokenizer(left_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    right_ids = tokenizer(right_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # 3. Classification target — training-only, never in inference prompt
    target_text = build_classification_target(pair.label)
    target_ids = tokenizer(
        target_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_target_tokens,
        return_tensors="pt",
    ).input_ids.to(device)
    eos = torch.tensor([[tokenizer.eos_token_id]], device=device, dtype=target_ids.dtype)
    target_ids = torch.cat([target_ids, eos], dim=1)

    # 4. Vision → projector
    with torch.no_grad():
        pixel_values = vision.preprocess(pair.image).to(device)
        vision_tokens = vision(pixel_values)             # (1, N, C_v)
    visual_embeds = projector(vision_tokens)             # (1, K, D_llm) bf16

    # 5. Embed text chunks
    embed = llm.model.get_input_embeddings()
    left_emb = embed(left_ids)
    right_emb = embed(right_ids)
    target_emb = embed(target_ids)

    dtype = left_emb.dtype
    visual_embeds = visual_embeds.to(dtype)

    # Full sequence: prompt (no label) + target
    prompt_emb = torch.cat([left_emb, visual_embeds, right_emb], dim=1)
    inputs_embeds = torch.cat([prompt_emb, target_emb], dim=1)

    T = inputs_embeds.shape[1]
    T_prompt = prompt_emb.shape[1]

    attention_mask = torch.ones((1, T), dtype=torch.long, device=device)

    # 6. Labels: -100 for prompt, real ids for target (SRS §9 no padding loss)
    labels = torch.full((1, T), -100, dtype=torch.long, device=device)
    labels[0, T_prompt:] = target_ids[0]

    return BatchTensors(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )


# ── Training loop ──────────────────────────────────────────────────────────

def _save_checkpoint_s2(
    ckpt_dir: Path,
    model,
    optimizer: AdamW,
    scheduler,
    step: int,
    epoch: int,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # LoRA adapter weights (via peft)
    model.save_pretrained(str(ckpt_dir))
    # Optimizer + scheduler state alongside the adapter
    torch.save({
        "step": step, "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, ckpt_dir / "train_state.pt")
    print(f"[stage2] Checkpoint saved → {ckpt_dir}  (step {step})")


def train(
    config: Config,
    projector_path: Path | None,
    max_pairs: int,
    epochs: int,
    lr: float,
    log_every: int,
    lora_save_dir: Path,
    balance_check_samples: int,
    warmup_steps: int = 100,
    save_every: int = 500,
    resume_from: Path | None = None,
    grad_accum_steps: int = 1,
) -> None:
    config.validate()

    print("[stage2] Loading vision encoder...")
    vision = VisionEncoder(config)

    print("[stage2] Loading LLM (4-bit NF4)...")
    llm = load_llm(config)

    if llm.hidden_dim != config.llm_hidden_dim:
        raise RuntimeError(
            f"LLM hidden_dim {llm.hidden_dim} != config {config.llm_hidden_dim}"
        )

    print("[stage2] Loading projector (frozen from Stage 1)...")
    projector = PerceiverResampler(
        vision_dim=config.vision_hidden_dim,
        llm_dim=llm.hidden_dim,
        num_latents=config.num_visual_tokens,
        num_heads=config.projector_num_heads,
        num_layers=config.projector_num_layers,
    )
    if projector_path is not None and projector_path.exists():
        state = torch.load(projector_path, map_location="cpu")
        projector.load_state_dict(state)
        print(f"[stage2] Projector weights loaded from {projector_path}")
    else:
        print("[stage2] WARNING: no projector checkpoint found — using random weights.")

    vision.to(llm.device)
    projector.to(llm.device)

    # Freeze vision + projector — only LoRA params train in Stage 2
    for p in vision.parameters():
        p.requires_grad = False
    for p in projector.parameters():
        p.requires_grad = False
    projector.eval()
    vision.eval()

    print("[stage2] Applying LoRA (q_proj, v_proj)...")
    llm.model = _apply_lora(llm.model, config)
    llm.model.train()

    print("[stage2] Loading FAISS retriever...")
    retriever = Retriever(config)
    retriever.load()

    # Collect balance-check sample before committing to a full training run
    print(f"[stage2] Pre-training balance check ({balance_check_samples} samples)...")
    check_samples: list[LabeledPair] = []
    for pair in balanced_mimic_stream(config, max_pairs=balance_check_samples):
        check_samples.append(pair)
    check_label_distribution(check_samples)
    print("[stage2] Label distribution OK.")

    # Optimizer + cosine scheduler over LoRA params only
    lora_params = [p for p in llm.model.parameters() if p.requires_grad]
    print(f"[stage2] Trainable LoRA params: {sum(p.numel() for p in lora_params):,}")
    optimizer = AdamW(lora_params, lr=lr, weight_decay=0.01)

    total_steps = epochs * max_pairs // max(grad_accum_steps, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    if resume_from is not None and resume_from.exists():
        train_state = resume_from / "train_state.pt"
        if train_state.exists():
            ckpt = torch.load(train_state, map_location="cpu")
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            global_step  = ckpt["step"]
            start_epoch  = ckpt["epoch"]
            print(f"[stage2] Resumed from {resume_from}  (step={global_step}, epoch={start_epoch})")
        else:
            print(f"[stage2] WARNING: resume_from given but no train_state.pt found at {resume_from}")

    log_path = config.logs_dir / "stage2.jsonl"
    log_mode = "a" if resume_from is not None else "w"
    log_f = open(log_path, log_mode, encoding="utf-8")

    t_start = time.time()
    first_step_grad_checked = False
    accum_loss = 0.0
    micro_step = 0

    for epoch in range(start_epoch, epochs):
        print(f"[stage2] === Epoch {epoch + 1}/{epochs} ===")
        optimizer.zero_grad(set_to_none=True)

        stream = balanced_mimic_stream(config, split="train", max_pairs=max_pairs)
        for pair in stream:
            try:
                batch = _encode_example(pair, vision, projector, llm, retriever, config)
            except Exception as e:
                print(f"[stage2] skip sample ({pair.label}): {type(e).__name__}: {e}")
                continue

            out = llm.model(
                inputs_embeds=batch.inputs_embeds,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
            )
            loss = out.loss
            if loss is None or not torch.isfinite(loss):
                print(f"[stage2] step {global_step}: non-finite loss, skipping")
                continue

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            accum_loss += loss.item()
            micro_step += 1

            if micro_step % grad_accum_steps != 0:
                continue   # accumulate more before updating

            # ── Parameter update ──────────────────────────────────────────────
            if not first_step_grad_checked:
                total_g = sum(
                    p.grad.detach().abs().sum().item()
                    for p in lora_params if p.grad is not None
                )
                if total_g == 0.0:
                    raise RuntimeError(
                        "LoRA received zero gradient on step 1 — check target_modules."
                    )
                print(f"[stage2] LoRA gradient flow OK (sum|grad|={total_g:.4g})")
                first_step_grad_checked = True

            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            avg_loss = accum_loss / grad_accum_steps
            accum_loss = 0.0
            global_step += 1

            if global_step % log_every == 0:
                vram_gb = (
                    torch.cuda.memory_allocated(llm.device) / (1024 ** 3)
                    if llm.device.type == "cuda" else 0.0
                )
                current_lr = scheduler.get_last_lr()[0]
                row = {
                    "step": global_step, "epoch": epoch,
                    "loss": round(avg_loss, 4),
                    "lr": round(current_lr, 8),
                    "label": pair.label,
                    "vram_gb": round(vram_gb, 2),
                    "elapsed_s": round(time.time() - t_start, 1),
                }
                print(
                    f"[stage2] step {global_step:5d} | loss {row['loss']:.4f} "
                    f"| lr {current_lr:.2e} | label {pair.label:<8} | vram {row['vram_gb']:.2f}GB"
                )
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

            # ── Mid-training checkpoint ───────────────────────────────────────
            if save_every > 0 and global_step % save_every == 0:
                ckpt_dir = lora_save_dir.parent / f"lora_step{global_step}"
                _save_checkpoint_s2(ckpt_dir, llm.model, optimizer, scheduler, global_step, epoch)

    # Write final log row even if global_step never hit log_every (short runs)
    if global_step > 0:
        row = {
            "step": global_step, "epoch": epochs - 1,
            "loss": round(accum_loss / max(micro_step % grad_accum_steps or grad_accum_steps, 1), 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "vram_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0,
            "elapsed_s": round(time.time() - t_start, 1),
        }
        log_f.write(json.dumps(row) + "\n")
    log_f.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Final save — LoRA adapter only (SRS §15: checkpoint separation)
    lora_save_dir.mkdir(parents=True, exist_ok=True)
    llm.model.save_pretrained(str(lora_save_dir))
    print(f"[stage2] Final LoRA adapter saved → {lora_save_dir}")
    print(f"[stage2] Projector NOT re-saved here — load from {projector_path} for inference.")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2: LoRA binary classification")
    ap.add_argument(
        "--projector-path",
        type=Path,
        default=CONFIG.models_dir / "projector_stage1.pt",
        help="Path to Stage-1 projector checkpoint.",
    )
    ap.add_argument("--max-pairs", type=int, default=4000,
                    help="Balanced pairs per epoch (NORMAL:ABNORMAL = 1:1).")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--warmup-steps", type=int, default=100,
                    help="Linear warmup steps before cosine decay.")
    ap.add_argument("--save-every", type=int, default=500,
                    help="Save mid-training checkpoint every N optimizer steps (0 = off).")
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="LoRA checkpoint directory to resume from (must contain train_state.pt).")
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="Gradient accumulation steps (effective batch size multiplier).")
    ap.add_argument("--lora-save-dir", type=Path,
                    default=CONFIG.models_dir / "lora_stage2")
    ap.add_argument("--balance-check-samples", type=int, default=100)
    args = ap.parse_args()

    try:
        train(
            config=CONFIG,
            projector_path=args.projector_path,
            max_pairs=args.max_pairs,
            epochs=args.epochs,
            lr=args.lr,
            log_every=args.log_every,
            lora_save_dir=args.lora_save_dir,
            balance_check_samples=args.balance_check_samples,
            warmup_steps=args.warmup_steps,
            save_every=args.save_every,
            resume_from=args.resume_from,
            grad_accum_steps=args.grad_accum_steps,
        )
    except KeyboardInterrupt:
        print("\n[stage2] Interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
