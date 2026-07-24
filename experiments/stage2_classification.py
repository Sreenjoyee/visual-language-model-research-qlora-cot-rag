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
import itertools
import json
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from src.classification_head import ClassificationHead
from src.config import CONFIG, Config
from src.data.balanced_stream import LabeledPair, balanced_mimic_stream, check_label_distribution
from src.data.iu_xray_stream import iu_xray_abnormal_training_stream, iu_xray_normal_training_stream
from src.data.pneumonia_xray_stream import pneumonia_xray_stream
from src.llm import LoadedLLM, load_llm
from src.projector import PerceiverResampler
from src.prompts import (
    IMAGE_PLACEHOLDER,
    build_chat_messages,
    build_classification_target,
    maybe_inject_adversarial,
)
from src.retrieval import Retriever
from src.vision import VisionEncoder


# ── LoRA config (SRS §7) ───────────────────────────────────────────────────

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


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
    perceiver_out: torch.Tensor   # (1, K, D_llm) — for ClassificationHead
    rag_embeddings: torch.Tensor  # (1, k, rag_dim) — for ClassificationHead
    label_id: int                 # 0 = NORMAL, 1 = ABNORMAL


def _encode_example(
    pair: LabeledPair,
    vision: VisionEncoder,
    projector: PerceiverResampler,
    llm: LoadedLLM,
    retriever: Retriever,
    config: Config,
    max_target_tokens: int = 500,
    step_idx: int = 0,
    adv_rng: "random.Random | None" = None,
) -> BatchTensors:
    """Encode one labeled pair into training tensors.

    Input sequence:  [left_prompt | visual_embeds | right_prompt]
    Target sequence: [classification_target_tokens]
    Labels:          [-100 ... -100 | target_ids]

    Retrieval uses the same generic query as inference (SRS §12 consistency).
    """
    device = llm.device
    tokenizer = llm.tokenizer

    # 1. FAISS retrieval — use first sentence of report as query to better match
    # the caption-style queries used at inference, reducing train/inference mismatch.
    first_sentence = pair.report.split(".")[0].strip()[:150]
    retrieved = retriever.query(first_sentence, k=config.retrieval_top_k)
    snippets = [r.text for r in retrieved]

    # Adversarial injection: with ADVERSARIAL_INJECTION_PROB, replace the last
    # snippet with a misleading one. The label stays correct — model must learn
    # to trust visual features over text when they conflict.
    if adv_rng is not None:
        snippets = maybe_inject_adversarial(snippets, adv_rng)

    # Stack RAG embeddings for ClassificationHead — fall back to zeros if any
    # snippet is missing its embedding (shouldn't happen with IndexFlatL2).
    rag_embs = [r.embedding for r in retrieved if r.embedding is not None]
    if rag_embs:
        rag_embeddings = torch.from_numpy(np.stack(rag_embs)).unsqueeze(0).float()  # (1, k, 384)
    else:
        rag_embeddings = torch.zeros(1, config.retrieval_top_k, config.embedder_dim)

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

    # 3. Classification target — training-only, never in inference prompt.
    # Pass report text so step 3 uses real clinical language instead of a template.
    target_text = build_classification_target(
        pair.label, idx=step_idx, report_snippet=pair.report
    )
    target_ids = tokenizer(
        target_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_target_tokens,
        return_tensors="pt",
    ).input_ids.to(device)
    eos = torch.tensor([[tokenizer.eos_token_id]], device=device, dtype=target_ids.dtype)
    target_ids = torch.cat([target_ids, eos], dim=1)

    # 4. Vision → projector. Optional train-time augmentation on the image; the
    #    encoder runs under no_grad unless fine-tuning is enabled (handled inside
    #    VisionEncoder.forward), so grads reach the encoder only when unfrozen.
    src_image = _augment_image(pair.image, adv_rng) if config.vision_augment else pair.image
    pixel_values = vision.preprocess(src_image).to(device)
    vision_tokens = vision(pixel_values)                 # (1, N, C_v)
    visual_embeds = projector(vision_tokens)             # (1, K, D_llm) bf16
    perceiver_out = visual_embeds                        # kept for ClassificationHead; no detach — cls_loss must reach input_norm

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

    if pair.label == "NORMAL":
        label_id = 0
    elif pair.label == "ABNORMAL":
        label_id = 1
    else:
        raise ValueError(f"Unexpected label {pair.label!r} — expected 'NORMAL' or 'ABNORMAL'")

    return BatchTensors(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        perceiver_out=perceiver_out,
        rag_embeddings=rag_embeddings.to(device),
        label_id=label_id,
    )


# ── Training loop ──────────────────────────────────────────────────────────

def _save_checkpoint_s2(
    ckpt_dir: Path,
    model,
    cls_head: ClassificationHead,
    cls_head_path: Path,
    optimizer: AdamW,
    scheduler,
    step: int,
    epoch: int,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # LoRA adapter weights (via peft)
    model.save_pretrained(str(ckpt_dir))
    # ClassificationHead — saved separately per SRS §15 checkpoint separation
    torch.save(cls_head.state_dict(), cls_head_path)
    # Also save a per-checkpoint copy so late checkpoints can be prediction-ensembled
    # post-hoc (the classification path is adapter-independent — only the head varies,
    # so averaging these heads' softmax needs no LLM reload). ~a few MB each.
    torch.save(cls_head.state_dict(), ckpt_dir / "cls_head.pt")
    # Optimizer + scheduler state alongside the adapter
    torch.save({
        "step": step, "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, ckpt_dir / "train_state.pt")
    print(f"[stage2] Checkpoint saved → {ckpt_dir}  (step {step})")


# Matches a mid-training checkpoint dir like "lora_step1500". Anchored with $ so it
# never matches the final "lora_adapter" / "lora_adapter_swa" output dirs.
_LORA_STEP_PAT = re.compile(r"lora_step(\d+)$")


def _rotate_checkpoints_s2(parent_dir: Path, keep_last: int) -> None:
    """Delete oldest lora_step* checkpoints, keeping only the newest `keep_last`.

    Stage-2 disk hygiene: mid-training checkpoints (~100 MB each) otherwise accumulate
    unbounded. Keeping the last N caps disk while preserving enough late checkpoints for
    SWA weight-averaging (experiments.average_checkpoints --last-n) and for resume
    (run_pipeline resumes from the highest-step dir, which is always kept). keep_last <= 0
    disables rotation. Only lora_step* dirs are matched, so the final lora_adapter /
    lora_adapter_swa dirs are never removed. Must be >= the SWA --last-n in use.
    """
    if keep_last <= 0:
        return
    ckpts = sorted(
        (int(_LORA_STEP_PAT.match(d.name).group(1)), d)
        for d in parent_dir.iterdir()
        if d.is_dir() and _LORA_STEP_PAT.match(d.name)
    )
    for _, d in ckpts[:-keep_last]:
        shutil.rmtree(d, ignore_errors=True)
        print(f"[stage2] Rotated out old checkpoint → {d.name}  (keep_last={keep_last})")


def classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: float = 1.0,
    focal_gamma: float = 0.0,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Recall-oriented classification loss (Run 4).

    With the defaults (pos_weight=1.0, focal_gamma=0.0) this is exactly the
    label-smoothed cross-entropy used through Run 3. Run 4 knobs:
      - pos_weight > 1 scales the loss on ABNORMAL (class id 1) samples so false
        negatives cost more -> higher recall/sensitivity. Applied as a per-sample
        weight because a mean-reduced weighted CE cancels the weight when the
        batch is a single sample (Stage 2 trains batch-size-1 with grad accum).
      - focal_gamma > 0 applies focal modulation (1 - p_true)^gamma, concentrating
        the gradient on hard / confidently-misclassified examples.
    """
    w = torch.where(
        target == 1,
        torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype),
        torch.as_tensor(1.0, device=logits.device, dtype=logits.dtype),
    )
    if focal_gamma and focal_gamma > 0.0:
        logp = F.log_softmax(logits, dim=-1)
        pt = logp.exp().gather(1, target.view(-1, 1)).squeeze(1).clamp_(1e-6, 1.0)
        ce_per = F.nll_loss(logp, target, reduction="none")
        return (w * ((1.0 - pt) ** focal_gamma) * ce_per).mean()
    ce_per = F.cross_entropy(logits, target, label_smoothing=label_smoothing, reduction="none")
    return (w * ce_per).mean()


def _augment_image(image, rng=None):
    """Light, CXR-appropriate train-time augmentation (Run-4 AUROC/OOD lever).

    Small rotation + brightness/contrast jitter — the variations a real scanner /
    positioning produces. Deliberately NO horizontal flip (that would swap
    anatomical laterality, e.g. put the heart on the right). Conservative ranges so
    diagnostic content is preserved.
    """
    import random as _random
    from PIL import Image as _Image, ImageEnhance
    r = rng if rng is not None else _random
    img = image.convert("RGB") if image.mode != "RGB" else image
    ang = r.uniform(-8.0, 8.0)
    if abs(ang) > 0.5:
        img = img.rotate(ang, resample=_Image.BILINEAR, fillcolor=(0, 0, 0))
    img = ImageEnhance.Brightness(img).enhance(r.uniform(0.9, 1.1))
    img = ImageEnhance.Contrast(img).enhance(r.uniform(0.9, 1.1))
    return img


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
    keep_last_checkpoints: int = 8,
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
        # strict=False: Stage-1 checkpoint predates input_norm; it initialises to
        # identity so pre-trained projector behaviour is preserved at step 0.
        projector.load_state_dict(state, strict=False)
        print(f"[stage2] Projector weights loaded from {projector_path}")
    else:
        print("[stage2] WARNING: no projector checkpoint found — using random weights.")

    vision.to(llm.device)
    projector.to(llm.device).to(torch.bfloat16)

    # Freeze vision entirely — never updated (default). Run 4 can unfreeze it below.
    for p in vision.parameters():
        p.requires_grad = False
    vision.eval()

    # Optional (AUROC lever): fine-tune the DenseNet-121 encoder so its CXR features
    # adapt further to the training set. Made trainable at a low LR.
    vision_params: list = []
    if config.vision_finetune:
        n_blocks = config.vision_finetune_blocks if config.vision_finetune_blocks > 0 else None
        vision_params = vision.enable_finetune(last_n_blocks=n_blocks)
        print(f"[stage2] Vision fine-tune ON: {sum(p.numel() for p in vision_params):,} params "
              f"@ lr={config.vision_lr} (last {n_blocks if n_blocks else 'all'} blocks)")
    if config.vision_augment:
        print("[stage2] Train-time CXR augmentation ON (rotation ±8°, brightness/contrast ±10%)")

    # Freeze projector but unfreeze input_norm so it can learn domain-invariant
    # feature scaling for OOD generalisation (LayerNorm starts at identity so
    # pre-trained projector behaviour is preserved at the start of Stage 2).
    for name, p in projector.named_parameters():
        p.requires_grad = name.startswith("input_norm")
    projector.eval()

    print("[stage2] Building ClassificationHead...")
    cls_head = ClassificationHead(
        llm_dim=llm.hidden_dim,
        rag_dim=config.embedder_dim,
        hidden_dim=config.cls_hidden_dim,
    )
    cls_head.to(llm.device)
    cls_head.train()

    print(f"[stage2] Applying LoRA ({', '.join(LORA_TARGET_MODULES)})...")
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

    # Optimizer covers LoRA params + ClassificationHead + projector input_norm.
    # input_norm uses a lower LR to avoid disrupting pre-trained projector geometry.
    lora_params = [p for p in llm.model.parameters() if p.requires_grad]
    input_norm_params = [p for p in projector.parameters() if p.requires_grad]
    print(f"[stage2] Trainable LoRA params:       {sum(p.numel() for p in lora_params):,}")
    print(f"[stage2] Trainable input_norm params:  {sum(p.numel() for p in input_norm_params):,}")
    print(f"[stage2] Trainable cls_head params:    {sum(p.numel() for p in cls_head.parameters()):,}")
    optimizer = AdamW(
        [
            {"params": lora_params,                        "lr": lr},
            {"params": list(cls_head.parameters()),        "lr": lr},
            {"params": input_norm_params,                  "lr": lr * 0.1},
        ]
        + ([{"params": vision_params, "lr": config.vision_lr}] if vision_params else []),
        weight_decay=0.01,
    )

    # Each MIMIC pair yields 5 samples when all supplemental sources are loaded:
    # 1 MIMIC + 1 IU-Normal + 1 IU-Abnormal + 1 Kermany-Normal + 1 Kermany-Abnormal
    _interleave_factor = 5
    steps_per_epoch = max(1, max_pairs * _interleave_factor // max(grad_accum_steps, 1))
    total_steps = epochs * steps_per_epoch
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
        # Restore ClassificationHead weights if they exist alongside the LoRA checkpoint
        if config.cls_head_path.exists():
            cls_head.load_state_dict(torch.load(config.cls_head_path, map_location="cpu"))
            print(f"[stage2] ClassificationHead weights restored from {config.cls_head_path}")

    log_path = config.logs_dir / "stage2.jsonl"
    log_mode = "a" if resume_from is not None else "w"
    log_f = open(log_path, log_mode, encoding="utf-8")

    t_start = time.time()
    resume_step = global_step  # steps already done before this session
    first_step_grad_checked = False
    accum_loss = 0.0
    accum_cls_loss = 0.0
    accum_lm_loss = 0.0
    micro_step = 0
    adv_rng = random.Random(42)  # seeded for reproducibility

    # Pre-load supplemental IU-Xray + Kermany samples once — reused across epochs.
    # Wrapped so a dataset failure (e.g. an HF Xet 403 on those parquets) degrades
    # gracefully to MIMIC-only training instead of crashing the run. The interleave
    # below already treats an empty supplemental list as "skip this source".
    def _safe_preload(stream_fn, label):
        try:
            out = list(stream_fn())
            print(f"[stage2] {label} loaded: {len(out)}")
            return out
        except Exception as e:
            print(f"[stage2] {label} UNAVAILABLE ({type(e).__name__}: {e}) — continuing without it")
            return []

    print("[stage2] Pre-loading supplemental sources (IU-Xray + Kermany)...")
    iu_xray_normals: list[LabeledPair]     = _safe_preload(lambda: iu_xray_normal_training_stream(max_samples=2000), "IU-Xray NORMAL")
    iu_xray_abnormals: list[LabeledPair]   = _safe_preload(lambda: iu_xray_abnormal_training_stream(max_samples=500), "IU-Xray ABNORMAL")
    pneumonia_normals: list[LabeledPair]   = _safe_preload(lambda: pneumonia_xray_stream(label="NORMAL", max_samples=400), "Kermany NORMAL")
    pneumonia_abnormals: list[LabeledPair] = _safe_preload(lambda: pneumonia_xray_stream(label="ABNORMAL", max_samples=400), "Kermany ABNORMAL")

    # Step-budget loop (resume-safe): train until global_step reaches total_steps,
    # regardless of how many data passes that takes. On resume we continue from the
    # saved global_step and stop exactly at total_steps — no redoing a full epoch and
    # no overshooting the LR schedule. Each outer iteration is one fresh pass over the
    # interleaved stream; data cycles as needed until the step budget is met.
    while global_step < total_steps:
        epoch = min(global_step // steps_per_epoch, epochs - 1)   # 0-indexed, for display
        print(f"[stage2] === Data pass | epoch {epoch + 1}/{epochs} "
              f"| step {global_step}/{total_steps} ===")
        optimizer.zero_grad(set_to_none=True)

        # Interleave MIMIC stream with IU-Xray NORMAL and ABNORMAL samples.
        # Pattern: MIMIC, IU-Normal, MIMIC, IU-Abnormal — keeps 1:1 overall balance
        # while adding institutional diversity to both classes.
        mimic_stream = balanced_mimic_stream(config, split="train", max_pairs=max_pairs)
        iu_normal_cycle   = itertools.cycle(iu_xray_normals)    if iu_xray_normals    else None
        iu_abnormal_cycle = itertools.cycle(iu_xray_abnormals)  if iu_xray_abnormals  else None
        pn_normal_cycle   = itertools.cycle(pneumonia_normals)   if pneumonia_normals   else None
        pn_abnormal_cycle = itertools.cycle(pneumonia_abnormals) if pneumonia_abnormals else None

        def _interleaved(mimic, iu_n, iu_a, pn_n, pn_a):
            for mimic_pair in mimic:
                yield mimic_pair
                if iu_n is not None:
                    yield next(iu_n)
                if iu_a is not None:
                    yield next(iu_a)
                if pn_n is not None:
                    yield next(pn_n)
                if pn_a is not None:
                    yield next(pn_a)

        stream = _interleaved(
            mimic_stream,
            iu_normal_cycle, iu_abnormal_cycle,
            pn_normal_cycle, pn_abnormal_cycle,
        )
        for pair in stream:
            if global_step >= total_steps:
                break
            try:
                batch = _encode_example(pair, vision, projector, llm, retriever, config, step_idx=global_step, adv_rng=adv_rng)
            except Exception as e:
                print(f"[stage2] skip sample ({pair.label}): {type(e).__name__}: {e}")
                continue

            try:
                out = llm.model(
                    inputs_embeds=batch.inputs_embeds,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[stage2] OOM on forward ({pair.label}) — skipping sample")
                    del batch
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    micro_step = 0
                    accum_loss = accum_cls_loss = accum_lm_loss = 0.0
                    continue
                raise
            lm_loss = out.loss
            if lm_loss is None or not torch.isfinite(lm_loss):
                print(f"[stage2] step {global_step}: non-finite lm_loss, skipping")
                continue

            # Classification loss. The interleaved stream is balanced 1:1, so by
            # default this is plain label-smoothed CE (label smoothing curbs
            # overconfidence and improves OOD calibration). Run 4 can up-weight the
            # ABNORMAL class (cls_pos_weight) and/or enable focal loss
            # (cls_focal_gamma) to raise recall on abnormals — see classification_loss.
            label_tensor = torch.tensor([batch.label_id], device=llm.device)
            cls_logits = cls_head(batch.perceiver_out, batch.rag_embeddings)  # (1, 2)
            cls_loss = classification_loss(
                cls_logits,
                label_tensor,
                pos_weight=config.cls_pos_weight,
                focal_gamma=config.cls_focal_gamma,
                label_smoothing=0.1,
            )

            if not torch.isfinite(cls_loss):
                print(f"[stage2] step {global_step}: non-finite cls_loss, skipping")
                continue

            # Multi-task loss: weighted sum of generation and classification objectives
            alpha = config.cls_alpha
            loss = alpha * cls_loss + (1.0 - alpha) * lm_loss

            scaled_loss = loss / grad_accum_steps
            try:
                scaled_loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[stage2] OOM on backward — skipping sample and clearing cache")
                    try:
                        # Delete tensors explicitly before clearing cache
                        del scaled_loss, loss, cls_loss, lm_loss, out
                        del cls_logits, label_tensor, batch
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        optimizer.zero_grad(set_to_none=True)
                    except Exception:
                        pass
                    micro_step = 0
                    accum_loss = 0.0
                    accum_cls_loss = 0.0
                    accum_lm_loss = 0.0
                    continue
                raise

            # Periodic cache clearing every 100 micro-steps to prevent fragmentation
            if micro_step % 100 == 0:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            accum_loss += loss.item()
            accum_cls_loss += cls_loss.item()
            accum_lm_loss += lm_loss.item()
            micro_step += 1

            if micro_step % grad_accum_steps != 0:
                continue   # accumulate more before updating

            # ── Parameter update ──────────────────────────────────────────────
            all_trainable = lora_params + list(cls_head.parameters()) + input_norm_params
            if not first_step_grad_checked:
                total_g = sum(
                    p.grad.detach().abs().sum().item()
                    for p in all_trainable if p.grad is not None
                )
                if total_g == 0.0:
                    raise RuntimeError(
                        "Zero gradient on step 1 — check LoRA target_modules and cls_head."
                    )
                print(f"[stage2] Gradient flow OK (sum|grad|={total_g:.4g})")
                first_step_grad_checked = True

            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            avg_loss = accum_loss / grad_accum_steps
            avg_cls_loss = accum_cls_loss / grad_accum_steps
            avg_lm_loss = accum_lm_loss / grad_accum_steps
            accum_loss = 0.0
            accum_cls_loss = 0.0
            accum_lm_loss = 0.0
            global_step += 1
            epoch = min(global_step // steps_per_epoch, epochs - 1)   # keep label in sync

            if global_step % log_every == 0:
                vram_gb = (
                    torch.cuda.memory_allocated(llm.device) / (1024 ** 3)
                    if llm.device.type == "cuda" else 0.0
                )
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_start
                steps_this_session = global_step - resume_step
                secs_per_step = elapsed / max(steps_this_session, 1)
                steps_left = total_steps - global_step
                eta_s = secs_per_step * steps_left
                eta_h = eta_s / 3600

                row = {
                    "step": global_step, "epoch": epoch,
                    "loss": round(avg_loss, 4),
                    "cls_loss": round(avg_cls_loss, 4),
                    "lm_loss": round(avg_lm_loss, 4),
                    "lr": round(current_lr, 8),
                    "label": pair.label,
                    "vram_gb": round(vram_gb, 2),
                    "elapsed_s": round(elapsed, 1),
                    "secs_per_step": round(secs_per_step, 1),
                    "eta_h": round(eta_h, 2),
                }
                print(
                    f"[stage2] step {global_step:5d}/{total_steps} "
                    f"| epoch {epoch + 1}/{epochs} "
                    f"| loss {avg_loss:.4f} (cls={avg_cls_loss:.4f} lm={avg_lm_loss:.4f}) "
                    f"| lr {current_lr:.2e} "
                    f"| {pair.label:<8} "
                    f"| vram {vram_gb:.2f}GB "
                    f"| {secs_per_step:.0f}s/step "
                    f"| ETA {eta_h:.1f}h"
                )
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

            # ── Mid-training checkpoint ───────────────────────────────────────
            if save_every > 0 and global_step % save_every == 0:
                ckpt_dir = lora_save_dir.parent / f"lora_step{global_step}"
                _save_checkpoint_s2(
                    ckpt_dir, llm.model, cls_head, config.cls_head_path,
                    optimizer, scheduler, global_step, epoch,
                )
                _rotate_checkpoints_s2(lora_save_dir.parent, keep_last_checkpoints)

    # Write final log row even if global_step never hit log_every (short runs)
    if global_step > 0:
        row = {
            "step": global_step, "epoch": epochs - 1,
            "loss": round(accum_loss / (micro_step % grad_accum_steps or grad_accum_steps), 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "vram_gb": round(torch.cuda.memory_allocated(llm.device) / (1024 ** 3), 2) if llm.device.type == "cuda" else 0.0,
            "elapsed_s": round(time.time() - t_start, 1),
        }
        log_f.write(json.dumps(row) + "\n")
    log_f.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Final save — LoRA adapter + ClassificationHead (SRS §15: checkpoint separation)
    lora_save_dir.mkdir(parents=True, exist_ok=True)
    llm.model.save_pretrained(str(lora_save_dir))
    torch.save(cls_head.state_dict(), config.cls_head_path)
    print(f"[stage2] Final LoRA adapter saved → {lora_save_dir}")
    print(f"[stage2] ClassificationHead saved → {config.cls_head_path}")
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
    ap.add_argument("--keep-last-checkpoints", type=int, default=8,
                    help="Keep only the newest N lora_step* checkpoints on disk (0 = keep all). "
                         "Must be >= the SWA --last-n so weight-averaging still has enough.")
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
            keep_last_checkpoints=args.keep_last_checkpoints,
        )
    except KeyboardInterrupt:
        print("\n[stage2] Interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
