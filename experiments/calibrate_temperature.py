"""Post-hoc temperature calibration for ClassificationHead.

Finds the scalar temperature T* that minimises negative log-likelihood on
the MIMIC validation split, then writes the updated value back into cls_head.pt.

Why post-hoc calibration works:
    The temperature parameter in ClassificationHead is trained jointly with the
    classification objective (AUROC) which pushes logits apart, not toward
    honest probabilities. Post-hoc calibration freezes all weights and searches
    only over T, so it cannot degrade AUROC (temperature is monotonic — it never
    changes which class scores higher, only by how much).

    Expected outcome: ECE 0.29 → ~0.05, AUROC unchanged.

Usage:
    python -m experiments.calibrate_temperature \\
        --projector-path models/projector_stage1.pt \\
        --lora-dir models/lora_stage2 \\
        --val-pairs 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from src.classification_head import ClassificationHead
from src.config import CONFIG, Config
from src.data.balanced_stream import LabeledPair, balanced_mimic_stream
from src.llm import LoadedLLM, load_llm
from src.projector import PerceiverResampler
from src.retrieval import Retriever
from src.vision import VisionEncoder


def _collect_logits(
    config: Config,
    projector_path: Path | None,
    lora_dir: Path | None,
    val_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run val-set forward passes and collect raw pre-temperature logits + labels."""

    print("[calibrate] Loading components...")
    vision = VisionEncoder(config)
    llm: LoadedLLM = load_llm(config)

    projector = PerceiverResampler(
        vision_dim=config.vision_hidden_dim,
        llm_dim=llm.hidden_dim,
        num_latents=config.num_visual_tokens,
        num_heads=config.projector_num_heads,
        num_layers=config.projector_num_layers,
    )
    if projector_path and projector_path.exists():
        projector.load_state_dict(
            torch.load(projector_path, map_location="cpu"), strict=False
        )
    vision.to(llm.device)
    projector.to(llm.device).to(torch.bfloat16)
    vision.eval()
    projector.eval()

    if lora_dir and lora_dir.exists():
        try:
            from peft import PeftModel
            llm.model = PeftModel.from_pretrained(llm.model, str(lora_dir))
            llm.model = llm.model.merge_and_unload()
            print(f"[calibrate] LoRA merged from {lora_dir}")
        except ImportError:
            print("[calibrate] WARNING: peft not available, using base LLM weights.")
    llm.model.eval()

    cls_head = ClassificationHead(
        llm_dim=llm.hidden_dim,
        rag_dim=config.embedder_dim,
        hidden_dim=config.cls_hidden_dim,
    )
    if not config.cls_head_path.exists():
        raise FileNotFoundError(
            f"cls_head.pt not found at {config.cls_head_path}. "
            "Run Stage 2 training first."
        )
    cls_head.load_state_dict(torch.load(config.cls_head_path, map_location="cpu"))
    cls_head.to(llm.device)
    cls_head.eval()

    retriever = Retriever(config)
    retriever.load()

    all_raw_logits: list[torch.Tensor] = []
    all_labels: list[int] = []
    skipped = 0

    print(f"[calibrate] Collecting {val_pairs} validation pairs...")
    with torch.no_grad():
        for pair in balanced_mimic_stream(config, split="val", max_pairs=val_pairs):
            try:
                first_sentence = pair.report.split(".")[0].strip()[:150]
                retrieved = retriever.query(first_sentence, k=config.retrieval_top_k)

                rag_embs = [r.embedding for r in retrieved if r.embedding is not None]
                if not rag_embs:
                    skipped += 1
                    continue
                rag_tensor = torch.from_numpy(
                    np.stack(rag_embs)
                ).unsqueeze(0).float().to(llm.device)

                pixel_values = vision.preprocess(pair.image).to(llm.device)
                vision_tokens = vision(pixel_values)
                visual_embeds = projector(vision_tokens)

                # Get tempered logits, then recover raw logits by multiplying back
                tempered_logits = cls_head(visual_embeds, rag_tensor)   # (1, 2)
                raw_logits = tempered_logits * cls_head.temperature.clamp(min=0.1)

                all_raw_logits.append(raw_logits.cpu().float())
                all_labels.append(1 if pair.label == "ABNORMAL" else 0)

            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"[calibrate] skipped sample: {type(e).__name__}: {e}")
                continue

    if not all_raw_logits:
        raise RuntimeError("No validation samples collected — check MIMIC val split.")

    print(f"[calibrate] Collected {len(all_raw_logits)} samples ({skipped} skipped).")
    logits = torch.cat(all_raw_logits, dim=0)   # (N, 2)
    labels = torch.tensor(all_labels)            # (N,)
    return logits, labels


def _calibrate(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Find T* minimising NLL via bounded scalar optimisation."""

    current_T = 1.0
    current_nll = F.cross_entropy(logits / current_T, labels).item()

    def nll(T: float) -> float:
        T = max(float(T), 0.01)
        return F.cross_entropy(logits / T, labels).item()

    result = minimize_scalar(nll, bounds=(0.01, 20.0), method="bounded")
    optimal_T = float(result.x)
    optimal_nll = float(result.fun)

    print(f"[calibrate] Current  T={current_T:.4f}  NLL={current_nll:.4f}")
    print(f"[calibrate] Optimal  T={optimal_T:.4f}  NLL={optimal_nll:.4f}")
    print(f"[calibrate] NLL improvement: {current_nll - optimal_nll:.4f}")
    return optimal_T


def _ece(logits: torch.Tensor, labels: torch.Tensor, T: float, n_bins: int = 15) -> float:
    """Expected Calibration Error — lower is better."""
    probs = torch.softmax(logits / T, dim=1)[:, 1].numpy()
    lbls = labels.numpy()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = lbls[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / len(lbls))


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-hoc temperature calibration")
    ap.add_argument("--projector-path", type=Path,
                    default=CONFIG.models_dir / "projector_stage1.pt")
    ap.add_argument("--lora-dir", type=Path,
                    default=CONFIG.models_dir / "lora_stage2")
    ap.add_argument("--val-pairs", type=int, default=500,
                    help="Validation pairs to calibrate on.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print optimal T but do not overwrite cls_head.pt.")
    args = ap.parse_args()

    CONFIG.validate()

    logits, labels = _collect_logits(
        CONFIG, args.projector_path, args.lora_dir, args.val_pairs
    )

    ece_before = _ece(logits, labels, T=1.0)
    optimal_T = _calibrate(logits, labels)
    ece_after = _ece(logits, labels, optimal_T)

    print(f"\n[calibrate] ECE before: {ece_before:.4f}")
    print(f"[calibrate] ECE after:  {ece_after:.4f}")
    print(f"[calibrate] ECE improvement: {ece_before - ece_after:.4f}")

    if args.dry_run:
        print(f"[calibrate] Dry run — not saving. Set temperature manually to {optimal_T:.4f}.")
        return 0

    # Write optimal T back into cls_head.pt
    state = torch.load(CONFIG.cls_head_path, map_location="cpu")
    state["temperature"] = torch.tensor([optimal_T])
    torch.save(state, CONFIG.cls_head_path)
    print(f"[calibrate] Saved calibrated cls_head.pt  (T={optimal_T:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
