"""Phase 6 — MEDDIAG Evaluation Suite.

Metrics computed
────────────────
Binary classification : accuracy, precision, recall, F1
AUROC                 : from first-token P(ABNORMAL) via generate(output_scores=True)
ECE                   : Expected Calibration Error (10 equal-width bins)
Evidence citation rate: fraction of outputs citing at least one snippet
Reasoning completeness: fraction with all 4 CoT steps present
Evidence alignment    : fraction with valid cited IDs (1 ≤ id ≤ retrieval_top_k)
Unparseable rate      : fraction returning UNPARSEABLE
Latency               : mean, p50, p95, p99 per-sample seconds
VRAM                  : peak GB per sample
RAG ablation          : optional AUROC delta (with vs without retrieval)
Robustness            : optional AUROC under brightness / contrast augmentation
GREEN composite       : G/R/E/E/N equal-weighted multi-criteria score

Output
──────
logs/eval_report_<timestamp>.json   — full structured report

Usage
─────
    python -m experiments.evaluate \\
        --projector-path models/projector_stage1.pt \\
        --lora-adapter-dir models/lora_stage2 \\
        --max-samples 200 \\
        --rag-ablation \\
        --robustness
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import CONFIG, Config
from src.data.balanced_stream import LabeledPair, balanced_mimic_stream
from src.eval_runner import ScoredResult, run_eval_stream
from src.metrics import (
    auroc_score,
    binary_metrics,
    evidence_alignment_rate,
    evidence_citation_rate,
    expected_calibration_error,
    green_judge,
    latency_stats,
    reasoning_completeness_score,
    unparseable_rate,
)
from src.pipeline import MeddiagPipeline


# ── Robustness ────────────────────────────────────────────────────────────────

def _augment_stream(base_stream, augment_fn):
    """Wrap a LabeledPair stream, applying augment_fn to each image."""
    for pair in base_stream:
        yield LabeledPair(
            image=augment_fn(pair.image),
            report=pair.report,
            label=pair.label,
            source=pair.source,
        )


def _run_robustness(
    pipeline: MeddiagPipeline,
    config: Config,
    n_samples: int,
) -> dict[str, float]:
    """AUROC under 4 PIL augmentations: brightness ×0.5/×2, contrast ×0.5/×2."""
    from PIL import ImageEnhance

    augmentations: dict[str, object] = {
        "brightness_0.5": lambda img: ImageEnhance.Brightness(img.convert("RGB")).enhance(0.5),
        "brightness_2.0": lambda img: ImageEnhance.Brightness(img.convert("RGB")).enhance(2.0),
        "contrast_0.5":   lambda img: ImageEnhance.Contrast(img.convert("RGB")).enhance(0.5),
        "contrast_2.0":   lambda img: ImageEnhance.Contrast(img.convert("RGB")).enhance(2.0),
    }

    result: dict[str, float] = {}
    for aug_name, aug_fn in augmentations.items():
        print(f"  [robustness] {aug_name} ({n_samples} samples)...")
        stream = balanced_mimic_stream(config, split="train", max_pairs=n_samples)
        aug_stream = _augment_stream(stream, aug_fn)
        res = run_eval_stream(pipeline, aug_stream, max_samples=n_samples)
        if not res:
            result[f"{aug_name}_auroc"] = float("nan")
            continue
        y_true  = [1 if r.true_label == "ABNORMAL" else 0 for r in res]
        y_scores = [r.p_abnormal for r in res]
        result[f"{aug_name}_auroc"] = auroc_score(y_true, y_scores)
        print(f"    AUROC = {result[f'{aug_name}_auroc']:.4f}")

    return result


# ── Report ────────────────────────────────────────────────────────────────────

def _compile_report(
    results: list[ScoredResult],
    rag_ablation_auroc: float | None,
    robustness: dict[str, float] | None,
    config: Config,
) -> dict:
    n = len(results)
    y_true      = [1 if r.true_label == "ABNORMAL" else 0 for r in results]
    y_pred      = [1 if r.pred_label == "ABNORMAL" else 0 for r in results]
    y_scores    = [r.p_abnormal for r in results]
    diagnoses   = [r.pred_label for r in results]
    ev_lists    = [r.evidence_used for r in results]
    reasonings  = [r.reasoning for r in results]
    latencies   = [r.latency_s for r in results]
    vrams       = [r.vram_peak_gb for r in results]

    clf  = binary_metrics(y_true, y_pred)
    auc  = auroc_score(y_true, y_scores)
    ece  = expected_calibration_error(y_true, y_scores)
    cit  = evidence_citation_rate(ev_lists)
    rcs  = reasoning_completeness_score(reasonings)
    upr  = unparseable_rate(diagnoses)
    eal  = evidence_alignment_rate(ev_lists, config.retrieval_top_k)
    lat  = latency_stats(latencies)
    green = green_judge(
        groundedness=cit,
        reasoning=rcs,
        evidence_alignment=eal,
        error_free=1.0 - upr,
        numerical=(auc + clf["f1"]) / 2,
    )

    report: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "n_samples": n,
        "metrics": {
            **clf,
            "auroc": auc,
            "ece": ece,
            "evidence_citation_rate": cit,
            "reasoning_completeness": rcs,
            "evidence_alignment_rate": eal,
            "unparseable_rate": upr,
        },
        "latency": lat,
        "vram": {
            "peak_gb": round(max(vrams) if vrams else 0.0, 3),
            "mean_gb": round(sum(vrams) / n if n else 0.0, 3),
        },
        "green_judge": green,
    }

    if rag_ablation_auroc is not None:
        report["rag_ablation"] = {
            "with_rag_auroc":    auc,
            "without_rag_auroc": rag_ablation_auroc,
            "delta":             round(auc - rag_ablation_auroc, 4),
        }

    if robustness:
        report["robustness"] = robustness

    report["per_sample"] = [
        {
            "true":         r.true_label,
            "pred":         r.pred_label,
            "p_abnormal":   round(r.p_abnormal, 4),
            "correct":      r.correct,
            "evidence_used": r.evidence_used,
            "latency_s":    r.latency_s,
            "vram_peak_gb": r.vram_peak_gb,
            "source":       r.source,
        }
        for r in results
    ]
    return report


def _print_summary(report: dict) -> None:
    m = report["metrics"]
    g = report["green_judge"]
    lat = report["latency"]
    n = report["n_samples"]

    print("\n" + "=" * 62)
    print(f"  MEDDIAG Evaluation — {n} samples")
    print("=" * 62)
    print(f"  Accuracy   : {m['accuracy']:.4f}")
    print(f"  Precision  : {m['precision']:.4f}")
    print(f"  Recall     : {m['recall']:.4f}")
    print(f"  F1         : {m['f1']:.4f}")
    print(f"  AUROC      : {m['auroc']:.4f}")
    print(f"  ECE        : {m['ece']:.4f}")
    print(f"  Parseable  : {1 - m['unparseable_rate']:.4f}")
    print(f"  Cites evid.: {m['evidence_citation_rate']:.4f}")
    print(f"  Reasoning  : {m['reasoning_completeness']:.4f}")
    print(
        f"\n  GREEN composite: {g['composite']:.4f}"
        f"  (G={g['G_groundedness']:.3f} R={g['R_reasoning']:.3f}"
        f" E={g['E_alignment']:.3f} E={g['E_error_free']:.3f}"
        f" N={g['N_numerical']:.3f})"
    )
    print(f"\n  Latency: mean={lat['mean_s']}s  p95={lat['p95_s']}s  p99={lat['p99_s']}s")
    v = report["vram"]
    print(f"  VRAM peak: {v['peak_gb']:.2f}GB  mean: {v['mean_gb']:.2f}GB")

    if "rag_ablation" in report:
        r = report["rag_ablation"]
        print(
            f"\n  RAG delta: {r['delta']:+.4f}"
            f"  (w/ {r['with_rag_auroc']:.4f} vs w/o {r['without_rag_auroc']:.4f})"
        )

    if "robustness" in report:
        print("\n  Robustness (AUROC):")
        for k, v in report["robustness"].items():
            label = k.replace("_auroc", "")
            val = f"{v:.4f}" if v == v else "NaN"  # NaN check
            print(f"    {label:<20}: {val}")

    # Youden J optimal threshold — copy this into src/config.py after each full run
    samples = report.get("per_sample", [])
    if len(samples) >= 10:
        from sklearn.metrics import roc_curve
        yt = [1 if s["true"] == "ABNORMAL" else 0 for s in samples]
        yp = [s["p_abnormal"] for s in samples]
        fpr, tpr, thresholds = roc_curve(yt, yp)
        j = tpr - fpr
        best = int(np.argmax(j))
        print(f"\n  Optimal threshold (Youden J={j[best]:.4f}): {thresholds[best]:.4f}")
        print(f"  → Update src/config.py classification_threshold to {thresholds[best]:.4f}")

    print("=" * 62)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="MEDDIAG Phase 6 — Evaluation Suite")
    ap.add_argument(
        "--projector-path",
        type=Path,
        default=CONFIG.models_dir / "projector_stage1.pt",
    )
    ap.add_argument(
        "--lora-adapter-dir",
        type=Path,
        default=None,
        help="Optional Stage-2 LoRA adapter directory.",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Balanced samples to evaluate (half NORMAL, half ABNORMAL).",
    )
    ap.add_argument(
        "--rag-ablation",
        action="store_true",
        help="Also run inference without retrieval; report AUROC delta.",
    )
    ap.add_argument(
        "--robustness",
        action="store_true",
        help="Run robustness evaluation under brightness/contrast augmentations.",
    )
    ap.add_argument(
        "--robustness-samples",
        type=int,
        default=50,
        help="Samples per augmentation (default 50 to keep runtime tractable).",
    )
    ap.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="JSON output path. Defaults to logs/eval_report_<timestamp>.json.",
    )
    args = ap.parse_args()

    CONFIG.validate(require_token=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file: Path = args.out_file or (CONFIG.logs_dir / f"eval_report_{ts}.json")

    print("[eval] Loading pipeline...")
    pipeline = MeddiagPipeline(
        config=CONFIG,
        projector_weights=args.projector_path,
        lora_adapter_dir=args.lora_adapter_dir,
    )

    # ── Main evaluation ───────────────────────────────────────────────────────
    print(f"\n[eval] Main evaluation — {args.max_samples} balanced samples...")
    stream = balanced_mimic_stream(CONFIG, split="train", max_pairs=args.max_samples)
    results = run_eval_stream(pipeline, iter(stream), max_samples=args.max_samples)

    if not results:
        print("[eval] ERROR: no samples evaluated — check dataset access.")
        return 1

    # ── RAG ablation ──────────────────────────────────────────────────────────
    rag_ablation_auroc: float | None = None
    if args.rag_ablation:
        print(f"\n[eval] RAG ablation — {args.max_samples} samples (no retrieval)...")
        abl_stream = balanced_mimic_stream(
            CONFIG, split="train", max_pairs=args.max_samples
        )
        abl_results = run_eval_stream(
            pipeline, iter(abl_stream), max_samples=args.max_samples, no_rag=True
        )
        if abl_results:
            y_true_abl  = [1 if r.true_label == "ABNORMAL" else 0 for r in abl_results]
            y_score_abl = [r.p_abnormal for r in abl_results]
            rag_ablation_auroc = auroc_score(y_true_abl, y_score_abl)
            print(f"[eval] No-RAG AUROC: {rag_ablation_auroc:.4f}")

    # ── Robustness ────────────────────────────────────────────────────────────
    robustness_dict: dict[str, float] | None = None
    if args.robustness:
        print(f"\n[eval] Robustness — {args.robustness_samples} samples per augmentation...")
        robustness_dict = _run_robustness(pipeline, CONFIG, args.robustness_samples)

    # ── Compile + print + save ────────────────────────────────────────────────
    report = _compile_report(results, rag_ablation_auroc, robustness_dict, CONFIG)
    _print_summary(report)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n[eval] Report saved → {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
