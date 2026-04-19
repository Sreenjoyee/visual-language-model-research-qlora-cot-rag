"""Phase 7 — MEDDIAG Visualization Suite.

Reads evaluation report JSON and stage training logs, then produces
publication-quality plots (300 DPI PNG or PDF) in the diagnostics/ directory.

Plots generated
───────────────
  learning_curve_stage1.png     — loss vs step (Stage 1 projector training)
  learning_curve_stage2.png     — loss vs step (Stage 2 LoRA training)
  dual_axis_stage1.png          — loss + VRAM (Stage 1)
  dual_axis_stage2.png          — loss + VRAM (Stage 2)
  confusion_matrix.png          — 2×2 heatmap with counts and row %
  roc_curve.png                 — ROC with shaded AUC region
  calibration_curve.png         — reliability diagram
  latency_distribution.png      — violin + box by prediction class
  green_judge.png               — horizontal bar G/R/E/E/N sub-scores
  evidence_histogram.png        — citation count histogram

Usage
─────
    python -m experiments.visualize \\
        --eval-report logs/eval_report_20240101_000000.json \\
        --out-dir diagnostics/

    # Also render stage training curves
    python -m experiments.visualize \\
        --eval-report logs/eval_report_*.json \\
        --stage1-log logs/stage1.jsonl \\
        --stage2-log logs/stage2.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import CONFIG
from src.plotting import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_dual_axis_loss_vram,
    plot_evidence_histogram,
    plot_green_bar,
    plot_latency_distribution,
    plot_learning_curve,
    plot_roc_curve,
    save_figure,
)


# ── Loader helpers ────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_eval_report(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Plot dispatcher ───────────────────────────────────────────────────────────

def render_training_plots(
    log_records: list[dict],
    stage_name: str,
    out_dir: Path,
    fmt: str,
) -> list[Path]:
    """Render learning curve + dual-axis plot for one training stage log."""
    saved: list[Path] = []
    slug = stage_name.lower().replace(" ", "_")

    fig = plot_learning_curve(log_records, stage_name=stage_name)
    saved.append(save_figure(fig, out_dir / f"learning_curve_{slug}", fmt=fmt))

    # Dual-axis only if VRAM data is present
    if any("vram_gb" in r for r in log_records):
        fig = plot_dual_axis_loss_vram(log_records, stage_name=stage_name)
        saved.append(save_figure(fig, out_dir / f"dual_axis_{slug}", fmt=fmt))

    return saved


def render_eval_plots(
    report: dict,
    out_dir: Path,
    fmt: str,
) -> list[Path]:
    """Render all evaluation-derived plots from a report dict."""
    saved: list[Path] = []
    per_sample = report.get("per_sample", [])

    if not per_sample:
        print("[visualize] WARNING: no per_sample data in report — skipping most plots.")
        if "green_judge" in report:
            fig = plot_green_bar(report["green_judge"])
            saved.append(save_figure(fig, out_dir / "green_judge", fmt=fmt))
        return saved

    y_true   = [1 if s["true"] == "ABNORMAL" else 0 for s in per_sample]
    y_pred   = [1 if s["pred"] == "ABNORMAL" else 0 for s in per_sample]
    y_scores = [s["p_abnormal"] for s in per_sample]
    ev_lists = [s["evidence_used"] for s in per_sample]

    lat_by_label: dict[str, list[float]] = {"NORMAL": [], "ABNORMAL": []}
    for s in per_sample:
        key = s["pred"] if s["pred"] in lat_by_label else "NORMAL"
        lat_by_label[key].append(s["latency_s"])

    auroc = report.get("metrics", {}).get("auroc")

    # Confusion matrix
    fig = plot_confusion_matrix(y_true, y_pred)
    saved.append(save_figure(fig, out_dir / "confusion_matrix", fmt=fmt))

    # ROC curve
    fig = plot_roc_curve(y_true, y_scores, auroc=auroc)
    saved.append(save_figure(fig, out_dir / "roc_curve", fmt=fmt))

    # Calibration
    fig = plot_calibration_curve(y_true, y_scores)
    saved.append(save_figure(fig, out_dir / "calibration_curve", fmt=fmt))

    # Latency distribution (only if both classes have data)
    non_empty = {k: v for k, v in lat_by_label.items() if v}
    if non_empty:
        fig = plot_latency_distribution(non_empty)
        saved.append(save_figure(fig, out_dir / "latency_distribution", fmt=fmt))

    # GREEN bar
    if "green_judge" in report:
        fig = plot_green_bar(report["green_judge"])
        saved.append(save_figure(fig, out_dir / "green_judge", fmt=fmt))

    # Evidence histogram
    fig = plot_evidence_histogram(ev_lists)
    saved.append(save_figure(fig, out_dir / "evidence_histogram", fmt=fmt))

    return saved


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="MEDDIAG Phase 7 — Visualization Suite")
    ap.add_argument(
        "--eval-report",
        type=Path,
        default=None,
        help="Path to eval_report JSON produced by experiments/evaluate.py.",
    )
    ap.add_argument(
        "--stage1-log",
        type=Path,
        default=CONFIG.logs_dir / "stage1.jsonl",
        help="Stage-1 training log (JSONL).",
    )
    ap.add_argument(
        "--stage2-log",
        type=Path,
        default=CONFIG.logs_dir / "stage2.jsonl",
        help="Stage-2 training log (JSONL).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=CONFIG.diagnostics_dir,
        help="Output directory for plots (default: diagnostics/).",
    )
    ap.add_argument(
        "--format",
        choices=["png", "pdf"],
        default="png",
        help="Output image format (default: png).",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt: str = args.format
    saved: list[Path] = []

    # ── Training logs ─────────────────────────────────────────────────────────
    for log_path, stage_name in [
        (args.stage1_log, "stage1"),
        (args.stage2_log, "stage2"),
    ]:
        if log_path and log_path.exists():
            print(f"[visualize] Loading {log_path.name}...")
            records = _load_jsonl(log_path)
            if records:
                new = render_training_plots(records, stage_name, out_dir, fmt)
                saved.extend(new)
                print(f"  -> {len(new)} plot(s) saved")
            else:
                print(f"  [visualize] {log_path.name} is empty — skipped.")
        else:
            print(f"[visualize] {log_path} not found — skipping training plots.")

    # ── Eval report ───────────────────────────────────────────────────────────
    if args.eval_report:
        if not args.eval_report.exists():
            print(f"[visualize] ERROR: eval report not found: {args.eval_report}")
            return 1
        print(f"[visualize] Loading {args.eval_report.name}...")
        report = _load_eval_report(args.eval_report)
        new = render_eval_plots(report, out_dir, fmt)
        saved.extend(new)
        print(f"  -> {len(new)} plot(s) saved")
    else:
        print("[visualize] No --eval-report provided; skipping evaluation plots.")

    if not saved:
        print("[visualize] Nothing to plot. Provide --eval-report and/or training logs.")
        return 1

    print(f"\n[visualize] Done. {len(saved)} plot(s) saved to {out_dir}/")
    for p in saved:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
