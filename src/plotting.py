"""Publication-quality visualization for MEDDIAG evaluation results.

All functions return a matplotlib Figure at 300 DPI and accept only plain
Python data structures — no model, no torch — so they are fully testable
without GPU or real eval results.

Plot catalogue:
    plot_learning_curve        — loss vs training step (stage 1 or 2 log)
    plot_dual_axis_loss_vram   — loss (left axis) + VRAM GB (right axis) vs step
    plot_confusion_matrix      — 2×2 heatmap with counts and row-normalised %
    plot_roc_curve             — ROC with shaded AUC region and chance diagonal
    plot_calibration_curve     — reliability diagram (fraction positive vs confidence)
    plot_latency_distribution  — violin + embedded box per prediction class
    plot_green_bar             — horizontal bar chart of G/R/E/E/N sub-scores
    plot_evidence_histogram    — histogram of evidence citation counts per output
    save_figure                — save any Figure to disk at 300 DPI
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for servers and tests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style constants ────────────────────────────────────────────────────────────

_DPI       = 300
_FONT_SIZE = 11
_PAL       = {
    "normal":   "#4C9BE8",   # blue
    "abnormal": "#E8644C",   # red-orange
    "vram":     "#6DBF67",   # green
    "neutral":  "#8C8C8C",   # gray
    "accent":   "#F2A93B",   # amber
}

plt.rcParams.update({
    "font.size":       _FONT_SIZE,
    "axes.titlesize":  _FONT_SIZE + 1,
    "axes.labelsize":  _FONT_SIZE,
    "xtick.labelsize": _FONT_SIZE - 1,
    "ytick.labelsize": _FONT_SIZE - 1,
    "legend.fontsize": _FONT_SIZE - 1,
    "figure.dpi":      _DPI,
    "axes.grid":       True,
    "grid.alpha":      0.25,
    "grid.linestyle":  "--",
})


def _new_fig(w: float = 7.0, h: float = 5.0):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def _finish(fig, tight: bool = True):
    if tight:
        fig.tight_layout()
    return fig


# ── Learning curve ─────────────────────────────────────────────────────────────

def plot_learning_curve(
    log_records: list[dict],
    stage_name: str = "Stage",
    color: str | None = None,
) -> plt.Figure:
    """Loss vs training step from a stage JSONL log.

    Args:
        log_records: list of dicts with keys 'step' and 'loss'.
        stage_name:  label shown in the title and legend.
        color:       line colour (defaults to blue for stage 1, red for stage 2).
    """
    steps  = [r["step"] for r in log_records]
    losses = [r["loss"] for r in log_records]

    if color is None:
        color = _PAL["normal"] if "1" in stage_name else _PAL["abnormal"]

    fig, ax = _new_fig()
    ax.plot(steps, losses, color=color, linewidth=1.8, label=stage_name)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{stage_name} — Learning Curve")
    ax.legend()
    return _finish(fig)


# ── Dual-axis loss + VRAM ──────────────────────────────────────────────────────

def plot_dual_axis_loss_vram(
    log_records: list[dict],
    stage_name: str = "Stage",
) -> plt.Figure:
    """Loss (left axis) and VRAM GB (right axis) vs training step.

    Args:
        log_records: dicts with keys 'step', 'loss', 'vram_gb'.
    """
    steps  = [r["step"]    for r in log_records]
    losses = [r["loss"]    for r in log_records]
    vrams  = [r.get("vram_gb", 0.0) for r in log_records]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_loss = _PAL["normal"]
    color_vram = _PAL["vram"]

    ax1.plot(steps, losses, color=color_loss, linewidth=1.8, label="Loss")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Loss", color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax2 = ax1.twinx()
    ax2.plot(steps, vrams, color=color_vram, linewidth=1.4, linestyle="--", label="VRAM (GB)")
    ax2.set_ylabel("VRAM (GB)", color=color_vram)
    ax2.tick_params(axis="y", labelcolor=color_vram)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title(f"{stage_name} — Loss & VRAM")
    fig.tight_layout()
    return fig


# ── Confusion matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: tuple[str, str] = ("NORMAL", "ABNORMAL"),
) -> plt.Figure:
    """2×2 confusion matrix with counts and row-normalised percentages.

    Args:
        y_true:       binary ground-truth labels (0 / 1).
        y_pred:       binary predicted labels   (0 / 1).
        class_names:  labels for 0 and 1 respectively.
    """
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1

    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_pct   = cm / row_sums * 100

    fig, ax = _new_fig(5.5, 4.5)
    ax.grid(False)
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Row %")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred {c}" for c in class_names])
    ax.set_yticklabels([f"True {c}" for c in class_names])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = cm_pct[i, j]
            text_color = "white" if pct > 60 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", color=text_color, fontsize=10)

    return _finish(fig)


# ── ROC curve ─────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    auroc: float | None = None,
) -> plt.Figure:
    """ROC curve with shaded AUC region and diagonal chance line.

    Args:
        y_true:   binary ground-truth labels (0 / 1).
        y_scores: predicted probability for the positive class.
        auroc:    pre-computed AUROC to display in the legend (recomputed if None).
    """
    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    tprs, fprs = [0.0], [0.0]
    tp = fp = 0
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / max(n_pos, 1))
        fprs.append(fp / max(n_neg, 1))
    tprs.append(1.0)
    fprs.append(1.0)

    if auroc is None:
        trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auroc = float(trapz(tprs, fprs))

    fig, ax = _new_fig()
    ax.plot(fprs, tprs, color=_PAL["abnormal"], linewidth=2.0,
            label=f"MEDDIAG (AUROC = {auroc:.4f})")
    ax.fill_between(fprs, tprs, alpha=0.12, color=_PAL["abnormal"])
    ax.plot([0, 1], [0, 1], color=_PAL["neutral"], linewidth=1.2,
            linestyle="--", label="Chance (AUROC = 0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    return _finish(fig)


# ── Calibration curve (reliability diagram) ───────────────────────────────────

def plot_calibration_curve(
    y_true: Sequence[int],
    y_probs: Sequence[float],
    n_bins: int = 10,
) -> plt.Figure:
    """Reliability diagram: actual fraction positive vs mean confidence per bin.

    Args:
        y_true:  binary ground-truth labels (0 / 1).
        y_probs: predicted probability for positive class.
        n_bins:  number of equal-width confidence bins.
    """
    step = 1.0 / n_bins
    bin_centers, fracs, confs = [], [], []

    for b in range(n_bins):
        lo, hi = b * step, (b + 1) * step
        is_last = b == n_bins - 1
        idxs = [
            i for i, p in enumerate(y_probs)
            if (lo <= p <= hi if is_last else lo <= p < hi)
        ]
        if not idxs:
            continue
        fracs.append(sum(y_true[i] for i in idxs) / len(idxs))
        confs.append(sum(y_probs[i] for i in idxs) / len(idxs))
        bin_centers.append((lo + hi) / 2)

    fig, ax = _new_fig()
    ax.plot([0, 1], [0, 1], color=_PAL["neutral"], linestyle="--",
            linewidth=1.2, label="Perfect calibration")
    ax.bar(bin_centers, fracs, width=step * 0.8,
           color=_PAL["normal"], alpha=0.7, label="Fraction positive")
    ax.plot(confs, fracs, color=_PAL["abnormal"], linewidth=1.8,
            marker="o", markersize=5, label="Model calibration")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of positives (ABNORMAL)")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    return _finish(fig)


# ── Latency distribution ───────────────────────────────────────────────────────

def plot_latency_distribution(
    latencies_by_label: dict[str, list[float]],
) -> plt.Figure:
    """Violin + embedded box plot of per-sample latency grouped by predicted class.

    Args:
        latencies_by_label: mapping label → list of latency seconds.
                            e.g. {"NORMAL": [...], "ABNORMAL": [...]}
    """
    labels = list(latencies_by_label.keys())
    data   = [latencies_by_label[k] for k in labels]
    colors = [
        _PAL.get(k.lower(), _PAL["neutral"]) for k in labels
    ]

    fig, ax = _new_fig(6.0, 5.0)
    parts = ax.violinplot(data, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.55)

    # Overlay box plot
    bp = ax.boxplot(data, patch_artist=True, widths=0.15,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Inference latency (s)")
    ax.set_title("Latency Distribution by Prediction Class")
    return _finish(fig)


# ── GREEN judge bar chart ──────────────────────────────────────────────────────

def plot_green_bar(green_dict: dict[str, float]) -> plt.Figure:
    """Horizontal bar chart of G/R/E/E/N sub-scores and composite.

    Args:
        green_dict: output of metrics.green_judge() — keys G_*, R_*, E_*, N_*, composite.
    """
    label_map = {
        "G_groundedness": "G — Groundedness",
        "R_reasoning":    "R — Reasoning",
        "E_alignment":    "E — Evidence alignment",
        "E_error_free":   "E — Error-free rate",
        "N_numerical":    "N — Numerical accuracy",
        "composite":      "GREEN composite",
    }
    keys   = [k for k in label_map if k in green_dict]
    values = [green_dict[k] for k in keys]
    ylabels = [label_map[k] for k in keys]

    colors = [
        _PAL["accent"] if k == "composite" else _PAL["normal"]
        for k in keys
    ]

    fig, ax = _new_fig(8.0, 4.5)
    bars = ax.barh(ylabels, values, color=colors, edgecolor="white", height=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title("GREEN Multi-criteria Evaluation")
    ax.invert_yaxis()   # top-to-bottom reading order

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9,
        )

    patches = [
        mpatches.Patch(color=_PAL["normal"], label="Sub-score"),
        mpatches.Patch(color=_PAL["accent"], label="Composite"),
    ]
    ax.legend(handles=patches, loc="lower right")
    return _finish(fig)


# ── Evidence citation histogram ────────────────────────────────────────────────

def plot_evidence_histogram(
    evidence_lists: Sequence[list[int]],
) -> plt.Figure:
    """Histogram of evidence citation counts per model output.

    Args:
        evidence_lists: list of evidence_used lists from ScoredResult objects.
    """
    counts = [len(ev) for ev in evidence_lists]
    max_c  = max(counts, default=0)
    bins   = list(range(0, max_c + 2))

    fig, ax = _new_fig(6.0, 4.5)
    ax.hist(counts, bins=bins, align="left", rwidth=0.75,
            color=_PAL["normal"], edgecolor="white")
    ax.set_xlabel("Evidence snippets cited per output")
    ax.set_ylabel("Count")
    ax.set_title("Evidence Citation Distribution")
    ax.set_xticks(range(0, max_c + 1))
    return _finish(fig)


# ── System comparison (aggregate metrics per system) ──────────────────────────

def plot_system_comparison(
    metrics: dict[str, dict[str, float]],
    metric_keys: list[str] | None = None,
    title: str = "System Comparison",
) -> plt.Figure:
    """Grouped bar chart comparing aggregate metrics across systems.

    Each bar = one system's aggregate score (e.g. mean AUROC, mean F1).
    Not per-sample bars — each bar represents the whole system's performance.

    Args:
        metrics:     {system_name: {metric_name: aggregate_score}}
        metric_keys: which metrics to show (default: all keys from first system)
        title:       chart title
    """
    systems = list(metrics.keys())
    if metric_keys is None:
        metric_keys = list(next(iter(metrics.values())).keys())

    n_systems = len(systems)
    n_metrics = len(metric_keys)
    x = np.arange(n_metrics)
    bar_w = 0.8 / max(n_systems, 1)

    colors = list(_PAL.values()) + ["#9B59B6", "#1ABC9C", "#F39C12", "#2C3E50"]

    fig, ax = _new_fig(max(7.0, n_metrics * 1.8 + 1), 5.0)
    for i, system in enumerate(systems):
        vals = [metrics[system].get(k, 0.0) for k in metric_keys]
        offset = (i - n_systems / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w * 0.9,
                      label=system, color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper().replace("_", " ") for k in metric_keys])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (aggregate)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    return _finish(fig)


# ── RAG k-value ablation line chart ───────────────────────────────────────────

def plot_rag_ablation(
    k_values: list[int],
    bertscore_means: list[float],
    bertscore_stds: list[float] | None = None,
    latency_means: list[float] | None = None,
) -> plt.Figure:
    """Line plot of mean BERTScore F1 and latency vs RAG k value.

    Each point = mean over all eval samples for that k (aggregate view).
    Error band = ±1 std across samples.

    Args:
        k_values:        RAG k values evaluated (x-axis).
        bertscore_means: Mean BERTScore F1 per k.
        bertscore_stds:  Std of BERTScore F1 per k (optional error band).
        latency_means:   Mean latency in seconds per k (optional secondary axis).
    """
    color_bs  = _PAL["normal"]
    color_lat = _PAL["abnormal"]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(k_values, bertscore_means, color=color_bs, linewidth=2.0,
             marker="o", markersize=6, label="Mean BERTScore F1")
    if bertscore_stds is not None:
        lo = [m - s for m, s in zip(bertscore_means, bertscore_stds)]
        hi = [m + s for m, s in zip(bertscore_means, bertscore_stds)]
        ax1.fill_between(k_values, lo, hi, alpha=0.18, color=color_bs, label="±1 std")
    ax1.set_xlabel("RAG top-k retrieved snippets")
    ax1.set_ylabel("Mean BERTScore F1", color=color_bs)
    ax1.tick_params(axis="y", labelcolor=color_bs)
    ax1.set_xticks(k_values)
    ax1.grid(True, alpha=0.25, linestyle="--")

    if latency_means is not None:
        ax2 = ax1.twinx()
        ax2.plot(k_values, latency_means, color=color_lat, linewidth=1.6,
                 marker="s", markersize=5, linestyle="--", label="Mean latency (s)")
        ax2.set_ylabel("Mean latency (s)", color=color_lat)
        ax2.tick_params(axis="y", labelcolor=color_lat)
        ax2.grid(False)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    else:
        ax1.legend()

    ax1.set_title("Exp 7 — RAG k Ablation: Mean BERTScore F1 vs Latency")
    fig.tight_layout()
    return fig


# ── Save utility ───────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, path: Path | str, fmt: str = "png") -> Path:
    """Save a figure at 300 DPI.

    Args:
        fig:  matplotlib Figure to save.
        path: destination path (extension overridden by fmt if not present).
        fmt:  'png' or 'pdf'.

    Returns:
        Resolved Path of the saved file.
    """
    p = Path(path)
    if p.suffix.lstrip(".") not in {"png", "pdf", "svg"}:
        p = p.with_suffix(f".{fmt}")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return p.resolve()
