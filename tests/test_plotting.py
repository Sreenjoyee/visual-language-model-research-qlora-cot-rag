"""Tests for src/plotting.py — pure matplotlib, no GPU/model required."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotting import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_dual_axis_loss_vram,
    plot_evidence_histogram,
    plot_green_bar,
    plot_latency_distribution,
    plot_learning_curve,
    plot_rag_ablation,
    plot_roc_curve,
    plot_system_comparison,
    save_figure,
)


# ── shared synthetic data ──────────────────────────────────────────────────────

_LOG_RECORDS = [
    {"step": i * 25, "loss": max(0.1, 2.0 - i * 0.08), "vram_gb": 3.5 + i * 0.02}
    for i in range(1, 9)
]

_Y_TRUE   = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
_Y_PRED   = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
_Y_SCORES = [0.9, 0.1, 0.8, 0.6, 0.4, 0.85, 0.2, 0.7, 0.95, 0.15]

_GREEN = {
    "G_groundedness": 0.92,
    "R_reasoning":    0.78,
    "E_alignment":    0.95,
    "E_error_free":   0.97,
    "N_numerical":    0.84,
    "composite":      0.892,
}

_EVIDENCE_LISTS = [[1], [], [2, 3], [1], [], [3], [1, 2], [], [1], [2]]

_LATENCIES = {
    "NORMAL":   [1.1, 1.3, 0.9, 1.2, 1.0],
    "ABNORMAL": [1.5, 1.8, 1.4, 2.0, 1.6],
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _is_figure(obj) -> bool:
    return isinstance(obj, plt.Figure)


# ── plot_learning_curve ────────────────────────────────────────────────────────

class TestPlotLearningCurve:
    def test_returns_figure(self):
        fig = plot_learning_curve(_LOG_RECORDS, "Stage 1")
        assert _is_figure(fig)
        plt.close(fig)

    def test_has_one_axis(self):
        fig = plot_learning_curve(_LOG_RECORDS)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_title_contains_stage_name(self):
        fig = plot_learning_curve(_LOG_RECORDS, stage_name="Stage 2")
        ax = fig.axes[0]
        assert "Stage 2" in ax.get_title()
        plt.close(fig)

    def test_custom_color(self):
        fig = plot_learning_curve(_LOG_RECORDS, color="#FF0000")
        assert _is_figure(fig)
        plt.close(fig)


# ── plot_dual_axis_loss_vram ───────────────────────────────────────────────────

class TestPlotDualAxis:
    def test_returns_figure(self):
        fig = plot_dual_axis_loss_vram(_LOG_RECORDS, "Stage 1")
        assert _is_figure(fig)
        plt.close(fig)

    def test_has_two_axes(self):
        fig = plot_dual_axis_loss_vram(_LOG_RECORDS)
        assert len(fig.axes) == 2   # primary + twinx
        plt.close(fig)


# ── plot_confusion_matrix ──────────────────────────────────────────────────────

class TestPlotConfusionMatrix:
    def test_returns_figure(self):
        fig = plot_confusion_matrix(_Y_TRUE, _Y_PRED)
        assert _is_figure(fig)
        plt.close(fig)

    def test_title_set(self):
        fig = plot_confusion_matrix(_Y_TRUE, _Y_PRED)
        assert "Confusion" in fig.axes[0].get_title()
        plt.close(fig)

    def test_perfect_predictions(self):
        fig = plot_confusion_matrix([1, 0, 1], [1, 0, 1])
        assert _is_figure(fig)
        plt.close(fig)

    def test_all_zeros(self):
        fig = plot_confusion_matrix([0, 0], [0, 0])
        assert _is_figure(fig)
        plt.close(fig)


# ── plot_roc_curve ─────────────────────────────────────────────────────────────

class TestPlotRocCurve:
    def test_returns_figure(self):
        fig = plot_roc_curve(_Y_TRUE, _Y_SCORES)
        assert _is_figure(fig)
        plt.close(fig)

    def test_accepts_precomputed_auroc(self):
        fig = plot_roc_curve(_Y_TRUE, _Y_SCORES, auroc=0.87)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("0.87" in t for t in labels)
        plt.close(fig)

    def test_ylim_above_one(self):
        fig = plot_roc_curve(_Y_TRUE, _Y_SCORES)
        ylim = fig.axes[0].get_ylim()
        assert ylim[1] >= 1.0
        plt.close(fig)


# ── plot_calibration_curve ─────────────────────────────────────────────────────

class TestPlotCalibrationCurve:
    def test_returns_figure(self):
        fig = plot_calibration_curve(_Y_TRUE, _Y_SCORES)
        assert _is_figure(fig)
        plt.close(fig)

    def test_handles_uniform_scores(self):
        y_true   = [1, 0, 1, 0]
        y_probs  = [0.5, 0.5, 0.5, 0.5]
        fig = plot_calibration_curve(y_true, y_probs)
        assert _is_figure(fig)
        plt.close(fig)


# ── plot_latency_distribution ──────────────────────────────────────────────────

class TestPlotLatencyDistribution:
    def test_returns_figure(self):
        fig = plot_latency_distribution(_LATENCIES)
        assert _is_figure(fig)
        plt.close(fig)

    def test_single_class(self):
        fig = plot_latency_distribution({"NORMAL": [1.0, 1.2, 0.9]})
        assert _is_figure(fig)
        plt.close(fig)

    def test_x_tick_labels_match_keys(self):
        fig = plot_latency_distribution(_LATENCIES)
        ax = fig.axes[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert set(tick_labels) == set(_LATENCIES.keys())
        plt.close(fig)


# ── plot_green_bar ─────────────────────────────────────────────────────────────

class TestPlotGreenBar:
    def test_returns_figure(self):
        fig = plot_green_bar(_GREEN)
        assert _is_figure(fig)
        plt.close(fig)

    def test_title_contains_green(self):
        fig = plot_green_bar(_GREEN)
        assert "GREEN" in fig.axes[0].get_title()
        plt.close(fig)

    def test_partial_keys(self):
        fig = plot_green_bar({"composite": 0.85})
        assert _is_figure(fig)
        plt.close(fig)


# ── plot_evidence_histogram ────────────────────────────────────────────────────

class TestPlotEvidenceHistogram:
    def test_returns_figure(self):
        fig = plot_evidence_histogram(_EVIDENCE_LISTS)
        assert _is_figure(fig)
        plt.close(fig)

    def test_all_empty_citations(self):
        fig = plot_evidence_histogram([[], [], []])
        assert _is_figure(fig)
        plt.close(fig)

    def test_single_sample(self):
        fig = plot_evidence_histogram([[1, 2, 3]])
        assert _is_figure(fig)
        plt.close(fig)


# ── plot_system_comparison ────────────────────────────────────────────────────

_SYSTEMS = {
    "MEDDIAG (ours)": {"auroc": 0.85, "f1": 0.78},
    "Baseline A":     {"auroc": 0.80, "f1": 0.72},
    "Baseline B":     {"auroc": 0.83, "f1": 0.75},
}


class TestPlotSystemComparison:
    def test_returns_figure(self):
        fig = plot_system_comparison(_SYSTEMS)
        assert _is_figure(fig)
        plt.close(fig)

    def test_single_system(self):
        fig = plot_system_comparison({"Only": {"auroc": 0.90}})
        assert _is_figure(fig)
        plt.close(fig)

    def test_custom_metric_keys(self):
        fig = plot_system_comparison(_SYSTEMS, metric_keys=["f1"])
        ax = fig.axes[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert any("F1" in t.upper() for t in tick_labels)
        plt.close(fig)

    def test_ylim_above_zero(self):
        fig = plot_system_comparison(_SYSTEMS)
        assert fig.axes[0].get_ylim()[1] > 0
        plt.close(fig)

    def test_legend_contains_all_system_names(self):
        fig = plot_system_comparison(_SYSTEMS)
        legend_texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        for name in _SYSTEMS:
            assert any(name in t for t in legend_texts)
        plt.close(fig)


# ── plot_rag_ablation ─────────────────────────────────────────────────────────

_K_VALUES   = [1, 3, 5, 10]
_BS_MEANS   = [0.72, 0.75, 0.76, 0.74]
_BS_STDS    = [0.03, 0.02, 0.02, 0.03]
_LAT_MEANS  = [1.1, 1.3, 1.5, 2.0]


class TestPlotRagAblation:
    def test_returns_figure(self):
        fig = plot_rag_ablation(_K_VALUES, _BS_MEANS)
        assert _is_figure(fig)
        plt.close(fig)

    def test_with_stds_returns_figure(self):
        fig = plot_rag_ablation(_K_VALUES, _BS_MEANS, bertscore_stds=_BS_STDS)
        assert _is_figure(fig)
        plt.close(fig)

    def test_with_latency_returns_two_axes(self):
        fig = plot_rag_ablation(_K_VALUES, _BS_MEANS, latency_means=_LAT_MEANS)
        assert len(fig.axes) == 2  # primary + twinx
        plt.close(fig)

    def test_without_latency_returns_one_axis(self):
        fig = plot_rag_ablation(_K_VALUES, _BS_MEANS)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_full_args_returns_figure(self):
        fig = plot_rag_ablation(_K_VALUES, _BS_MEANS, _BS_STDS, _LAT_MEANS)
        assert _is_figure(fig)
        plt.close(fig)

    def test_single_k_value(self):
        fig = plot_rag_ablation([3], [0.75])
        assert _is_figure(fig)
        plt.close(fig)


# ── save_figure ────────────────────────────────────────────────────────────────

class TestSaveFigure:
    def test_saves_png(self, tmp_path):
        fig = plot_learning_curve(_LOG_RECORDS)
        out = save_figure(fig, tmp_path / "test_plot")
        assert out.exists()
        assert out.suffix == ".png"

    def test_saves_pdf(self, tmp_path):
        fig = plot_learning_curve(_LOG_RECORDS)
        out = save_figure(fig, tmp_path / "test_plot", fmt="pdf")
        assert out.exists()
        assert out.suffix == ".pdf"

    def test_creates_parent_dirs(self, tmp_path):
        fig = plot_learning_curve(_LOG_RECORDS)
        out = save_figure(fig, tmp_path / "a" / "b" / "plot")
        assert out.exists()

    def test_closes_figure(self, tmp_path):
        fig = plot_learning_curve(_LOG_RECORDS)
        save_figure(fig, tmp_path / "plot")
        # After save_figure, fig should be closed — plt.get_fignums should not include it
        assert fig.number not in plt.get_fignums()
