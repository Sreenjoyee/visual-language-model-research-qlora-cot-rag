"""Unit tests for the no-retraining eval improvements (#2 threshold, #4 bootstrap).

Pure-logic tests on synthetic data — no model, no GPU. Validates the metrics
helpers and the checkpoint weight-averaging math before they touch real data.
"""
import math

import pytest

from src.metrics import (
    auroc_score,
    binary_metrics,
    bootstrap_ci,
    expected_calibration_error,
    fuse_probabilities,
    metrics_at_threshold,
    optimal_threshold,
    predict_labels,
)


# ── predict_labels ──────────────────────────────────────────────────────────────

def test_predict_labels_threshold():
    assert predict_labels([0.4, 0.5, 0.6], threshold=0.5) == [0, 1, 1]
    assert predict_labels([0.9, 0.1], threshold=0.95) == [0, 0]


# ── optimal_threshold (Youden's J) ──────────────────────────────────────────────

def test_optimal_threshold_separable():
    # perfectly separable: threshold at 0.8 gives TPR=1, FPR=0, J=1
    yt = [0, 0, 1, 1]
    ys = [0.1, 0.2, 0.8, 0.9]
    out = optimal_threshold(yt, ys)
    assert out["youden_j"] == pytest.approx(1.0)
    # any threshold in (0.2, 0.8] perfectly separates; impl picks a score value
    assert 0.2 < out["threshold"] <= 0.8


def test_optimal_threshold_improves_over_half():
    # scores shifted high: at 0.5 everything is predicted positive (bad);
    # the optimal threshold should beat the default 0.5 F1.
    yt = [0, 0, 0, 1, 1, 1]
    ys = [0.55, 0.60, 0.65, 0.80, 0.85, 0.90]
    t = optimal_threshold(yt, ys)["threshold"]
    f1_default = metrics_at_threshold(yt, ys, 0.5)["f1"]
    f1_tuned = metrics_at_threshold(yt, ys, t)["f1"]
    assert f1_tuned >= f1_default
    assert f1_tuned == pytest.approx(1.0)  # this set is separable at ~0.8


def test_optimal_threshold_degenerate():
    assert optimal_threshold([1, 1, 1], [0.3, 0.4, 0.5]) == {"threshold": 0.5, "youden_j": 0.0}


# ── metrics_at_threshold ────────────────────────────────────────────────────────

def test_metrics_at_threshold_values():
    yt = [0, 0, 1, 1]
    ys = [0.1, 0.2, 0.8, 0.9]
    m = metrics_at_threshold(yt, ys, 0.5)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)

    # threshold 0.85 -> only 0.9 predicted positive
    m2 = metrics_at_threshold(yt, ys, 0.85)
    assert m2["recall"] == pytest.approx(0.5)
    assert m2["precision"] == pytest.approx(1.0)
    assert m2["accuracy"] == pytest.approx(0.75)


# ── bootstrap_ci ────────────────────────────────────────────────────────────────

def test_bootstrap_ci_brackets_point():
    yt = [0, 0, 0, 0, 1, 1, 1, 1]
    ys = [0.1, 0.2, 0.3, 0.45, 0.55, 0.7, 0.8, 0.9]
    out = bootstrap_ci(yt, ys, metric="auroc", n_boot=300, seed=1)
    assert out["ci_low"] <= out["point"] <= out["ci_high"]
    assert 0.0 <= out["ci_low"] <= 1.0
    assert 0.0 <= out["ci_high"] <= 1.0


def test_bootstrap_ci_deterministic():
    yt = [0, 1, 0, 1, 0, 1]
    ys = [0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    a = bootstrap_ci(yt, ys, metric="f1", threshold=0.5, n_boot=200, seed=42)
    b = bootstrap_ci(yt, ys, metric="f1", threshold=0.5, n_boot=200, seed=42)
    assert a == b  # same seed -> identical CI


def test_bootstrap_ci_point_matches_full():
    yt = [0, 0, 1, 1]
    ys = [0.1, 0.2, 0.8, 0.9]
    out = bootstrap_ci(yt, ys, metric="auroc", n_boot=100, seed=0)
    assert out["point"] == pytest.approx(auroc_score(yt, ys))


def test_bootstrap_ci_empty():
    out = bootstrap_ci([], [], metric="auroc")
    assert math.isnan(out["point"])


# ── checkpoint weight-averaging (#3) ─────────────────────────────────────────────

def _make_ckpt(base, step, tensors):
    import torch  # noqa: F401
    from safetensors.torch import save_file
    cd = base / f"lora_step{step}"
    cd.mkdir(parents=True)
    save_file(tensors, str(cd / "adapter_model.safetensors"))
    (cd / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    return cd


def test_average_adapters_is_exact_mean(tmp_path):
    import torch
    from safetensors.torch import load_file
    from experiments.average_checkpoints import average_adapters

    a = _make_ckpt(tmp_path, 100, {"w": torch.tensor([2.0, 4.0])})
    b = _make_ckpt(tmp_path, 200, {"w": torch.tensor([4.0, 8.0])})
    out = tmp_path / "avg"
    avg = average_adapters([a, b], out)

    assert torch.allclose(avg["w"], torch.tensor([3.0, 6.0]))
    loaded = load_file(str(out / "adapter_model.safetensors"))
    assert torch.allclose(loaded["w"], torch.tensor([3.0, 6.0]))
    assert (out / "adapter_config.json").exists()


def test_find_checkpoints_picks_last_n(tmp_path):
    import torch
    from experiments.average_checkpoints import find_checkpoints

    for s in [100, 200, 300, 400]:
        _make_ckpt(tmp_path, s, {"w": torch.tensor([1.0])})
    picks = find_checkpoints(tmp_path, last_n=2)
    assert [p.name for p in picks] == ["lora_step300", "lora_step400"]


def test_average_adapters_key_mismatch_raises(tmp_path):
    import torch
    from experiments.average_checkpoints import average_adapters

    a = _make_ckpt(tmp_path, 100, {"w": torch.tensor([1.0])})
    b = _make_ckpt(tmp_path, 200, {"other": torch.tensor([1.0])})
    with pytest.raises(ValueError):
        average_adapters([a, b], tmp_path / "avg")


# ── evaluate._compile_report wiring (#2 + #4 integration) ────────────────────────

def test_compile_report_emits_threshold_and_ci():
    from types import SimpleNamespace
    from experiments.evaluate import _compile_report
    from src.config import CONFIG

    rows = [
        ("NORMAL", 0.1), ("NORMAL", 0.2), ("NORMAL", 0.3), ("NORMAL", 0.4),
        ("ABNORMAL", 0.6), ("ABNORMAL", 0.7), ("ABNORMAL", 0.8), ("ABNORMAL", 0.9),
    ]
    results = [
        SimpleNamespace(
            true_label=lbl, pred_label=("ABNORMAL" if s >= 0.5 else "NORMAL"),
            p_abnormal=s, p_cls=s, p_lm=s, correct=True, evidence_used=[1],
            reasoning="x", latency_s=1.0, vram_peak_gb=0.3, source="mimic",
        )
        for lbl, s in rows
    ]
    rep = _compile_report(results, None, None, CONFIG)

    assert "optimal_threshold" in rep
    ot = rep["optimal_threshold"]
    assert set(ot) == {"threshold", "youden_j", "metrics_at_threshold"}
    assert ot["metrics_at_threshold"]["f1"] == pytest.approx(1.0)  # separable set

    assert "confidence_intervals_95" in rep
    ci = rep["confidence_intervals_95"]
    for key in ("auroc", "f1_at_optimal_threshold", "ece"):
        assert ci[key]["ci_low"] <= ci[key]["point"] <= ci[key]["ci_high"]

    # #2 dual-signal ablation + #6 calibration comparison sections present
    assert "dual_signal_ablation" in rep
    assert rep["dual_signal_ablation"]["fused_50_50_auroc"] is not None
    assert "calibration_comparison" in rep


# ── #2 dual-signal fusion ────────────────────────────────────────────────────────

def test_fuse_probabilities_convex():
    assert fuse_probabilities(0.8, 0.4, 0.5) == pytest.approx(0.6)
    assert fuse_probabilities(0.9, 0.1, 1.0) == pytest.approx(0.9)   # all weight on primary
    assert fuse_probabilities(0.9, 0.1, 0.0) == pytest.approx(0.1)   # all on secondary


def test_fuse_probabilities_handles_missing():
    assert fuse_probabilities(0.8, None) == pytest.approx(0.8)
    assert fuse_probabilities(None, 0.4) == pytest.approx(0.4)
    assert fuse_probabilities(None, None) == pytest.approx(0.5)


def test_fuse_probabilities_clamps_weight():
    assert fuse_probabilities(0.8, 0.2, 2.0) == pytest.approx(0.8)   # weight clamped to 1
    assert fuse_probabilities(0.8, 0.2, -1.0) == pytest.approx(0.2)  # clamped to 0


# ── #6 calibration comparison ────────────────────────────────────────────────────

def test_compare_calibration_reduces_ece_on_overconfident_data():
    from src.calibration_compare import compare_calibration
    # Overconfident: bin@0.9 is only 70% positive, bin@0.1 is 30% positive.
    y_true = [1] * 70 + [0] * 30 + [1] * 30 + [0] * 70
    y_prob = [0.9] * 100 + [0.1] * 100
    out = compare_calibration(y_true, y_prob, test_frac=0.5, seed=3)
    assert out["none"] > 0.1                       # genuinely miscalibrated
    assert out["temperature"] <= out["none"]       # temperature must not hurt; here it helps
    assert 0.0 <= out["temperature"] <= 1.0
    assert "best_method" in out


def test_compare_calibration_small_input_guard():
    from src.calibration_compare import compare_calibration
    out = compare_calibration([0, 1], [0.2, 0.8])
    assert "error" in out
