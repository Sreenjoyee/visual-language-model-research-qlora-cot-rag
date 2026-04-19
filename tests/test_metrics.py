"""Tests for src/metrics.py — pure functions, no GPU/model required."""
from __future__ import annotations

import math

import pytest

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


# ── binary_metrics ─────────────────────────────────────────────────────────────

class TestBinaryMetrics:
    def test_perfect_predictions(self):
        m = binary_metrics([1, 0, 1, 0], [1, 0, 1, 0])
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_all_wrong(self):
        m = binary_metrics([1, 0, 1, 0], [0, 1, 0, 1])
        assert m["accuracy"] == 0.0
        assert m["f1"] == 0.0

    def test_all_predicted_positive(self):
        m = binary_metrics([1, 0, 1, 0], [1, 1, 1, 1])
        assert m["recall"] == 1.0
        assert m["precision"] == 0.5

    def test_support_counts(self):
        m = binary_metrics([1, 1, 0, 0, 0], [1, 0, 0, 0, 1])
        assert m["support_pos"] == 2   # two actual positives
        assert m["support_neg"] == 3   # three actual negatives

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            binary_metrics([1, 0], [1, 0, 1])

    def test_all_negative_no_crash(self):
        # precision=0 when no positive predictions — safe_div returns 0
        m = binary_metrics([0, 0], [0, 0])
        assert m["accuracy"] == 1.0
        assert m["precision"] == 0.0
        assert m["f1"] == 0.0


# ── auroc_score ────────────────────────────────────────────────────────────────

class TestAurocScore:
    def test_perfect_ranking(self):
        y_true   = [1, 1, 0, 0]
        y_scores = [0.9, 0.8, 0.3, 0.1]
        assert auroc_score(y_true, y_scores) == 1.0

    def test_random_ranking(self):
        y_true   = [1, 0, 1, 0]
        y_scores = [0.5, 0.5, 0.5, 0.5]
        auc = auroc_score(y_true, y_scores)
        assert 0.0 <= auc <= 1.0

    def test_degenerate_single_class_returns_half(self):
        assert auroc_score([0, 0, 0], [0.9, 0.5, 0.1]) == 0.5
        assert auroc_score([1, 1, 1], [0.9, 0.5, 0.1]) == 0.5

    def test_worst_ranking(self):
        y_true   = [1, 1, 0, 0]
        y_scores = [0.1, 0.2, 0.8, 0.9]
        auc = auroc_score(y_true, y_scores)
        assert auc < 0.5   # inverted ranking

    def test_returns_float(self):
        result = auroc_score([1, 0], [0.9, 0.1])
        assert isinstance(result, float)


# ── expected_calibration_error ─────────────────────────────────────────────────

class TestECE:
    def test_perfect_calibration(self):
        # If every sample has p=1.0 and is truly positive, ECE=0
        y_true  = [1, 1, 1, 1]
        y_probs = [1.0, 1.0, 1.0, 1.0]
        ece = expected_calibration_error(y_true, y_probs)
        assert ece == 0.0

    def test_worst_calibration(self):
        # p=1.0 for true negatives
        y_true  = [0, 0]
        y_probs = [1.0, 1.0]
        ece = expected_calibration_error(y_true, y_probs)
        assert ece > 0.5

    def test_empty_returns_nan(self):
        ece = expected_calibration_error([], [])
        assert math.isnan(ece)

    def test_in_range(self):
        import random
        random.seed(42)
        y_true  = [random.randint(0, 1) for _ in range(100)]
        y_probs = [random.random() for _ in range(100)]
        ece = expected_calibration_error(y_true, y_probs)
        assert 0.0 <= ece <= 1.0


# ── evidence_citation_rate ─────────────────────────────────────────────────────

class TestEvidenceCitationRate:
    def test_all_cited(self):
        assert evidence_citation_rate([[1], [2, 3], [1]]) == 1.0

    def test_none_cited(self):
        assert evidence_citation_rate([[], [], []]) == 0.0

    def test_half_cited(self):
        rate = evidence_citation_rate([[1], [], [2], []])
        assert rate == 0.5

    def test_empty_input(self):
        assert evidence_citation_rate([]) == 0.0


# ── reasoning_completeness_score ───────────────────────────────────────────────

_FULL_REASONING = (
    "1. Visual observations: lungs clear.\n"
    "2. Clinical interpretation: no acute process.\n"
    "3. Evidence support: consistent with guideline.\n"
    "4. Justification: normal radiograph."
)

_PARTIAL_REASONING = "1. Visual observations: clear lungs."


class TestReasoningCompleteness:
    def test_all_complete(self):
        score = reasoning_completeness_score([_FULL_REASONING, _FULL_REASONING])
        assert score == 1.0

    def test_none_complete(self):
        score = reasoning_completeness_score([_PARTIAL_REASONING])
        assert score == 0.0

    def test_mixed(self):
        score = reasoning_completeness_score([_FULL_REASONING, _PARTIAL_REASONING])
        assert score == 0.5

    def test_empty_input(self):
        assert reasoning_completeness_score([]) == 0.0

    def test_case_insensitive(self):
        lower = _FULL_REASONING.lower()
        score = reasoning_completeness_score([lower])
        assert score == 1.0


# ── evidence_alignment_rate ────────────────────────────────────────────────────

class TestEvidenceAlignmentRate:
    def test_all_valid(self):
        rate = evidence_alignment_rate([[1], [2], [3]], retrieval_top_k=3)
        assert rate == 1.0

    def test_out_of_range(self):
        rate = evidence_alignment_rate([[5]], retrieval_top_k=3)
        assert rate == 0.0

    def test_empty_list_counts_as_aligned(self):
        rate = evidence_alignment_rate([[]], retrieval_top_k=3)
        assert rate == 1.0

    def test_mixed(self):
        rate = evidence_alignment_rate([[1], [99]], retrieval_top_k=3)
        assert rate == 0.5


# ── unparseable_rate ───────────────────────────────────────────────────────────

class TestUnparseableRate:
    def test_none_unparseable(self):
        assert unparseable_rate(["NORMAL", "ABNORMAL"]) == 0.0

    def test_all_unparseable(self):
        assert unparseable_rate(["UNPARSEABLE", "UNPARSEABLE"]) == 1.0

    def test_half_unparseable(self):
        assert unparseable_rate(["NORMAL", "UNPARSEABLE"]) == 0.5

    def test_empty(self):
        assert unparseable_rate([]) == 0.0


# ── latency_stats ──────────────────────────────────────────────────────────────

class TestLatencyStats:
    def test_single_value(self):
        stats = latency_stats([2.5])
        assert stats["mean_s"] == 2.5
        assert stats["p50_s"] == 2.5
        assert stats["p95_s"] == 2.5

    def test_sorted_percentiles(self):
        stats = latency_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["p50_s"] <= stats["p95_s"] <= stats["p99_s"]

    def test_mean_correct(self):
        stats = latency_stats([1.0, 3.0])
        assert abs(stats["mean_s"] - 2.0) < 0.01

    def test_empty(self):
        stats = latency_stats([])
        assert stats["mean_s"] == 0.0


# ── green_judge ────────────────────────────────────────────────────────────────

class TestGreenJudge:
    def test_perfect_composite(self):
        g = green_judge(1.0, 1.0, 1.0, 1.0, 1.0)
        assert g["composite"] == 1.0

    def test_zero_composite(self):
        g = green_judge(0.0, 0.0, 0.0, 0.0, 0.0)
        assert g["composite"] == 0.0

    def test_equal_weights(self):
        g = green_judge(1.0, 0.0, 0.0, 0.0, 0.0)
        assert abs(g["composite"] - 0.2) < 1e-6

    def test_keys_present(self):
        g = green_judge(0.9, 0.8, 0.95, 0.98, 0.87)
        expected_keys = {
            "G_groundedness", "R_reasoning", "E_alignment",
            "E_error_free", "N_numerical", "composite",
        }
        assert set(g.keys()) == expected_keys
