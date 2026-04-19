"""Evaluation metrics for MEDDIAG binary classification.

All functions are pure — no model, no torch dependencies — so they are fully
testable without GPU. Inputs are plain Python lists or sequences.

Metric catalogue:
    binary_metrics              — accuracy, precision, recall, F1
    auroc_score                 — trapezoidal AUROC (no sklearn dependency)
    expected_calibration_error  — ECE for calibration analysis
    evidence_citation_rate      — fraction of outputs citing at least one snippet
    reasoning_completeness_score — fraction with all 4 CoT steps present
    evidence_alignment_rate     — fraction with valid evidence IDs (1 ≤ id ≤ top_k)
    unparseable_rate            — fraction returning UNPARSEABLE
    latency_stats               — mean, p50, p95, p99 latency
    green_judge                 — G/R/E/E/N multi-criteria composite score
"""
from __future__ import annotations

from typing import Sequence


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den > 0 else default


# ── Binary classification ──────────────────────────────────────────────────────

def binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """Accuracy, precision, recall, F1 for binary classification.

    Args:
        y_true: ground truth labels (0 = NORMAL, 1 = ABNORMAL)
        y_pred: predicted labels   (0 = NORMAL, 1 = ABNORMAL)
    """
    yt = list(y_true)
    yp = list(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true={len(yt)}, y_pred={len(yp)}")

    tp = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 1)
    tn = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 0)
    fn = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 0)

    accuracy  = _safe_div(tp + tn, tp + fp + tn + fn)
    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)
    f1        = _safe_div(2 * precision * recall, precision + recall)

    return {
        "accuracy":    round(accuracy,  4),
        "precision":   round(precision, 4),
        "recall":      round(recall,    4),
        "f1":          round(f1,        4),
        "support_pos": tp + fn,   # total actual positives
        "support_neg": tn + fp,   # total actual negatives
    }


def auroc_score(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    """AUROC via trapezoidal rule (no sklearn dependency).

    Args:
        y_true:   binary ground truth (0/1)
        y_scores: probability/score for the positive class (ABNORMAL)

    Returns:
        AUROC in [0, 1]. Returns 0.5 for degenerate single-class inputs.
    """
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    tp, fp = 0, 0
    roc: list[tuple[float, float]] = [(0.0, 0.0)]
    prev_score = None
    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            roc.append((fp / n_neg, tp / n_pos))
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    roc.append((fp / n_neg, tp / n_pos))
    roc.append((1.0, 1.0))

    auc = sum(
        (roc[i][0] - roc[i - 1][0]) * (roc[i][1] + roc[i - 1][1]) / 2
        for i in range(1, len(roc))
    )
    return round(float(auc), 4)


def expected_calibration_error(
    y_true: Sequence[int],
    y_probs: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Bins predictions by confidence; measures mean |bin_accuracy - bin_confidence|.
    Lower is better. Returns NaN for empty input.
    """
    n = len(y_true)
    if n == 0:
        return float("nan")

    step = 1.0 / n_bins
    ece = 0.0
    for b in range(n_bins):
        lo = b * step
        hi = lo + step
        is_last = b == n_bins - 1
        idxs = [
            i for i, p in enumerate(y_probs)
            if (lo <= p <= hi if is_last else lo <= p < hi)
        ]
        if not idxs:
            continue
        bin_acc  = sum(y_true[i] for i in idxs) / len(idxs)
        bin_conf = sum(y_probs[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / n) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


# ── Output quality ─────────────────────────────────────────────────────────────

def evidence_citation_rate(evidence_lists: Sequence[list[int]]) -> float:
    """Fraction of outputs that cite at least one evidence snippet."""
    if not evidence_lists:
        return 0.0
    cited = sum(1 for ev in evidence_lists if len(ev) > 0)
    return round(cited / len(evidence_lists), 4)


_REASONING_STEPS = [
    "Visual observations",
    "Clinical interpretation",
    "Evidence support",
    "Justification",
]


def reasoning_completeness_score(reasoning_texts: Sequence[str]) -> float:
    """Fraction of reasoning blocks that contain all 4 CoT steps."""
    if not reasoning_texts:
        return 0.0
    complete = sum(
        1 for text in reasoning_texts
        if all(step.lower() in text.lower() for step in _REASONING_STEPS)
    )
    return round(complete / len(reasoning_texts), 4)


def evidence_alignment_rate(
    evidence_lists: Sequence[list[int]],
    retrieval_top_k: int,
) -> float:
    """Fraction where all cited evidence IDs are in range [1, retrieval_top_k]."""
    if not evidence_lists:
        return 0.0
    aligned = sum(
        1 for ev in evidence_lists
        if not ev or all(1 <= e <= retrieval_top_k for e in ev)
    )
    return round(aligned / len(evidence_lists), 4)


def unparseable_rate(diagnoses: Sequence[str]) -> float:
    """Fraction of diagnoses that are UNPARSEABLE."""
    if not diagnoses:
        return 0.0
    bad = sum(1 for d in diagnoses if d == "UNPARSEABLE")
    return round(bad / len(diagnoses), 4)


# ── Latency ────────────────────────────────────────────────────────────────────

def latency_stats(latencies_s: Sequence[float]) -> dict[str, float]:
    """Mean, p50, p95, p99 latency in seconds over a list of measurements."""
    if not latencies_s:
        return {"mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "p99_s": 0.0}
    arr = sorted(latencies_s)
    n = len(arr)

    def pct(q: float) -> float:
        return arr[min(int(q * n), n - 1)]

    return {
        "mean_s": round(sum(arr) / n, 3),
        "p50_s":  round(pct(0.50), 3),
        "p95_s":  round(pct(0.95), 3),
        "p99_s":  round(pct(0.99), 3),
    }


# ── GREEN multi-judge ──────────────────────────────────────────────────────────

def green_judge(
    groundedness: float,       # G: evidence citation rate
    reasoning: float,          # R: reasoning completeness
    evidence_alignment: float, # E1: valid evidence IDs
    error_free: float,         # E2: 1 - unparseable_rate
    numerical: float,          # N: (auroc + f1) / 2
) -> dict[str, float]:
    """GREEN multi-criteria evaluation report card (equal weights).

    G — Groundedness:       Does the model cite retrieved evidence?
    R — Reasoning quality:  Are all 4 CoT steps present?
    E — Evidence alignment: Are cited IDs valid (1 ≤ id ≤ top_k)?
    E — Error-free rate:    1 - unparseable_rate
    N — Numerical accuracy: (AUROC + F1) / 2

    Returns per-criterion scores plus a composite (equal-weighted mean).
    """
    composite = (groundedness + reasoning + evidence_alignment + error_free + numerical) / 5
    return {
        "G_groundedness": round(groundedness,       4),
        "R_reasoning":    round(reasoning,          4),
        "E_alignment":    round(evidence_alignment, 4),
        "E_error_free":   round(error_free,         4),
        "N_numerical":    round(numerical,          4),
        "composite":      round(composite,          4),
    }
