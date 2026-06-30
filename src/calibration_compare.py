"""Post-hoc calibration-method comparison (#6).

Given probabilities + labels from an eval run, fit temperature / Platt / isotonic
calibrators on a held-out split and report ECE for each (plus uncalibrated). This
is the calibration ablation — it tells you which post-hoc method gives the lowest
ECE. Pure post-hoc; no model, no GPU. sklearn is optional (Platt/isotonic skip if
unavailable); temperature uses scipy.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

from .metrics import expected_calibration_error


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_temperature(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Scalar T minimising NLL on logit(probs). T>1 softens overconfidence."""
    from scipy.optimize import minimize_scalar

    logits = [_logit(p) for p in probs]

    def nll(T: float) -> float:
        T = max(float(T), 1e-2)
        s = 0.0
        for l, y in zip(logits, labels):
            p = min(max(_sigmoid(l / T), 1e-7), 1.0 - 1e-7)
            s += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
        return s / len(labels)

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def apply_temperature(probs: Sequence[float], T: float) -> list[float]:
    return [_sigmoid(_logit(p) / T) for p in probs]


def compare_calibration(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    test_frac: float = 0.5,
    seed: int = 0,
) -> dict:
    """ECE under {none, temperature, platt, isotonic}, fit on a held-out split.

    Calibrators are fit on (1 - test_frac) of the data and ECE is measured on the
    remaining test portion (avoids in-sample optimism). Returns a dict of ECE
    values (lower is better); methods needing sklearn are omitted if unavailable.
    """
    yt = list(y_true)
    yp = list(y_prob)
    n = len(yt)
    if n < 10:
        return {"error": "need >= 10 samples for a fit/eval split"}

    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = max(1, int(n * (1.0 - test_frac)))
    fit_i, test_i = idx[:cut], idx[cut:]
    if not test_i:
        test_i = fit_i  # tiny-n fallback
    yt_fit = [yt[i] for i in fit_i]
    yp_fit = [yp[i] for i in fit_i]
    yt_te = [yt[i] for i in test_i]
    yp_te = [yp[i] for i in test_i]

    out: dict = {"none": expected_calibration_error(yt_te, yp_te)}

    try:
        T = fit_temperature(yp_fit, yt_fit)
        out["temperature"] = expected_calibration_error(yt_te, apply_temperature(yp_te, T))
        out["temperature_T"] = round(T, 4)
    except Exception:
        pass

    try:
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        if len(set(yt_fit)) == 2:
            lr = LogisticRegression().fit(np.array(yp_fit).reshape(-1, 1), np.array(yt_fit))
            platt = lr.predict_proba(np.array(yp_te).reshape(-1, 1))[:, 1].tolist()
            out["platt"] = expected_calibration_error(yt_te, platt)

            iso = IsotonicRegression(out_of_bounds="clip").fit(yp_fit, yt_fit)
            isop = list(iso.predict(yp_te))
            out["isotonic"] = expected_calibration_error(yt_te, isop)
    except Exception:
        pass

    valid = {k: v for k, v in out.items() if k not in ("temperature_T",) and isinstance(v, (int, float))}
    out["best_method"] = min(valid, key=valid.get) if valid else "none"
    return out
