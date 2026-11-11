"""Deterministic train / held-out partition of the MIMIC-CXR mirror.

The HF mirror (power2004/mimic-cxr-dataset) ships ONE split ("train", 30,633
rows). Without a partition, training (Stage-1, Stage-2, the FAISS knowledge
base) and evaluation (evaluate.py, Exps 2/3/5-9) all stream from row 0 — i.e.
the model is evaluated on images it was trained on, and RAG can retrieve the
test image's own report as "evidence". The paper (§4.1) reports a HELD-OUT
in-distribution split, so:

    rows [0, MIMIC_HOLDOUT_START)  -> "train"    (training + knowledge base)
    rows [MIMIC_HOLDOUT_START, …)  -> "heldout"  (evaluation only)

Streaming order of the parquet shards is deterministic, so the partition is
stable across sessions. Default boundary 27000 holds out ~3.6k cases.
"""
from __future__ import annotations

PARTS = ("train", "heldout")


def mimic_part(ds, part: str, holdout_start: int):
    """Restrict a streaming MIMIC dataset to one partition."""
    if part == "train":
        return ds.take(holdout_start)
    if part == "heldout":
        return ds.skip(holdout_start)
    raise ValueError(f"MIMIC partition must be one of {PARTS}, got {part!r}")
