"""Experiment dataset streams — NIH, IU-Xray, PadChest, MIMIC eval.

Each function yields LabeledPair objects compatible with run_eval_stream().
Streams are lazy (HF streaming mode) — nothing is cached to disk.
"""
from __future__ import annotations

import io
from typing import Iterator

from PIL import Image

from .data.balanced_stream import LabeledPair


def _to_pil(raw) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(raw).convert("RGB")


# ── NIH ChestX-ray14 ──────────────────────────────────────────────────────────

def nih_stream(
    hf_token: str = "",
    max_samples: int = 100,
    split: str = "test",
) -> Iterator[LabeledPair]:
    """NIH ChestX-ray14: NORMAL = 'No Finding', else ABNORMAL."""
    from datasets import load_dataset

    _REPOS = [
        "alkzar90/NIH-Chest-X-ray-dataset",
        "Mayflower917/NIH-Chest-X-Ray-Dataset",
    ]
    ds = None
    last_err: Exception | None = None
    for repo in _REPOS:
        try:
            ds = load_dataset(repo, split=split, streaming=True, token=hf_token or None)
            print(f"[nih_stream] loaded '{repo}'")
            break
        except Exception as e:
            last_err = e

    if ds is None:
        raise RuntimeError(f"Could not load NIH ChestX-ray14. Last error: {last_err}")

    count = 0
    for ex in ds:
        if count >= max_samples:
            break
        try:
            img = _to_pil(ex.get("image") or ex.get("jpg") or ex.get("png"))
        except Exception:
            continue

        raw_labels = ex.get("Finding_Labels") or ex.get("labels") or ex.get("label") or ""
        if isinstance(raw_labels, list):
            raw_labels = "|".join(str(l) for l in raw_labels)
        label = "NORMAL" if "No Finding" in str(raw_labels) else "ABNORMAL"

        yield LabeledPair(
            image=img,
            report=str(ex.get("report", ex.get("findings", ""))),
            label=label,
            source="nih-cxr14",
        )
        count += 1


# ── IU-Xray NORMAL-only stream ────────────────────────────────────────────────

def iu_xray_normal_stream(max_samples: int = 50) -> Iterator[LabeledPair]:
    """Confirmed-NORMAL IU-Xray samples for adversarial sycophancy probing (Exp 4A).

    Filters conservatively: report must contain 'normal' AND contain none of the
    common pathology keywords — ensuring ground-truth NORMAL label is reliable.
    """
    from datasets import load_dataset

    _REPOS = ["projectnateryan/iu_xray", "Soobin-Kim/iu_xray", "openi/chest-xray"]
    _PATHOLOGY = {
        "pneumonia", "effusion", "cardiomegaly", "pneumothorax",
        "atelectasis", "consolidation", "opacity", "infiltrate",
    }
    ds = None
    last_err: Exception | None = None
    for repo in _REPOS:
        try:
            ds = load_dataset(repo, split="train", streaming=True)
            print(f"[iu_xray_normal_stream] loaded '{repo}'")
            break
        except Exception as e:
            last_err = e

    if ds is None:
        raise RuntimeError(f"Could not load IU-Xray. Last error: {last_err}")

    count = 0
    for ex in ds:
        if count >= max_samples:
            break

        report = str(ex.get("impression", ex.get("findings", ex.get("report", "")))).lower()
        if not report or "normal" not in report:
            continue
        if any(p in report for p in _PATHOLOGY):
            continue

        try:
            raw_img = ex.get("image") or ex.get("jpg")
            img = _to_pil(raw_img) if raw_img is not None else Image.new("RGB", (224, 224))
        except Exception:
            continue

        yield LabeledPair(image=img, report=report, label="NORMAL", source="iu-xray")
        count += 1


# ── PadChest OOD stream ───────────────────────────────────────────────────────

def padchest_stream(
    hf_token: str = "",
    max_samples: int = 100,
) -> Iterator[LabeledPair]:
    """PadChest (Hospital Universitario San Juan) for OOD evaluation (Exp 4B).

    Different hospital domain than MIMIC/IU-Xray — distribution-shift test.
    """
    from datasets import load_dataset

    _REPOS = [
        "agomberto/PadChest-multilabel",
        "keremberke/chest-xray-classification",
    ]
    ds = None
    last_err: Exception | None = None
    for repo in _REPOS:
        try:
            ds = load_dataset(repo, split="test", streaming=True, token=hf_token or None)
            print(f"[padchest_stream] loaded '{repo}'")
            break
        except Exception as e:
            last_err = e

    if ds is None:
        raise RuntimeError(f"Could not load PadChest. Last error: {last_err}")

    count = 0
    for ex in ds:
        if count >= max_samples:
            break
        try:
            img = _to_pil(ex.get("image") or ex.get("jpg") or ex.get("img"))
        except Exception:
            continue

        raw_labels = ex.get("Labels") or ex.get("labels") or ex.get("label") or "Normal"
        if isinstance(raw_labels, list):
            raw_labels = "|".join(str(l) for l in raw_labels)
        label = "NORMAL" if "normal" in str(raw_labels).lower() else "ABNORMAL"

        yield LabeledPair(
            image=img,
            report=str(ex.get("report", "")),
            label=label,
            source="padchest",
        )
        count += 1


# ── MIMIC-CXR evaluation stream ───────────────────────────────────────────────

def mimic_eval_stream(
    hf_token: str = "",
    max_samples: int = 50,
    split: str = "test",
) -> Iterator[LabeledPair]:
    """MIMIC-CXR samples with reference reports for RAG quality evaluation.

    Used in Exp 2, 3, 5, 6, 7 wherever BERTScore / CHAIR / GREEN require
    a reference report to compare against.
    """
    from datasets import load_dataset

    from .data.labeler import assign_label

    _TEXT_COLS = ("impression", "findings", "report")

    ds = load_dataset(
        "itsanmolgupta/mimic-cxr-dataset",
        split=split,
        streaming=True,
        token=hf_token or None,
    )

    count = 0
    for ex in ds:
        if count >= max_samples:
            break

        report = ""
        for col in _TEXT_COLS:
            val = ex.get(col, "")
            if val and isinstance(val, str):
                report = val.strip()
                break
        if not report:
            continue

        try:
            raw_img = ex.get("image") or ex.get("jpg")
            img = _to_pil(raw_img) if raw_img is not None else Image.new("RGB", (224, 224))
        except Exception:
            img = Image.new("RGB", (224, 224))

        label = assign_label(report) or "NORMAL"

        yield LabeledPair(image=img, report=report, label=label, source="mimic-cxr")
        count += 1
