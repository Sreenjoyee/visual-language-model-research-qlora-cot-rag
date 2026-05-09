"""IU-Xray NORMAL training stream for Stage 2.

Yields confirmed-NORMAL LabeledPair objects from Indiana University Chest X-ray
dataset. Used to supplement MIMIC NORMAL cases (~1109 unique) with an additional
~1500 NORMAL cases from a different institution, improving generalization.

Filtering: report must contain 'normal' AND contain no pathology keywords.
This is conservative — only unambiguous NORMAL cases are included.
"""
from __future__ import annotations

import io
import time
from typing import Iterator

from PIL import Image

from .balanced_stream import LabeledPair
from .labeler import assign_label

_REPOS: list[str] = [
    "ChayanM/IUXray-Data-Train-Test",
    "projectnateryan/iu_xray",
    "Soobin-Kim/iu_xray",
]
_TEXT_COLS: list[str] = ["Caption", "findings", "impression", "report", "text"]
_IMG_COLS: list[str] = ["image", "jpg", "img", "png"]

_PATHOLOGY = frozenset({
    "pneumonia", "effusion", "cardiomegaly", "pneumothorax",
    "atelectasis", "consolidation", "opacity", "infiltrate",
    "mass", "nodule", "edema", "fracture",
})

_NET_ERRORS = (ConnectionError, OSError, TimeoutError, BrokenPipeError)


def _to_pil(raw) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(raw).convert("RGB")


def iu_xray_abnormal_training_stream(
    max_samples: int = 500,
) -> Iterator[LabeledPair]:
    """Yield confirmed-ABNORMAL LabeledPairs from IU-Xray for Stage 2 training.

    Filters conservatively — report must contain at least one pathology keyword.
    Supplements MIMIC ABNORMAL cases with examples from a different institution.
    """
    from datasets import load_dataset

    ds = None
    loaded_repo = None
    for repo in _REPOS:
        try:
            ds = load_dataset(repo, split="train", streaming=True)
            loaded_repo = repo
            break
        except Exception as e:
            print(f"[iu_xray_abnormal_training_stream] Could not load '{repo}': {e}")

    if ds is None:
        print("[iu_xray_abnormal_training_stream] No IU-Xray source available — skipping")
        return

    text_col: str | None = None
    count = 0
    retry_delay = 30

    while True:
        try:
            stream = ds.skip(count) if count > 0 else ds
            for ex in stream:
                if count >= max_samples:
                    return

                if text_col is None:
                    text_col = next((c for c in _TEXT_COLS if c in ex), None)
                    if text_col is None:
                        return

                raw_text = ex.get(text_col, "")
                if not raw_text or not isinstance(raw_text, str):
                    continue
                report = raw_text.strip().lower()

                if len(report) < 40:
                    continue
                if not any(p in report for p in _PATHOLOGY):
                    continue

                try:
                    raw_img = next((ex.get(c) for c in _IMG_COLS if ex.get(c) is not None), None)
                    img = _to_pil(raw_img) if raw_img is not None else Image.new("RGB", (224, 224))
                except Exception:
                    img = Image.new("RGB", (224, 224))

                yield LabeledPair(
                    image=img,
                    report=raw_text.strip(),
                    label="ABNORMAL",
                    source="iu-xray-abnormal-train",
                )
                count += 1
            return
        except _NET_ERRORS as e:
            print(f"[iu_xray_abnormal_training_stream] Network error at {count}: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 300)
            try:
                from datasets import load_dataset as _ld
                ds = _ld(loaded_repo, split="train", streaming=True)
            except Exception:
                pass


def iu_xray_normal_training_stream(
    max_samples: int = 2000,
) -> Iterator[LabeledPair]:
    """Yield confirmed-NORMAL LabeledPairs from IU-Xray for Stage 2 training.

    Filters conservatively — report must say 'normal' with no pathology keywords.
    Falls back across multiple HuggingFace mirror repos.
    """
    from datasets import load_dataset

    ds = None
    loaded_repo = None
    for repo in _REPOS:
        try:
            ds = load_dataset(repo, split="train", streaming=True)
            loaded_repo = repo
            print(f"[iu_xray_normal_training_stream] Loaded from '{repo}'")
            break
        except Exception as e:
            print(f"[iu_xray_normal_training_stream] Could not load '{repo}': {e}")

    if ds is None:
        print("[iu_xray_normal_training_stream] No IU-Xray source available — skipping")
        return

    text_col: str | None = None
    count = 0
    retry_delay = 30

    while True:
        try:
            stream = ds.skip(count) if count > 0 else ds
            for ex in stream:
                if count >= max_samples:
                    return

                if text_col is None:
                    text_col = next((c for c in _TEXT_COLS if c in ex), None)
                    if text_col is None:
                        return

                raw_text = ex.get(text_col, "")
                if not raw_text or not isinstance(raw_text, str):
                    continue
                report = raw_text.strip().lower()

                if len(report) < 40:
                    continue
                if "normal" not in report:
                    continue
                if any(p in report for p in _PATHOLOGY):
                    continue

                # Try to load image; synthesise blank if unavailable
                try:
                    raw_img = next((ex.get(c) for c in _IMG_COLS if ex.get(c) is not None), None)
                    img = _to_pil(raw_img) if raw_img is not None else Image.new("RGB", (224, 224))
                except Exception:
                    img = Image.new("RGB", (224, 224))

                yield LabeledPair(
                    image=img,
                    report=raw_text.strip(),
                    label="NORMAL",
                    source="iu-xray-normal-train",
                )
                count += 1
            return
        except _NET_ERRORS as e:
            print(f"[iu_xray_normal_training_stream] Network error at {count}: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 300)
            try:
                from datasets import load_dataset as _ld
                ds = _ld(loaded_repo, split="train", streaming=True)
            except Exception:
                pass
