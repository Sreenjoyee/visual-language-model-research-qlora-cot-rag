"""Chest X-ray NORMAL/PNEUMONIA stream for Stage 2 training.

Yields LabeledPairs from hf-vision/chest-xray-pneumonia — 1341 NORMAL and
3875 PNEUMONIA images from the Kaggle chest X-ray dataset (Kermany et al.).
No report text available — report field is set to a placeholder so the FAISS
query falls back to a generic chest X-ray query.
"""
from __future__ import annotations

import io
import time
from typing import Iterator

from PIL import Image

from .balanced_stream import LabeledPair

_REPO = "hf-vision/chest-xray-pneumonia"
_NET_ERRORS = (ConnectionError, OSError, TimeoutError, BrokenPipeError)

_LABEL_MAP = {0: "NORMAL", 1: "ABNORMAL"}
_FALLBACK_REPORT = {
    "NORMAL":   "No acute cardiopulmonary abnormality. Lungs are clear.",
    "ABNORMAL": "Pulmonary infiltrate or consolidation identified.",
}


def _to_pil(raw) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(raw).convert("RGB")


def pneumonia_xray_stream(
    label: str = "NORMAL",
    max_samples: int = 1000,
) -> Iterator[LabeledPair]:
    """Yield LabeledPairs from the chest X-ray pneumonia dataset.

    Args:
        label:       "NORMAL" or "ABNORMAL" (PNEUMONIA maps to ABNORMAL)
        max_samples: Maximum samples to yield
    """
    from datasets import load_dataset

    target_int = 0 if label == "NORMAL" else 1

    ds = None
    for split in ["train", "test"]:
        try:
            ds = load_dataset(_REPO, split=split, streaming=True)
            break
        except Exception as e:
            print(f"[pneumonia_xray_stream] Could not load '{_REPO}' split '{split}': {e}")

    if ds is None:
        print(f"[pneumonia_xray_stream] Dataset unavailable — skipping {label} samples")
        return

    count = 0
    retry_delay = 30

    while True:
        try:
            for ex in ds:
                if count >= max_samples:
                    return
                if ex.get("label") != target_int:
                    continue
                try:
                    img = _to_pil(ex["image"])
                except Exception:
                    img = Image.new("RGB", (224, 224))

                yield LabeledPair(
                    image=img,
                    report=_FALLBACK_REPORT[label],
                    label=label,
                    source="pneumonia-xray",
                )
                count += 1
            return
        except _NET_ERRORS as e:
            print(f"[pneumonia_xray_stream] Network error at {count}: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 300)
            try:
                from datasets import load_dataset as _ld
                ds = _ld(_REPO, split="train", streaming=True)
            except Exception:
                pass
