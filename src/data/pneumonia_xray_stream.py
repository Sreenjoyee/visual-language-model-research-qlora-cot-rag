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


def _kermany_local_root() -> str | None:
    """Root of a locally-attached Kaggle Kermany dataset (folder layout with
    NORMAL/ + PNEUMONIA/ subdirs, e.g. paultimothymooney/chest-xray-pneumonia),
    or None. Set MEDDIAG_KERMANY_DIR to override. Avoids HF's Xet CDN (403s on Kaggle)."""
    import os, glob
    d = os.environ.get("MEDDIAG_KERMANY_DIR")
    if d and os.path.isdir(d):
        return d
    return "/kaggle/input" if glob.glob("/kaggle/input/**/PNEUMONIA", recursive=True) else None


def pneumonia_xray_stream(
    label: str = "NORMAL",
    max_samples: int = 1000,
) -> Iterator[LabeledPair]:
    """Yield LabeledPairs from the chest X-ray pneumonia dataset.

    Prefers a locally-attached Kaggle Kermany dataset (folder format) to avoid HF's
    Xet CDN; otherwise streams hf-vision/chest-xray-pneumonia. Same NORMAL/ABNORMAL
    labels either way.

    Args:
        label:       "NORMAL" or "ABNORMAL" (PNEUMONIA maps to ABNORMAL)
        max_samples: Maximum samples to yield
    """
    import os, glob
    _root = _kermany_local_root()
    if _root:
        _folder = "NORMAL" if label == "NORMAL" else "PNEUMONIA"
        _files = sorted(f for f in glob.glob(os.path.join(_root, "**", _folder, "*"), recursive=True)
                        if f.lower().endswith((".jpeg", ".jpg", ".png")))
        print(f"[pneumonia_xray_stream] LOCAL Kermany: {len(_files)} {label} files under {_root}")
        _count = 0
        for _f in _files:
            if _count >= max_samples:
                return
            try:
                _img = Image.open(_f).convert("RGB")
            except Exception:
                continue
            yield LabeledPair(image=_img, report=_FALLBACK_REPORT[label], label=label, source="pneumonia-xray-local")
            _count += 1
        return

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
