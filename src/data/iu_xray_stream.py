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


def _iu_local_dir() -> str | None:
    """Directory of a locally-attached Kaggle IU-Xray dataset (raddar/chest-xrays-
    indiana-university layout: indiana_reports.csv + indiana_projections.csv +
    images/), or None. Set MEDDIAG_IU_XRAY_DIR to override. Avoids HF's Xet CDN."""
    import os, glob
    d = os.environ.get("MEDDIAG_IU_XRAY_DIR")
    if d and os.path.isdir(d):
        return d
    hits = glob.glob("/kaggle/input/**/indiana_reports.csv", recursive=True)
    return os.path.dirname(hits[0]) if hits else None


def _iu_local_stream(want_label: str, max_samples: int) -> Iterator[LabeledPair]:
    """Yield LabeledPairs from a local Kaggle IU-Xray (raddar) dataset. Label is
    derived from the report text with the SAME NORMAL/ABNORMAL filter as the HF
    stream, so the training signal is identical — only the source is local."""
    import os, glob, csv
    base = _iu_local_dir()
    if not base:
        return
    rpath = os.path.join(base, "indiana_reports.csv")
    ppath = os.path.join(base, "indiana_projections.csv")
    if not os.path.exists(rpath):
        return
    img_dirs = (glob.glob(os.path.join(base, "**", "images_normalized"), recursive=True)
                or glob.glob(os.path.join(base, "**", "images"), recursive=True))
    img_dir = img_dirs[0] if img_dirs else base
    uid2file: dict = {}
    if os.path.exists(ppath):
        with open(ppath, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "frontal" in str(row.get("projection", "")).lower():
                    uid2file.setdefault(row.get("uid"), row.get("filename"))
    print(f"[iu_xray] LOCAL raddar IU-Xray under {base} (images: {img_dir})")
    count = 0
    with open(rpath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if count >= max_samples:
                return
            fname = uid2file.get(row.get("uid"))
            if not fname:
                continue
            report = " ".join(str(row.get(c, "") or "") for c in ("findings", "impression", "MeSH")).strip()
            rl = report.lower()
            if len(rl) < 10:
                continue
            if want_label == "NORMAL":
                if "normal" not in rl or any(p in rl for p in _PATHOLOGY):
                    continue
            else:
                if not any(p in rl for p in _PATHOLOGY):
                    continue
            imgp = os.path.join(img_dir, str(fname))
            if not os.path.exists(imgp):
                continue
            try:
                img = Image.open(imgp).convert("RGB")
            except Exception:
                continue
            yield LabeledPair(image=img, report=report, label=want_label,
                              source=f"iu-xray-{want_label.lower()}-local")
            count += 1


def iu_xray_abnormal_training_stream(
    max_samples: int = 500,
) -> Iterator[LabeledPair]:
    """Yield confirmed-ABNORMAL LabeledPairs from IU-Xray for Stage 2 training.

    Filters conservatively — report must contain at least one pathology keyword.
    Supplements MIMIC ABNORMAL cases with examples from a different institution.
    Prefers a locally-attached Kaggle IU-Xray dataset (avoids HF's Xet CDN).
    """
    if _iu_local_dir():
        yield from _iu_local_stream("ABNORMAL", max_samples)
        return

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
    Prefers a locally-attached Kaggle IU-Xray dataset (avoids HF's Xet CDN); else
    falls back across multiple HuggingFace mirror repos.
    """
    if _iu_local_dir():
        yield from _iu_local_stream("NORMAL", max_samples)
        return

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
