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

# HuggingFace IU-Xray (Indiana / Open-I) mirrors WITH IMAGES, in preference order.
# (repo, split, image column, text columns, native-normal column or None)
# NOTE: the previous mirror, ChayanM/IUXray-Data-Train-Test, has NO image column
# (only Image_Name + Caption) — the old code silently used a blank 224x224 image,
# so IU training samples and the Exp 4A probe were run on empty images.
_SOURCES: list[tuple[str, str, str, tuple[str, ...], str | None]] = [
    # official Open-I release: frontal view + findings/impression + native labels
    ("ykumards/open-i", "train", "img_frontal", ("findings", "impression"), "Problems"),
    ("MLforHealthcare/Indiana_University_Chest_X-ray_Collection", "train", "image", ("report",), None),
    ("simwit/iu-xray", "test", "image", ("findings", "impression"), None),
]

_NET_ERRORS = (ConnectionError, OSError, TimeoutError, BrokenPipeError)


def _to_pil(raw) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict) and raw.get("bytes"):
        raw = raw["bytes"]
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
            if assign_label(report) != want_label:   # same negation-aware labeler as MIMIC
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


def iu_hf_labeled_stream(want_label: str, max_samples: int, tag: str,
                         source: str | None = None) -> Iterator[LabeledPair]:
    """IU-Xray from HuggingFace with REAL images, labelled by the project's
    negation-aware report labeler (src/data/labeler.assign_label — same as MIMIC).

    Where the mirror carries the native Open-I label (Problems == "normal"), it
    must AGREE with the report label ("confirmed" normal/abnormal, paper §3.10.1).
    Rows without a decodable image are skipped — never a blank substitute.
    """
    from datasets import load_dataset

    for repo, split, img_col, text_cols, native_col in _SOURCES:
        try:
            ds = load_dataset(repo, split=split, streaming=True)
        except Exception as e:
            print(f"[{tag}] could not load '{repo}': {type(e).__name__}: {e}")
            continue
        print(f"[{tag}] loaded '{repo}' ({split})")
        count, seen, retry_delay = 0, 0, 30
        while True:
            try:
                stream = ds.skip(seen) if seen > 0 else ds
                for ex in stream:
                    seen += 1
                    if count >= max_samples:
                        return
                    report = " ".join(str(ex.get(c) or "").strip() for c in text_cols).strip()
                    if len(report) < 20 or assign_label(report) != want_label:
                        continue
                    if native_col is not None:
                        native_normal = str(ex.get(native_col) or "").strip().lower() == "normal"
                        if native_normal != (want_label == "NORMAL"):
                            continue
                    raw = ex.get(img_col)
                    if raw is None:
                        continue
                    try:
                        img = _to_pil(raw)
                    except Exception:
                        continue
                    yield LabeledPair(image=img, report=report, label=want_label,
                                      source=source or f"iu-xray-{want_label.lower()}")
                    count += 1
                if count > 0:
                    return
                print(f"[{tag}] '{repo}' gave no usable {want_label} rows — trying next mirror")
                break
            except _NET_ERRORS as e:
                print(f"[{tag}] network error at row {seen}: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 300)
                try:
                    ds = load_dataset(repo, split=split, streaming=True)
                except Exception:
                    pass
    print(f"[{tag}] no IU-Xray mirror with images is available")


def iu_xray_abnormal_training_stream(max_samples: int = 500) -> Iterator[LabeledPair]:
    """ABNORMAL IU-Xray cases for Stage-2 training (report-labelled, negation-aware).
    Uses a locally-attached Kaggle copy only if one is present (fallback)."""
    if _iu_local_dir():
        yield from _iu_local_stream("ABNORMAL", max_samples)
        return
    yield from iu_hf_labeled_stream("ABNORMAL", max_samples, "iu_xray_abnormal_training_stream", "iu-xray-abnormal-train")


def iu_xray_normal_training_stream(max_samples: int = 2000) -> Iterator[LabeledPair]:
    """NORMAL IU-Xray cases for Stage-2 training (report-labelled, negation-aware).
    Uses a locally-attached Kaggle copy only if one is present (fallback)."""
    if _iu_local_dir():
        yield from _iu_local_stream("NORMAL", max_samples)
        return
    yield from iu_hf_labeled_stream("NORMAL", max_samples, "iu_xray_normal_training_stream", "iu-xray-normal-train")
