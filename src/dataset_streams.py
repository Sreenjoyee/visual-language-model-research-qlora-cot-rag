"""Experiment dataset streams — NIH, IU-Xray, PadChest, MIMIC eval.

Each function yields LabeledPair objects compatible with run_eval_stream().
Streams are lazy (HF streaming mode) — nothing is cached to disk.
All streams include network-retry with exponential backoff and mid-stream
reconnect (skip already-yielded samples on reconnect).
"""
from __future__ import annotations

import io
import time
from typing import Iterator

from PIL import Image

from .data.balanced_stream import LabeledPair
from .data.labeler import assign_label

_NET_ERRORS = (ConnectionError, OSError, TimeoutError, BrokenPipeError)


def _to_pil(raw) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(raw).convert("RGB")


def _retry_load(repo: str, split: str, token: str = "", max_attempts: int = 5):
    """Load an HF streaming dataset with exponential-backoff retry."""
    from datasets import load_dataset
    delay = 15
    for attempt in range(max_attempts):
        try:
            return load_dataset(repo, split=split, streaming=True, token=token or None)
        except _NET_ERRORS as e:
            if attempt == max_attempts - 1:
                raise
            print(f"[stream] Network error loading '{repo}': {e}. Retry in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError(f"Failed to load '{repo}' after {max_attempts} attempts")


# ── NIH ChestX-ray14 ──────────────────────────────────────────────────────────

def nih_stream(
    hf_token: str = "",
    max_samples: int = 100,
    split: str = "train",
) -> Iterator[LabeledPair]:
    """NIH ChestX-ray14: NORMAL = 'No Finding', else ABNORMAL."""
    _REPOS = [
        ("BahaaEldin0/NIH-Chest-Xray-14", "train"),
        ("Sohaibsoussi/NIH-Chest-X-ray-dataset-small", "train"),
        ("ud-medical/nih-chest-x-ray-dataset", "train"),
        ("alkzar90/NIH-Chest-X-ray-dataset", "train"),
    ]
    ds = None
    loaded_repo = loaded_split = None
    last_err: Exception | None = None
    for repo, repo_split in _REPOS:
        try:
            ds = _retry_load(repo, repo_split, hf_token)
            loaded_repo, loaded_split = repo, repo_split
            print(f"[nih_stream] loaded '{repo}'")
            break
        except Exception as e:
            last_err = e

    if ds is None:
        raise RuntimeError(f"Could not load NIH ChestX-ray14. Last error: {last_err}")

    count = 0
    retry_delay = 30
    while True:
        try:
            stream = ds.skip(count) if count > 0 else ds
            for ex in stream:
                if count >= max_samples:
                    return
                try:
                    img = _to_pil(ex.get("image") or ex.get("jpg") or ex.get("png"))
                except Exception:
                    continue

                raw_labels = ex.get("Finding_Labels") or ex.get("label") or ex.get("labels") or ""
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
            return  # exhausted normally
        except _NET_ERRORS as e:
            print(f"[nih_stream] Network error at sample {count}: {e}. Reconnecting in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 300)
            try:
                ds = _retry_load(loaded_repo, loaded_split, hf_token)
            except Exception:
                pass


# ── IU-Xray NORMAL-only stream ────────────────────────────────────────────────

def iu_xray_normal_stream(max_samples: int = 50) -> Iterator[LabeledPair]:
    """Confirmed-NORMAL IU-Xray cases with REAL frontal images for the adversarial
    sycophancy probe (Exp 4A, paper §4.4). Normal = native Open-I label "normal"
    AND the negation-aware report labeler agree. See src/data/iu_xray_stream.py."""
    from .data.iu_xray_stream import iu_hf_labeled_stream
    n = 0
    for pair in iu_hf_labeled_stream("NORMAL", max_samples, "iu_xray_normal_stream", "iu-xray"):
        n += 1
        yield pair
    if n == 0:
        raise RuntimeError("No IU-Xray mirror with images is reachable — Exp 4A cannot run.")


# ── PadChest OOD stream ───────────────────────────────────────────────────────

def padchest_stream(
    hf_token: str = "",
    max_samples: int = 100,
) -> Iterator[LabeledPair]:
    """PadChest (Hospital Universitario San Juan) for OOD evaluation (Exp 4B)."""
    _REPOS = [
        ("JasonZZ0601/PadChest-GR-Xray", "train"),
        ("agomberto/PadChest-multilabel", "train"),
    ]
    ds = None
    loaded_repo = loaded_split = None
    last_err: Exception | None = None
    for repo, repo_split in _REPOS:
        try:
            ds = _retry_load(repo, repo_split, hf_token)
            loaded_repo, loaded_split = repo, repo_split
            print(f"[padchest_stream] loaded '{repo}'")
            break
        except Exception as e:
            last_err = e

    if ds is None:
        raise RuntimeError(f"Could not load PadChest. Last error: {last_err}")

    # Balanced sampling: collect into per-class queues, yield alternating NORMAL/ABNORMAL.
    from collections import deque
    queues: dict[str, deque] = {"NORMAL": deque(), "ABNORMAL": deque()}
    classes = ["NORMAL", "ABNORMAL"]
    class_idx = 0
    total_yielded = 0
    scanned = 0
    retry_delay = 30

    while True:
        try:
            stream = ds.skip(scanned) if scanned > 0 else ds
            for ex in stream:
                scanned += 1
                try:
                    img = _to_pil(ex.get("image") or ex.get("jpg") or ex.get("img"))
                except Exception:
                    continue
                raw_labels = ex.get("Labels") or ex.get("label") or ex.get("labels") or "Normal"
                if isinstance(raw_labels, list):
                    raw_labels = "|".join(str(l) for l in raw_labels)
                label = "NORMAL" if "normal" in str(raw_labels).lower() else "ABNORMAL"
                report = str(ex.get("sentence_en", ex.get("report", ex.get("findings", ""))))
                pair = LabeledPair(image=img, report=report, label=label, source="padchest")

                if len(queues[label]) < 64:
                    queues[label].append(pair)

                # Yield whenever both queues have at least one item
                while queues["NORMAL"] and queues["ABNORMAL"]:
                    yield queues[classes[class_idx]].popleft()
                    class_idx = 1 - class_idx
                    total_yielded += 1
                    if total_yielded >= max_samples:
                        imbalance = scanned - total_yielded
                        if imbalance > 0:
                            print(f"PadChest class imbalance: scanned {scanned}, yielded {total_yielded} balanced pairs.")
                        return
            # Stream exhausted — drain whichever class still has items
            for cls in classes:
                while queues[cls] and total_yielded < max_samples:
                    yield queues[cls].popleft()
                    total_yielded += 1
            return
        except _NET_ERRORS as e:
            print(f"[padchest_stream] Network error after {scanned} scanned: {e}. Reconnecting in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 300)
            try:
                ds = _retry_load(loaded_repo, loaded_split, hf_token)
            except Exception:
                pass


# ── MIMIC-CXR evaluation stream ───────────────────────────────────────────────

def mimic_eval_stream(
    hf_token: str = "",
    max_samples: int = 50,
    split: str = "train",
) -> Iterator[LabeledPair]:
    """MIMIC-CXR samples with reference reports for BERTScore / CHAIR / GREEN."""

    import os
    _REPO = os.environ.get("MEDDIAG_MIMIC_REPO", "power2004/mimic-cxr-dataset")
    _TEXT_COLS = ("impression", "findings", "report")

    from .config import CONFIG
    from .data.mimic_split import mimic_part
    _H = CONFIG.mimic_holdout_start   # evaluation uses the HELD-OUT partition only
    ds = mimic_part(_retry_load(_REPO, split, hf_token), "heldout", _H)

    count = 0
    seen = 0   # rows consumed (resume point); count = samples yielded
    retry_delay = 30
    while True:
        try:
            stream = ds.skip(seen) if seen > 0 else ds
            for ex in stream:
                seen += 1
                if count >= max_samples:
                    return
                report = ""
                for col in _TEXT_COLS:
                    val = ex.get(col, "")
                    if val and isinstance(val, str):
                        report = val.strip()
                        break
                if not report:
                    continue
                raw_img = ex.get("image") or ex.get("jpg")
                if raw_img is None:
                    continue
                try:
                    img = _to_pil(raw_img)
                except Exception:
                    continue            # never evaluate on a blank substitute image
                # Paper §3.10.1: MIMIC labels derive from report text. A report the
                # labeler can't classify is SKIPPED — it used to default to NORMAL,
                # silently injecting label noise into Exps 2, 3, 5-9.
                label = assign_label(report)
                if label is None:
                    continue
                yield LabeledPair(image=img, report=report, label=label, source="mimic-cxr")
                count += 1
            return
        except _NET_ERRORS as e:
            print(f"[mimic_eval_stream] Network error at sample {count}: {e}. Reconnecting in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 300)
            try:
                ds = mimic_part(_retry_load(_REPO, split, hf_token), "heldout", _H)
            except Exception:
                pass
