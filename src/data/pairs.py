"""Streaming (image, report) pairs for Stage-1 projector alignment training.

SRS §3/§4: streaming only, no image storage, num_workers=0.
SRS §7 Stage 1: "next-token prediction on MIMIC reports."

For Stage-2 balanced classification training, use balanced_stream.py instead.
This module is Stage-1 only: yields (image, report) without labels, applying
the full SRS §4 filter stack (filters.py) to skip device-only and ambiguous.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

from PIL import Image

from ..config import Config
from .filters import clean_mimic_text, is_usable

_NETWORK_ERRORS = (ConnectionError, OSError, TimeoutError, BrokenPipeError)


@dataclass
class Pair:
    image: Image.Image
    report: str
    source: str = "mimic-cxr"


def _is_usable_report(text: str) -> bool:
    """Full usability gate using the SRS §4 filter stack (filters.py)."""
    cleaned = clean_mimic_text(text)
    return is_usable(cleaned)


def stream_mimic_pairs(
    config: Config,
    split: str | None = None,
    max_pairs: int | None = None,
) -> Iterator[Pair]:
    """Yield Pair(image, report) from the configured MIMIC dataset via HF streaming.

    Uses config.mimic_dataset_repo and config.mimic_*_columns. If the repo
    doesn't exist or the columns don't match, raises with actionable guidance
    rather than silently yielding nothing.
    """
    from datasets import load_dataset

    repo = config.mimic_dataset_repo
    split = split or config.mimic_split
    try:
        ds = load_dataset(
            repo,
            split=split,
            streaming=True,
            token=config.hf_token or None,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not load MIMIC dataset '{repo}' (split='{split}'). "
            f"Underlying error: {type(e).__name__}: {e}\n"
            f"See retrieval.py's MimicReportsSource for the same fix instructions."
        ) from e

    text_cols = config.mimic_text_columns
    image_cols = config.mimic_image_columns

    count = 0
    schema_checked = False
    retry_delay = 30  # seconds between reconnect attempts

    while True:
        try:
            # Skip already-yielded samples on reconnect
            stream = ds.skip(count) if count > 0 else ds
            for example in stream:
                if not schema_checked:
                    available = list(example.keys())
                    if not any(c in example for c in text_cols):
                        raise RuntimeError(
                            f"No text column in {list(text_cols)} found in '{repo}'. "
                            f"Available: {available}"
                        )
                    if not any(c in example for c in image_cols):
                        raise RuntimeError(
                            f"No image column in {list(image_cols)} found in '{repo}'. "
                            f"Available: {available}"
                        )
                    schema_checked = True

                report = ""
                for col in text_cols:
                    val = example.get(col)
                    if val:
                        report = str(val).strip()
                        if report:
                            break
                if not _is_usable_report(report):
                    continue

                img = None
                for col in image_cols:
                    val = example.get(col)
                    if val is not None:
                        img = val
                        break
                if img is None or not isinstance(img, Image.Image):
                    continue

                yield Pair(image=img, report=report)
                count += 1
                if max_pairs is not None and count >= max_pairs:
                    return
            return  # exhausted dataset normally

        except _NETWORK_ERRORS as e:
            print(f"[stream_mimic] Network error at sample {count}: {type(e).__name__}: {e}")
            print(f"[stream_mimic] Reconnecting in {retry_delay}s (will skip {count} samples)...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 300)  # cap at 5 min
            # Reload the dataset handle for a fresh connection
            try:
                ds = load_dataset(repo, split=split, streaming=True, token=config.hf_token or None)
            except Exception:
                pass  # will retry on next loop iteration
