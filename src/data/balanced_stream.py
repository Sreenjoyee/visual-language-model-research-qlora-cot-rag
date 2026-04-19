"""Balanced NORMAL/ABNORMAL streaming sampler — SRS §4, §13.

SRS rules:
  - Balanced sampling required
  - No ordering bias
  - Streaming-only ingestion (no caching)
  - dataloader_num_workers = 0 (Windows, single-threaded)

Design:
  Maintains two in-memory queues (one per class), filled from the stream.
  Yields samples in strict NORMAL→ABNORMAL→NORMAL→... alternation to prevent
  ordering bias. When one queue drains, it refills from the stream rather than
  stopping — this handles natural MIMIC imbalance transparently.

  If the stream is exhausted before a queue can be refilled, the function
  stops cleanly. An imbalance warning is emitted if one class supplies
  significantly more samples than the other.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Iterator

from PIL import Image

from ..config import Config
from .filters import clean_mimic_text, is_usable
from .labeler import assign_label
from .mimic_stream import retrying_iter

logger = logging.getLogger(__name__)


@dataclass
class LabeledPair:
    image: Image.Image
    report: str
    label: str          # "NORMAL" or "ABNORMAL"
    source: str = "mimic-cxr"


def _raw_mimic_stream(config: Config, split: str) -> Iterator[dict]:
    """Yield raw HF dataset examples, with retry on disconnect."""
    from datasets import load_dataset

    def _factory():
        return iter(
            load_dataset(
                config.mimic_dataset_repo,
                split=split,
                streaming=True,
                token=config.hf_token or None,
            )
        )

    yield from retrying_iter(_factory)


def _extract_report(example: dict, text_cols: tuple[str, ...]) -> str:
    for col in text_cols:
        val = example.get(col)
        if val:
            text = str(val).strip()
            if text:
                return text
    return ""


def _extract_image(example: dict, image_cols: tuple[str, ...]) -> Image.Image | None:
    for col in image_cols:
        val = example.get(col)
        if val is not None and isinstance(val, Image.Image):
            return val
    return None


def balanced_mimic_stream(
    config: Config,
    split: str | None = None,
    max_pairs: int | None = None,
    buffer_per_class: int = 128,
    imbalance_warn_threshold: float = 3.0,
) -> Iterator[LabeledPair]:
    """Stream balanced (NORMAL, ABNORMAL) LabeledPairs from MIMIC-CXR.

    Args:
        config: Pipeline config.
        split: Dataset split (defaults to config.mimic_split).
        max_pairs: Stop after yielding this many pairs (None = run to exhaustion).
        buffer_per_class: Max queued samples per class before back-pressure.
        imbalance_warn_threshold: Warn if skipped:yielded ratio exceeds this.

    Yields:
        LabeledPair — alternating NORMAL / ABNORMAL with strict 1:1 balance.
    """
    split = split or config.mimic_split
    text_cols = config.mimic_text_columns
    image_cols = config.mimic_image_columns

    queues: dict[str, deque] = {
        "NORMAL": deque(),
        "ABNORMAL": deque(),
    }
    classes = ["NORMAL", "ABNORMAL"]
    class_idx = 0  # alternating pointer

    # Counters for imbalance monitoring
    seen = {"NORMAL": 0, "ABNORMAL": 0, "skipped": 0}

    stream = _raw_mimic_stream(config, split)
    schema_checked = False
    total_yielded = 0

    def _fill_queues_from_stream(needed_class: str) -> bool:
        """Pull from stream until needed_class has at least one item, or stream ends."""
        nonlocal schema_checked
        for example in stream:
            if not schema_checked:
                available = list(example.keys())
                if not any(c in example for c in text_cols):
                    raise RuntimeError(
                        f"No text column in {list(text_cols)} found. Available: {available}"
                    )
                if not any(c in example for c in image_cols):
                    raise RuntimeError(
                        f"No image column in {list(image_cols)} found. Available: {available}"
                    )
                schema_checked = True

            raw_text = _extract_report(example, text_cols)
            if not raw_text:
                seen["skipped"] += 1
                continue

            cleaned = clean_mimic_text(raw_text)
            if not is_usable(cleaned):
                seen["skipped"] += 1
                continue

            label = assign_label(cleaned)
            if label is None:
                seen["skipped"] += 1
                continue

            img = _extract_image(example, image_cols)
            if img is None:
                seen["skipped"] += 1
                continue

            seen[label] += 1

            # Only queue if below buffer limit to avoid unbounded memory
            if len(queues[label]) < buffer_per_class:
                queues[label].append(LabeledPair(image=img, report=cleaned, label=label))

            if queues[needed_class]:
                return True  # the class we need now has at least one item

        return False  # stream exhausted

    while True:
        needed = classes[class_idx]

        # If the needed queue is empty, try to fill it
        if not queues[needed]:
            stream_has_more = _fill_queues_from_stream(needed)
            if not stream_has_more and not queues[needed]:
                break  # stream exhausted, can't fill needed class

        if not queues[needed]:
            break

        pair = queues[needed].popleft()
        yield pair
        total_yielded += 1
        class_idx = (class_idx + 1) % 2

        if max_pairs is not None and total_yielded >= max_pairs:
            break

    # Post-stream imbalance diagnostics
    n = seen["NORMAL"]
    a = seen["ABNORMAL"]
    s = seen["skipped"]
    total_seen = n + a + s

    if total_seen > 0:
        skip_rate = s / total_seen
        if skip_rate > 0.7:
            logger.warning(
                "Balanced sampler: %.0f%% of MIMIC examples were skipped. "
                "Filters may be too aggressive.", skip_rate * 100
            )

    if n > 0 and a > 0:
        ratio = max(n, a) / min(n, a)
        if ratio > imbalance_warn_threshold:
            dominant = "NORMAL" if n > a else "ABNORMAL"
            logger.warning(
                "MIMIC class imbalance: %s:%s = %.1f:1 (dominant=%s). "
                "Balanced sampler discarded excess %s samples.",
                n, a, ratio, dominant, dominant
            )

    logger.info(
        "Balanced stream complete: yielded=%d NORMAL=%d ABNORMAL=%d skipped=%d",
        total_yielded, n, a, s,
    )


def check_label_distribution(pairs: list[LabeledPair], min_ratio: float = 0.8) -> None:
    """Assert that a collected batch is balanced before training starts.

    SRS §19.2 module-6: 'Label distribution check before training.'
    Raises RuntimeError if the batch is too skewed.

    Args:
        pairs: List of LabeledPair objects.
        min_ratio: Minimum minority/majority ratio to pass (default 0.8 = 80%).
    """
    if not pairs:
        raise RuntimeError("Empty batch passed to check_label_distribution.")

    counts: dict[str, int] = {}
    for p in pairs:
        counts[p.label] = counts.get(p.label, 0) + 1

    if len(counts) < 2:
        raise RuntimeError(
            f"Only one class present in batch: {counts}. "
            "Training requires both NORMAL and ABNORMAL samples."
        )

    minority = min(counts.values())
    majority = max(counts.values())
    ratio = minority / majority

    if ratio < min_ratio:
        raise RuntimeError(
            f"Batch is too imbalanced: {counts} (ratio={ratio:.2f} < {min_ratio}). "
            "Fix balanced_mimic_stream or increase buffer_per_class."
        )
