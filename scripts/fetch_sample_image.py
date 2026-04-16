"""Fetch a sample chest X-ray from the configured MIMIC dataset.

Streams the dataset (no local dataset caching) and writes one image to disk
for use with `scripts.infer`. This is a test/development utility, not part of
the production pipeline — but per SRS §4 ("no image storage") we intentionally
only save ONE image, not a batch, and the script makes that obvious.

Usage:
    # Default: save a random image to ./sample_xray.jpg
    python -m scripts.fetch_sample_image

    # Pick a specific row index (deterministic), custom output path:
    python -m scripts.fetch_sample_image --index 42 --out ./xray.jpg

    # Overwrite existing file:
    python -m scripts.fetch_sample_image --force

    # Print the associated report too (useful for sanity-checking):
    python -m scripts.fetch_sample_image --print-report
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image

from src.config import CONFIG


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("./sample_xray.jpg"),
        help="Output image path. Default: ./sample_xray.jpg",
    )
    ap.add_argument(
        "--index",
        type=int,
        default=None,
        help="Row index to fetch (deterministic). If omitted, pick a random "
             "index in [0, 1000) so repeated runs give you different images.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --out if it exists.",
    )
    ap.add_argument(
        "--print-report",
        action="store_true",
        help="Print the associated findings/impression text.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for index selection (only used when --index is not given).",
    )
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(
            f"Output already exists: {args.out}\n"
            f"Use --force to overwrite, or pass a different --out path.",
            file=sys.stderr,
        )
        return 1

    # No HF_TOKEN needed for the default public mirror.
    CONFIG.validate(require_token=False)

    # Decide target index
    if args.index is not None:
        target_index = args.index
        if target_index < 0:
            print(f"--index must be >= 0, got {target_index}", file=sys.stderr)
            return 2
    else:
        if args.seed is not None:
            random.seed(args.seed)
        # 1000 is a balance: big enough to feel random, small enough to iterate
        # fewer rows than that. The full dataset has ~30k rows.
        target_index = random.randint(0, 999)

    print(
        f"[fetch_sample] Dataset: {CONFIG.mimic_dataset_repo} "
        f"(split={CONFIG.mimic_split}, target_index={target_index})"
    )

    # Stream — reuse the same column-detection logic as pairs.py, inlined here
    # because we don't need the Pair dataclass and the usability filter would
    # just make some indices silently unavailable (confusing for a demo).
    from datasets import load_dataset

    try:
        ds = load_dataset(
            CONFIG.mimic_dataset_repo,
            split=CONFIG.mimic_split,
            streaming=True,
            token=CONFIG.hf_token or None,
        )
    except Exception as e:
        print(
            f"[fetch_sample] Could not load '{CONFIG.mimic_dataset_repo}': "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 3

    text_cols = CONFIG.mimic_text_columns
    image_cols = CONFIG.mimic_image_columns

    selected_image: Image.Image | None = None
    selected_report: str = ""
    schema_checked = False

    # Iterate until we reach target_index. Streaming means this takes O(target_index)
    # time and bandwidth — keep target_index modest.
    for i, example in enumerate(ds):
        if not schema_checked:
            available = list(example.keys())
            if not any(c in example for c in image_cols):
                print(
                    f"[fetch_sample] No image column in {list(image_cols)}. "
                    f"Available: {available}",
                    file=sys.stderr,
                )
                return 4
            schema_checked = True

        if i != target_index:
            continue

        # Found the row we want.
        for col in image_cols:
            val = example.get(col)
            if isinstance(val, Image.Image):
                selected_image = val
                break

        for col in text_cols:
            val = example.get(col)
            if val:
                selected_report = str(val).strip()
                break

        break  # don't keep streaming

    if selected_image is None:
        print(
            f"[fetch_sample] Row {target_index} had no decodable image. "
            f"Try a different --index.",
            file=sys.stderr,
        )
        return 5

    # Convert to RGB before saving as JPG (grayscale X-rays fail JPEG encode in 'L'
    # mode with some Pillow versions; RGB is universally safe).
    if selected_image.mode != "RGB":
        selected_image = selected_image.convert("RGB")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    selected_image.save(args.out, format="JPEG", quality=95)
    size_kb = args.out.stat().st_size / 1024
    print(
        f"[fetch_sample] Saved image: {args.out} "
        f"({selected_image.size[0]}x{selected_image.size[1]}, {size_kb:.0f} KB)"
    )

    if args.print_report and selected_report:
        print("\n[fetch_sample] Associated report text:")
        print("-" * 60)
        print(selected_report)
        print("-" * 60)

    print(
        f"\n[fetch_sample] Next step:\n"
        f"    python -m scripts.infer --image {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())