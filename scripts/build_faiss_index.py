"""Build the FAISS index from MIMIC-CXR report snippets.

Usage:
    python -m scripts.build_faiss_index --max-snippets 50000

SRS §6: index is built once and loaded at inference time. Snippets live in
memory+on-disk as text (not images), so there's no conflict with the "no
dataset caching / no image storage" rules.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import CONFIG
from src.retrieval import MimicReportsSource, Retriever


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-snippets",
        type=int,
        default=20000,
        help="Cap total snippets ingested (streaming; no local caching of the dataset).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output dir for the index. Defaults to CONFIG.faiss_index_dir.",
    )
    args = parser.parse_args()

    # Building the FAISS index against a public dataset doesn't need an HF token.
    # The token check is deferred to LLM load time.
    CONFIG.validate(require_token=False)

    sources = [MimicReportsSource(CONFIG, max_snippets=args.max_snippets)]
    # NOTE: non-MIMIC sources (Radiopaedia, MedPix, guidelines) are stubbed in
    # retrieval.py. Add them here once their loaders are implemented.

    retriever = Retriever(CONFIG)
    print(
        f"[build_faiss_index] Dataset: {CONFIG.mimic_dataset_repo} "
        f"(split={CONFIG.mimic_split}, max_snippets={args.max_snippets})"
    )
    print(f"[build_faiss_index] Building index from {len(sources)} source(s)...")
    retriever.build(sources)
    out_dir = args.out_dir or CONFIG.faiss_index_dir
    retriever.save(out_dir)
    assert retriever.index is not None
    print(
        f"[build_faiss_index] Done. "
        f"{retriever.index.ntotal} snippets indexed -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())