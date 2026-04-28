"""Build the multi-source FAISS index.

Sources (SRS §6 — all four must be included):
  1. MIMIC-CXR report impressions (HuggingFace streaming)
  2. PubMed radiology abstracts (RadiopaediaSource — NCBI E-utilities, free)
  3. IU-Xray clinical cases (MedPixSource — HuggingFace)
  4. ACR/RSNA/WHO guideline snippets (GuidelinesSource — static, always added)

Usage:
    python -m scripts.build_faiss_index --max-mimic 20000

The MIMIC source requires HF_TOKEN + credentialed access. The other three
sources are fully public and will always run. To build a guidelines-only index
for smoke testing, pass --max-mimic 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import CONFIG
from src.retrieval import (
    EuropePMCSource,
    GuidelinesSource,
    HFPubMedQASource,
    MedPixSource,
    MimicReportsSource,
    RadiopaediaSource,
    Retriever,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build multi-source FAISS index")
    parser.add_argument(
        "--max-mimic",
        type=int,
        default=20000,
        help="Cap MIMIC-CXR snippets (0 to skip MIMIC entirely).",
    )
    parser.add_argument(
        "--max-pubmed",
        type=int,
        default=300,
        help="Cap PubMed radiology abstract snippets.",
    )
    parser.add_argument(
        "--max-iuxray",
        type=int,
        default=2000,
        help="Cap IU-Xray clinical case snippets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output dir for the index. Defaults to CONFIG.faiss_index_dir.",
    )
    parser.add_argument(
        "--skip-mimic",
        action="store_true",
        help="Skip MIMIC source (useful when HF_TOKEN is not available).",
    )
    args = parser.parse_args()

    # MIMIC requires token; other sources do not.
    require_token = not args.skip_mimic and args.max_mimic > 0
    CONFIG.validate(require_token=require_token)

    sources = []

    # Source 1: MIMIC-CXR (gated; skipped if --skip-mimic or --max-mimic 0)
    if not args.skip_mimic and args.max_mimic > 0:
        sources.append(MimicReportsSource(CONFIG, max_snippets=args.max_mimic))
        print(f"[build_faiss] Source 1: MIMIC-CXR (max {args.max_mimic} snippets)")
    else:
        print("[build_faiss] Source 1: MIMIC-CXR SKIPPED")

    # Source 2: PubMed via NCBI E-utilities (may fail on restricted networks)
    sources.append(RadiopaediaSource(max_snippets=args.max_pubmed))
    print(f"[build_faiss] Source 2: PubMed/NCBI radiology (max {args.max_pubmed} snippets)")

    # Source 3: PubMed via Europe PMC REST API (fallback — different endpoint,
    # no auth, accessible when NCBI is blocked or unreachable)
    sources.append(EuropePMCSource(max_snippets=args.max_pubmed))
    print(f"[build_faiss] Source 3: PubMed/EuropePMC radiology (max {args.max_pubmed} snippets)")

    # Source 4: PubMedQA radiology contexts via HuggingFace (no external API)
    sources.append(HFPubMedQASource(max_snippets=args.max_pubmed))
    print(f"[build_faiss] Source 4: PubMedQA/HuggingFace radiology (max {args.max_pubmed} snippets)")

    # Source 5: IU-Xray clinical cases (public HuggingFace dataset)
    sources.append(MedPixSource(max_snippets=args.max_iuxray))
    print(f"[build_faiss] Source 5: IU-Xray/MedPix (max {args.max_iuxray} snippets)")

    # Source 6: Static ACR/RSNA/WHO guidelines (always included, no network needed)
    sources.append(GuidelinesSource())
    print("[build_faiss] Source 6: ACR/RSNA/WHO guidelines (22 static snippets)")

    retriever = Retriever(CONFIG)
    out_dir = args.out_dir or CONFIG.faiss_index_dir
    print(f"[build_faiss] Building index from {len(sources)} source(s) -> {out_dir}")

    retriever.build(sources)
    retriever.save(out_dir)

    assert retriever.index is not None
    total = retriever.index.ntotal
    print(f"[build_faiss] Done. {total} snippets indexed -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
