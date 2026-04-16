"""Run inference on a single chest X-ray.

Usage:
    python -m scripts.infer --image /c/meddiag/sample.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import CONFIG
from src.pipeline import MeddiagPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--projector-weights",
        type=Path,
        default=None,
        help="Optional path to trained projector weights (Stage 1 output). "
             "If omitted, projector uses random init — outputs will be garbage "
             "but the pipeline plumbing is testable.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 2

    pipeline = MeddiagPipeline(CONFIG, projector_weights=args.projector_weights)
    result = pipeline.diagnose(args.image)

    if args.json:
        print(json.dumps({
            "diagnosis": result.diagnosis,
            "evidence_used": result.evidence_used,
            "reasoning": result.reasoning,
            "retrieved": [
                {"text": r.text, "source": r.source, "distance": r.distance}
                for r in result.retrieved
            ],
            "raw_output": result.raw_output,
        }, indent=2))
    else:
        print(f"DIAGNOSIS: {result.diagnosis}")
        print(f"EVIDENCE USED: {result.evidence_used}")
        print("REASONING:")
        print(result.reasoning)
        print("\n--- Retrieved evidence ---")
        for i, r in enumerate(result.retrieved, start=1):
            print(f"[{i}] ({r.source}, d={r.distance:.3f}) {r.text[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())