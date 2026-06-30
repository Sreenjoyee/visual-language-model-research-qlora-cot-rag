"""Single-image inference through the full MEDDIAG pipeline.

Sends one image (default ``sample_xray.jpg``) through vision -> projector -> RAG
-> ClassificationHead + LLaMA reasoning via the real ``diagnose()`` entry point,
then prints and saves the diagnosis, P(ABNORMAL), reasoning chain, cited evidence
ids, and the retrieved RAG snippets. This is the qualitative sanity check — what
the model actually says on a single case.

Usage:
    python -m experiments.single_inference \\
        --image sample_xray.jpg \\
        --lora-adapter-dir models/lora_adapter_swa \\
        --output-dir reports/

Outputs (to --output-dir):
    single_inference.json   structured record
    single_inference.md     human-readable result
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import CONFIG
from src.pipeline import MeddiagPipeline


def _write_md(rec: dict, path: Path) -> None:
    pa = f"{rec['p_abnormal']:.3f}" if rec["p_abnormal"] is not None else "N/A (text-parse fallback)"
    lines = [
        "# Single-Inference Result", "",
        f"- **Image:** `{rec['image']}`",
        f"- **Prediction:** {rec['prediction']}",
        f"- **P(ABNORMAL):** {pa}",
        f"- **Evidence cited:** {rec['evidence_cited']}",
        "", "## Reasoning", "", rec["reasoning"] or "_(none)_",
        "", "## Retrieved snippets", "",
    ]
    if rec["retrieved_snippets"]:
        lines += [f"{i}. {s}" for i, s in enumerate(rec["retrieved_snippets"], 1)]
    else:
        lines.append("_(none)_")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(pipeline: MeddiagPipeline, image_path: Path, out_dir: Path | None) -> dict:
    res = pipeline.diagnose(str(image_path))
    pa = f"{res.cls_confidence:.3f}" if res.cls_confidence is not None else "N/A (text-parse fallback)"

    print("\n" + "=" * 66)
    print(f"  SINGLE INFERENCE — {image_path}")
    print("-" * 66)
    print(f"  Prediction  : {res.diagnosis}")
    print(f"  P(ABNORMAL) : {pa}")
    print(f"  Evidence    : {res.evidence_used}")
    print(f"  Reasoning   :\n{res.reasoning}")
    print("  Retrieved snippets:")
    for i, r in enumerate(res.retrieved, 1):
        print(f"    {i}. {r.text[:200]}")
    print("=" * 66)

    rec = {
        "image": str(image_path),
        "prediction": res.diagnosis,
        "p_abnormal": res.cls_confidence,
        "evidence_cited": res.evidence_used,
        "reasoning": res.reasoning,
        "retrieved_snippets": [r.text[:300] for r in res.retrieved],
        "raw_output": (res.raw_output or "")[:2000],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "single_inference.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        _write_md(rec, out_dir / "single_inference.md")
        print(f"[single-inference] saved -> {out_dir}/single_inference.{{json,md}}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Single-image inference through the MEDDIAG pipeline")
    ap.add_argument("--image", type=Path, default=Path("sample_xray.jpg"),
                    help="Image to diagnose (default: sample_xray.jpg).")
    ap.add_argument("--projector-path", type=Path, default=CONFIG.models_dir / "projector_stage1.pt")
    ap.add_argument("--lora-adapter-dir", type=Path, default=None,
                    help="Optional Stage-2 LoRA adapter directory.")
    ap.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    if not args.image.exists():
        print(f"[single-inference] image not found: {args.image} — skipping.")
        return 1

    CONFIG.validate(require_token=True)
    print("[single-inference] loading pipeline...")
    pipeline = MeddiagPipeline(
        config=CONFIG,
        projector_weights=args.projector_path,
        lora_adapter_dir=args.lora_adapter_dir,
    )
    run(pipeline, args.image, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
