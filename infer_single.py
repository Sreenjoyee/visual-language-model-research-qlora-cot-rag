"""
Single X-ray inference demo — shows full CoT reasoning pipeline.
Run after training completes:
    python infer_single.py --image path/to/xray.jpg
"""
import argparse
import textwrap
from pathlib import Path

from PIL import Image


def run(image_path: str, projector: str, lora: str | None) -> None:
    from src.config import CONFIG
    from src.pipeline import MeddiagPipeline

    print("\n" + "=" * 60)
    print("  MEDDIAG — Single Image Inference")
    print("=" * 60)

    # ── Load pipeline ──────────────────────────────────────────
    print("\n[1/4] Loading pipeline (this takes ~20s)...")
    pipe = MeddiagPipeline(
        config=CONFIG,
        projector_weights=Path(projector),
        lora_adapter_dir=Path(lora) if lora and Path(lora).is_dir() else None,
    )
    print("      Pipeline ready.")

    # ── Load image ─────────────────────────────────────────────
    print(f"\n[2/4] Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    print(f"      Size: {w}×{h} px")

    # ── Run inference ──────────────────────────────────────────
    print("\n[3/4] Running inference (RAG + CoT generation)...")
    result = pipe.diagnose(image_path)
    print("      Done.")

    # ── Print results ──────────────────────────────────────────
    print("\n[4/4] Results")
    print("=" * 60)

    # Diagnosis banner
    colour = "ABNORMAL ⚠" if result.diagnosis == "ABNORMAL" else "NORMAL ✓"
    print(f"\n  DIAGNOSIS   : {colour}")
    if result.cls_confidence is not None:
        print(f"  P(ABNORMAL) : {result.cls_confidence:.3f}  (classification head)")

    # Retrieved evidence
    if result.retrieved:
        print("\n── Retrieved Evidence (RAG) " + "─" * 33)
        for snippet in result.retrieved[:6]:
            print(f"  [{result.retrieved.index(snippet) + 1}] {snippet.text[:120]}")

    # Full CoT response
    print("\n── Model Response (Chain-of-Thought) " + "─" * 23)
    wrapped = textwrap.fill(result.reasoning.strip(), width=70,
                            initial_indent="  ", subsequent_indent="  ")
    print(wrapped)

    print("\n" + "=" * 60 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="MEDDIAG single X-ray inference")
    ap.add_argument("--image",
                    required=True,
                    help="Path to chest X-ray image (jpg/png)")
    ap.add_argument("--projector",
                    default="models/projector_stage1.pt",
                    help="Path to trained projector weights")
    ap.add_argument("--lora",
                    default="models/lora_adapter",
                    help="Path to LoRA adapter directory (optional)")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"[ERROR] Image not found: {args.image}")
        raise SystemExit(1)
    if not Path(args.projector).exists():
        print(f"[ERROR] Projector not found: {args.projector}")
        print("        Run the full pipeline first: bash run_pipeline.sh")
        raise SystemExit(1)

    run(args.image, args.projector, args.lora)


if __name__ == "__main__":
    main()
