"""Single-image qualitative inference demo.

Runs the full pipeline (vision -> projector -> RAG -> ClassificationHead + LLaMA
reasoning) on a handful of balanced MIMIC samples via the real ``diagnose()``
entry point, and saves the per-sample qualitative output: predicted diagnosis,
P(ABNORMAL), the reasoning chain, cited evidence ids, and the retrieved RAG
snippets — alongside the ground-truth label and report.

This is the qualitative counterpart to the aggregate metrics: it shows *what the
model actually says* on individual cases (useful as paper case-study figures).

Outputs (to --output-dir, default reports/):
  single_inference.json          structured results, one record per sample
  single_inference.md            human-readable report with reasoning chains
  single_inference_examples.png  figure: image + diagnosis panel per sample
"""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.config import CONFIG
from src.data.balanced_stream import balanced_mimic_stream
from src.pipeline import MeddiagPipeline


def _write_markdown(records: list[dict], path: Path) -> None:
    """Render the qualitative records as a readable markdown report."""
    lines = ["# Single-Inference Qualitative Results", ""]
    for r in records:
        pa = f"{r['p_abnormal']:.3f}" if r["p_abnormal"] is not None else "n/a"
        mark = "correct" if r["correct"] else "WRONG"
        lines += [
            f"## Sample {r['index']} — {r['source']}",
            f"- **Ground truth:** {r['true_label']}",
            f"- **Predicted:** {r['predicted_diagnosis']}  "
            f"(P(ABNORMAL)={pa}) — {mark}",
            f"- **Evidence cited:** {r['evidence_cited']}",
            "",
            "**Reasoning:**",
            "",
            r["reasoning"] or "_(none)_",
            "",
            "**Retrieved snippets:**",
        ]
        if r["retrieved_snippets"]:
            lines += [f"  {j}. {s}" for j, s in enumerate(r["retrieved_snippets"], 1)]
        else:
            lines.append("  _(none — RAG returned nothing)_")
        lines += [
            "",
            "**Ground-truth report:**",
            "",
            "> " + (r["ground_truth_report"] or "_(none)_").replace("\n", "\n> "),
            "",
            "---",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_figure(items: list[tuple], path: Path) -> None:
    """Grid of the sampled images, each titled with true/pred labels + P(abn)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(items)
    if n == 0:
        return
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.3))
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for ax, (img_path, r) in zip(axes, items):
        ax.imshow(Image.open(img_path), cmap="gray")
        pa = f"{r['p_abnormal']:.2f}" if r["p_abnormal"] is not None else "n/a"
        color = "green" if r["correct"] else "red"
        ax.set_title(
            f"true={r['true_label']} | pred={r['predicted_diagnosis']}\nP(abn)={pa}",
            color=color, fontsize=10,
        )
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Single-inference examples", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_single_inference(pipeline: MeddiagPipeline, n_samples: int, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = balanced_mimic_stream(CONFIG, split="train", max_pairs=n_samples)
    tmpdir = Path(tempfile.mkdtemp(prefix="single_inf_"))
    records: list[dict] = []
    fig_items: list[tuple] = []

    for i, pair in enumerate(stream):
        if i >= n_samples:
            break
        img_path = tmpdir / f"sample_{i}.png"
        pair.image.convert("RGB").save(img_path)
        try:
            res = pipeline.diagnose(img_path)
        except Exception as e:  # noqa: BLE001 — demo must not abort the eval run
            print(f"  [{i}] inference failed: {type(e).__name__}: {e}")
            continue
        rec = {
            "index": i,
            "source": pair.source,
            "true_label": pair.label,
            "ground_truth_report": (pair.report or "")[:1000],
            "predicted_diagnosis": res.diagnosis,
            "p_abnormal": res.cls_confidence,
            "correct": (res.diagnosis == pair.label),
            "reasoning": res.reasoning,
            "evidence_cited": res.evidence_used,
            "retrieved_snippets": [r.text[:300] for r in res.retrieved],
            "raw_output": (res.raw_output or "")[:2000],
        }
        records.append(rec)
        fig_items.append((img_path, rec))
        pa = f"{res.cls_confidence:.3f}" if res.cls_confidence is not None else "n/a"
        tick = "OK" if rec["correct"] else "XX"
        print(f"  [{i}] {tick} true={pair.label:<8} pred={res.diagnosis:<10} P(abn)={pa}")

    (out_dir / "single_inference.json").write_text(
        json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(),
                    "n": len(records), "samples": records}, indent=2),
        encoding="utf-8",
    )
    _write_markdown(records, out_dir / "single_inference.md")
    try:
        _save_figure(fig_items, out_dir / "single_inference_examples.png")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] figure render failed: {type(e).__name__}: {e}")
    print(f"\n[single-inference] {len(records)} samples -> "
          f"{out_dir}/single_inference.{{json,md,png}}")
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="MEDDIAG single-image qualitative inference demo")
    ap.add_argument("--projector-path", type=Path, default=CONFIG.models_dir / "projector_stage1.pt")
    ap.add_argument("--lora-adapter-dir", type=Path, default=None,
                    help="Optional Stage-2 LoRA adapter directory.")
    ap.add_argument("--n-samples", type=int, default=6,
                    help="Number of balanced samples to run (half NORMAL, half ABNORMAL).")
    ap.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    CONFIG.validate(require_token=True)
    print("[single-inference] loading pipeline...")
    pipeline = MeddiagPipeline(
        config=CONFIG,
        projector_weights=args.projector_path,
        lora_adapter_dir=args.lora_adapter_dir,
    )
    run_single_inference(pipeline, args.n_samples, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
