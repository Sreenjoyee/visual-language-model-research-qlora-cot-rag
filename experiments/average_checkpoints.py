"""Weight-averaging (SWA) of the last N LoRA checkpoints into one adapter.

Averaging the adapter weights of several late checkpoints usually yields a model
at least as good as — and often better-calibrated than — any single checkpoint,
at no extra inference cost. Purely post-hoc; no retraining.

Usage:
    python -m experiments.average_checkpoints \\
        --models-dir models --last-n 8 --out models/lora_adapter_swa
Then evaluate with:  --lora-adapter-dir models/lora_adapter_swa
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

_PAT = re.compile(r"lora_step(\d+)$")


def find_checkpoints(models_dir: Path, last_n: int) -> list[Path]:
    """Return the `last_n` highest-step checkpoint dirs that have an adapter file."""
    cks = sorted(
        (int(_PAT.match(d.name).group(1)), d)
        for d in models_dir.iterdir()
        if d.is_dir() and _PAT.match(d.name)
        and (d / "adapter_model.safetensors").exists()
    )
    return [d for _, d in cks[-last_n:]]


def average_adapters(checkpoint_dirs: list[Path], out_dir: Path) -> dict:
    """Element-wise mean of adapter_model.safetensors across checkpoints.

    Averages in float32 then casts back to each tensor's original dtype. Writes
    the averaged adapter (+ copied adapter_config.json) to out_dir and returns it.
    """
    if not checkpoint_dirs:
        raise ValueError("no checkpoints to average")

    states = [load_file(str(d / "adapter_model.safetensors")) for d in checkpoint_dirs]
    keys = set(states[0].keys())
    for d, s in zip(checkpoint_dirs[1:], states[1:]):
        if set(s.keys()) != keys:
            raise ValueError(f"adapter key mismatch at {d.name} — cannot average")

    averaged = {}
    for k in states[0].keys():
        orig_dtype = states[0][k].dtype
        acc = sum(s[k].float() for s in states) / len(states)
        averaged[k] = acc.to(orig_dtype)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(averaged, str(out_dir / "adapter_model.safetensors"))
    cfg = checkpoint_dirs[-1] / "adapter_config.json"
    if cfg.exists():
        shutil.copy2(cfg, out_dir / "adapter_config.json")
    return averaged


def main() -> int:
    ap = argparse.ArgumentParser(description="SWA weight-averaging of last-N LoRA checkpoints")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--last-n", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("models/lora_adapter_swa"))
    args = ap.parse_args()

    cks = find_checkpoints(args.models_dir, args.last_n)
    if not cks:
        print("No lora_step* checkpoints with adapter_model.safetensors found.")
        return 1
    print(f"Averaging {len(cks)} checkpoints: {[d.name for d in cks]}")
    average_adapters(cks, args.out)
    print(f"Saved averaged adapter → {args.out}")
    print(f"Evaluate with:  --lora-adapter-dir {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
