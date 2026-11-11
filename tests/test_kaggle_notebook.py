"""Tests for kaggle_train_fresh.ipynb, run against the notebook's own cell code.

No Kaggle, secrets or network needed. Covered:
  * every code cell is valid Python
  * checkpoint round trip: package() -> Kaggle --dir-mode zip -> _expand_zips ->
    _restore_bundle, including dropping a corrupt newest checkpoint
  * pipeline-state reconstruction (cell 7) and the run_pipeline.sh patch
  * the MEDDIAG_VRAM_CAP_GB hard cap on a real GPU (skipped without CUDA)
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "kaggle_train_fresh.ipynb"


def _code_cells() -> list[str]:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _cell(header: str) -> str:
    hits = [c for c in _code_cells() if c.lstrip().startswith(header)]
    assert len(hits) == 1, f"expected one cell starting with {header!r}"
    return hits[0]


def _defs_only(src: str) -> str:
    """Just the function definitions of a cell, so they can be exec'd without side effects."""
    mod = ast.parse(src)
    mod.body = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    return ast.unparse(mod)


def test_all_code_cells_compile():
    cells = _code_cells()
    assert len(cells) == 8
    for i, src in enumerate(cells):
        compile(src, f"<cell {i}>", "exec")


# --------------------------------------------------------------------------- #
# Checkpoint round trip
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


def _ns(repo: Path) -> dict:
    return {
        "os": os, "re": re, "json": json, "time": time, "shutil": shutil,
        "zipfile": zipfile, "subprocess": subprocess, "torch": torch, "Path": Path,
        "REPO_DIR": str(repo), "MODELS": repo / "models", "LOGS": repo / "logs",
        "FAISS": repo / "faiss_index", "OUT": repo.parent / "_upload",
        "_FRESH_S": 0, "STATE_DATASET": "meddiag-fresh-t4", "KAGGLE_USERNAME": "u",
    }


def _make_training_state(repo: Path) -> None:
    m = repo / "models"; lg = repo / "logs"; fa = repo / "faiss_index"
    for d in (m, lg, fa):
        d.mkdir(parents=True)
    proj = {"vision_proj.weight": torch.zeros(4, 1024)}
    for step in (250, 500):   # Stage-1 still running: only mid-checkpoints exist
        torch.save({"step": step, "projector": proj, "optimizer": {}, "scheduler": {}},
                   m / f"projector_step{step}.pt")
    for step in (250, 500, 750):
        d = m / f"lora_step{step}"; d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(b"\0" * 16)
        (d / "adapter_config.json").write_text("{}")
        torch.save({"global_step": step}, d / "train_state.pt")
    # newest checkpoint was cut off mid-save (session killed)
    (m / "lora_step750" / "train_state.pt").write_bytes(b"truncated")
    torch.save({"w": torch.zeros(2)}, m / "cls_head.pt")
    (fa / "index.faiss").write_bytes(b"faiss")
    (fa / "meta.jsonl").write_text('{"text": "x"}\n')
    (lg / ".pipeline_state").write_text("step0\nstep1\n")
    (lg / "stage2.jsonl").write_text('{"step": 500, "vram_gb": 4.1}\n')
    rep = repo / "results" / "experiments" / "20260101_000000"; rep.mkdir(parents=True)
    (rep / "exp1_results.json").write_text("{}")
    time.sleep(0.05)


def _kaggle_dir_mode_zip(src: Path, dst: Path, nest: bool) -> None:
    """Imitate `kaggle datasets version --dir-mode zip` + `download --unzip`:
    top-level files stay as files, each top-level folder becomes <name>.zip."""
    dst.mkdir(parents=True)
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)
            continue
        with zipfile.ZipFile(dst / f"{p.name}.zip", "w") as z:
            for f in p.rglob("*"):
                rel = f.relative_to(src if nest else p)
                z.write(f, rel.as_posix())


@pytest.mark.parametrize("nest", [True, False], ids=["zip-with-top-folder", "zip-flat"])
def test_checkpoint_roundtrip_drops_corrupt_newest(tmp_path, nest):
    run1 = tmp_path / "run1" / "vlm"
    _make_training_state(run1)
    ns1 = _ns(run1)
    exec(_defs_only(_cell("# 8. TRAIN")), ns1)
    n = ns1["package"]()
    assert n >= 6   # 2 projector_step + 3 lora_step + cls_head

    dl = tmp_path / "download"
    _kaggle_dir_mode_zip(ns1["OUT"], dl, nest)

    run2 = tmp_path / "run2" / "vlm"
    for d in ("models", "logs"):
        (run2 / d).mkdir(parents=True)
    ns2 = _ns(run2)
    exec(_defs_only(_cell("# 5. RESTORE")), ns2)
    ns2["_expand_zips"](dl)
    assert not list(dl.rglob("*.zip"))
    got = ns2["_restore_bundle"](dl)

    m = run2 / "models"
    assert got["stage2_step"] == 500, got
    assert not (m / "lora_step750").exists(), "corrupt newest checkpoint must be dropped"
    assert (m / "lora_step500" / "train_state.pt").exists()
    assert (m / "lora_step250").exists()
    assert got.get("stage1_mid") == 500 and "projector" not in got
    assert [p.name for p in m.glob("projector_*.pt")] == ["projector_step500.pt"]
    assert got["faiss"] and (run2 / "faiss_index" / "meta.jsonl").exists()
    assert got.get("cls_head.pt")
    assert (run2 / "logs" / ".pipeline_state").read_text() == "step0\nstep1\n"
    assert (run2 / "logs" / "stage2.jsonl").exists()
    assert (run2 / "results" / "experiments" / "20260101_000000" / "exp1_results.json").exists()


def test_package_skips_files_still_being_written(tmp_path):
    repo = tmp_path / "vlm"
    _make_training_state(repo)
    ns = _ns(repo)
    ns["_FRESH_S"] = 3600          # everything counts as "just written"
    exec(_defs_only(_cell("# 8. TRAIN")), ns)
    assert ns["package"]() == 0


# --------------------------------------------------------------------------- #
# Pipeline state (cell 7)
# --------------------------------------------------------------------------- #
def _run_cell7(repo: Path, state: str) -> list[str]:
    for d in ("models", "logs", "faiss_index"):
        (repo / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "run_pipeline.sh", repo / "run_pipeline.sh")
    (repo / "logs" / ".pipeline_state").write_text(state)
    ns = _ns(repo)
    ns.update(GRAD_ACCUM=4, MAX_PAIRS_S2=4000)
    exec(_cell("# 7. PIPELINE STATE"), ns)
    return (repo / "logs" / ".pipeline_state").read_text().split()


def test_state_fresh_run_runs_everything(tmp_path):
    assert _run_cell7(tmp_path / "vlm", "") == ["step0"]


def test_state_trusts_markers_only_with_artifacts(tmp_path):
    repo = tmp_path / "vlm"
    (repo / "faiss_index").mkdir(parents=True)
    (repo / "faiss_index" / "index.faiss").write_bytes(b"x")
    (repo / "faiss_index" / "meta.jsonl").write_text("{}")
    # stale markers claiming Stage-1/2 and eval are done, but no artifacts for them
    assert _run_cell7(repo, "step0\nstep1\nstep2\nstep3\nstep4\n") == ["step0", "step1"]


def test_state_finished_training_keeps_eval_progress(tmp_path):
    repo = tmp_path / "vlm"
    (repo / "models" / "lora_adapter").mkdir(parents=True)
    (repo / "models" / "cls_head.pt").write_bytes(b"x")
    (repo / "models" / "projector_stage1.pt").write_bytes(b"x")
    (repo / "faiss_index").mkdir()
    (repo / "faiss_index" / "index.faiss").write_bytes(b"x")
    (repo / "faiss_index" / "meta.jsonl").write_text("{}")
    out = _run_cell7(repo, "step0\nstep1\nstep2\nstep3\nstep4\n")
    assert out == ["step0", "step1", "step2", "step3", "step4"]


def test_run_pipeline_patch_only_touches_full_run_values(tmp_path):
    repo = tmp_path / "vlm"
    _run_cell7(repo, "")
    sh = (repo / "run_pipeline.sh").read_text()
    assert re.search(r"^\s*GRAD_ACCUM=4\b", sh, re.M)
    assert re.search(r"^\s*GRAD_ACCUM=1\b", sh, re.M), "smoke-mode value must be untouched"
    assert not re.search(r"^\s*GRAD_ACCUM=8\b", sh, re.M)


# --------------------------------------------------------------------------- #
# Hard VRAM cap (src/config.py) on a real GPU
# --------------------------------------------------------------------------- #
def _alloc_in_child(gb: float, cap: str | None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("MEDDIAG_VRAM_CAP_GB", None)
    if cap is not None:
        env["MEDDIAG_VRAM_CAP_GB"] = cap
    code = f"import src.config, torch; torch.empty(int({gb} * 2**30), dtype=torch.uint8, device='cuda')"
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_vram_cap_blocks_allocations_above_budget():
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    probe = round(total * 0.6, 1)          # fits the GPU comfortably...
    cap = str(round(probe - 1.0, 1))       # ...but not the cap
    ok = _alloc_in_child(probe, None)
    assert ok.returncode == 0, f"control allocation failed without a cap: {ok.stderr[-400:]}"
    blocked = _alloc_in_child(probe, cap)
    assert blocked.returncode != 0 and "OutOfMemory" in blocked.stderr, blocked.stderr[-400:]
    under = _alloc_in_child(float(cap) - 1.0, cap)
    assert under.returncode == 0, f"allocation under the cap failed: {under.stderr[-400:]}"
