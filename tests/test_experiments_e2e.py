"""End-to-end test of ALL 10 research experiments (Exp 1, 2, 3, 4a, 4b, 5, 6, 7, 8, 9).

Runs the REAL experiment code in experiments/run_experiments.py, including
main() orchestration (results folder, failure detection, resume), plus the real
eval loop (src/eval_runner.py), metrics and plotting. Only three things are faked
so it runs on CPU in seconds with no network or secrets:

  * the pipeline: a tiny stand-in with the same interface as MeddiagPipeline
    (vision / projector / llm / retriever / cls_head / _splice_visual / _caption_image)
  * the HF data streams: synthetic X-rays where brighter images are ABNORMAL, so
    the ClassificationHead path is separable and metrics are meaningful
  * BERTScore (would download roberta-large) and Exp 1's config.py rewrite
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from PIL import Image

import experiments.run_experiments as rx
from src.config import CONFIG
from src.data.balanced_stream import LabeledPair

ALL = ["1", "2", "3", "4a", "4b", "5", "6", "7", "8", "9"]
OUTPUT = ("DIAGNOSIS: ABNORMAL\nEVIDENCE_USED: [1, 2]\n"
          "REASONING: Step 1: opacity in the right lower lobe. Step 2: consistent with evidence [1].")


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _Vision:
    def preprocess(self, image):
        v = float(np.asarray(image.convert("L"), dtype=np.float32).mean() / 255.0)
        return torch.full((1, 1, 4, 4), v)

    def __call__(self, px):
        return px.flatten(1).mean(1, keepdim=True).expand(1, 4, 16).clone()


class _Projector:
    def __call__(self, tokens):
        return torch.full((1, 8, 32), float(tokens.mean()))   # (B, K visual tokens, D_llm)


class _ClsHead:
    def __call__(self, ve, rag):
        assert rag.dim() == 3 and rag.shape[-1] == 384, f"bad rag embeddings {tuple(rag.shape)}"
        s = (ve.mean() - 0.5) * 10.0
        return torch.stack([-s, s]).unsqueeze(0)


class _Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert messages and all("content" in m for m in messages)
        return " ".join(m["content"] for m in messages)

    def decode(self, ids, skip_special_tokens=True):
        return OUTPUT

    def encode(self, text, add_special_tokens=False):
        return [1] if "AB" in text else [2]


class _Model:
    def generate(self, **kw):
        assert "inputs_embeds" in kw and "input_ids" not in kw, "SRS: inputs_embeds-only forward"
        seq = torch.tensor([[5, 6, 1]])
        if kw.get("return_dict_in_generate"):
            return SimpleNamespace(sequences=seq, scores=())
        return seq


class _Retriever:
    def __init__(self):
        self.embedder = SimpleNamespace(
            encode=lambda texts, **kw: np.ones((len(texts), 384), dtype=np.float32) / 384)

    def query(self, text, k=None):
        k = CONFIG.retrieval_top_k if k is None else k
        return [SimpleNamespace(text=f"evidence {i}", source="test",
                                embedding=np.full(384, 0.01 * (i + 1), dtype=np.float32))
                for i in range(k)]


class FakePipeline:
    def __init__(self, *a, **kw):
        self.config = dataclasses.replace(CONFIG)
        self.vision, self.projector, self.cls_head = _Vision(), _Projector(), _ClsHead()
        self.llm = SimpleNamespace(device=torch.device("cpu"), tokenizer=_Tokenizer(), model=_Model())
        self.retriever = _Retriever()

    def _caption_image(self, ve):
        return "chest x-ray"

    def _splice_visual(self, prompt, ve):
        return torch.zeros(1, 12, ve.shape[-1]), torch.ones(1, 12, dtype=torch.long)


def _pairs(n, labels=("NORMAL", "ABNORMAL"), source="test", **_):
    out = []
    for i in range(n):
        lbl = labels[i % len(labels)]
        level = 60 + 10 * i if lbl == "NORMAL" else 170 + 10 * i
        img = Image.new("L", (64, 64), min(level, 250)).convert("RGB")
        out.append(LabeledPair(image=img, label=lbl, source=source,
                               report="Right lower lobe opacity. No pneumothorax." if lbl == "ABNORMAL"
                               else "Lungs are clear. No acute cardiopulmonary process."))
    return iter(out)


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setattr(rx, "MeddiagPipeline", FakePipeline)
    monkeypatch.setattr(rx, "nih_stream", lambda hf_token="", max_samples=10, **k: _pairs(max_samples, source="nih"))
    monkeypatch.setattr(rx, "padchest_stream", lambda hf_token="", max_samples=10, **k: _pairs(max_samples, source="padchest"))
    monkeypatch.setattr(rx, "mimic_eval_stream", lambda hf_token="", max_samples=10, **k: _pairs(max_samples, source="mimic"))
    monkeypatch.setattr(rx, "iu_xray_normal_stream", lambda max_samples=10, **k: _pairs(max_samples, labels=("NORMAL",), source="iu"))
    monkeypatch.setattr(rx, "bertscore_f1_batch", lambda cands, refs: [0.8] * len(cands))
    monkeypatch.setattr(rx, "_apply_optimal_threshold",   # never rewrite the real src/config.py
                        lambda pipeline, t: setattr(pipeline.config, "classification_threshold", t))


def _run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_experiments", *args])
    rx.main()


def _run_dir(base: Path) -> Path:
    dirs = [d for d in base.iterdir() if d.is_dir()]
    assert len(dirs) == 1, dirs
    return dirs[0]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_all_ten_experiments_run_and_save_results(fake_env, monkeypatch, tmp_path):
    out_base = tmp_path / "results" / "experiments"
    _run_main(monkeypatch, "--exp", "all", "--max-samples", "8", "--output-dir", str(out_base))

    run = _run_dir(out_base)                      # one timestamped folder per run
    for e in ALL:
        f = run / f"exp{e}_results.json"
        assert f.exists(), f"Exp {e} wrote no results file"
        r = json.loads(f.read_text(encoding="utf-8"))
        assert "error" not in r, f"Exp {e} reported an error: {r}"

    allr = json.loads((run / "all_results.json").read_text(encoding="utf-8"))
    assert sorted(allr) == sorted(f"exp{e}" for e in ALL)
    assert list(run.glob("*.png")), "experiments should also save their plots"

    # Sanity of key numbers (synthetic data is separable on the cls_head path)
    exp1 = allr["exp1"]
    assert exp1["n_samples"] == 8 and exp1["metrics"]["auroc"] == pytest.approx(1.0)
    assert allr["exp2"]["rag"]["bertscore_f1"]["mean"] == pytest.approx(0.8)
    assert allr["exp3"]["meddiag_rag_bertscore"] == pytest.approx(0.8)   # reused from Exp 2
    assert allr["exp4a"]["n_samples"] == 8
    exp6 = allr["exp6"]          # raw + temperature-scaled ECE on disjoint calib/eval halves
    assert exp6["n_samples"] == 8 and exp6["n_calibration"] == 8
    assert {"ece", "ece_temperature_scaled", "temperature"} <= set(exp6)
    assert (_run_dir(out_base) / "exp6_calibration_curve_temperature_scaled.png").exists()
    assert set(allr["exp7"]["results_by_k"]) == {"1", "3", "5", "10"}
    # fake inference takes <1 ms, so per-inference energy rounds to ~0; check the structure
    assert allr["exp8"]["energy_per_inference_kwh"] >= 0 and allr["exp8"]["gpt4v_cloud_estimate_kwh"] > 0


def test_exp9_every_ablation_actually_runs(fake_env, monkeypatch, tmp_path):
    """Regression: 'No RAG' and 'No CoT' used to skip every sample (n=0, AUROC=nan)."""
    _run_main(monkeypatch, "--exp", "9", "--max-samples", "6", "--output-dir", str(tmp_path))
    abl = json.loads((_run_dir(tmp_path) / "exp9_results.json").read_text(encoding="utf-8"))["ablations"]
    for cfg in ("full_system", "no_rag", "no_cls_head", "no_cot"):
        assert abl[cfg]["n"] == 6, f"{cfg} evaluated {abl[cfg]['n']}/6 samples"
    assert abl["full_system"]["auroc"] == pytest.approx(1.0)
    assert abl["no_lora"]["n"] == 0 and abl["no_lora"]["auroc"] is None   # documented skip


def test_failing_experiment_fails_the_run_and_resume_redoes_only_it(fake_env, monkeypatch, tmp_path):
    calls = {"nih": 0, "padchest": 0}

    def broken_padchest(hf_token="", max_samples=10, **k):
        calls["padchest"] += 1
        raise ConnectionError("PadChest mirror down")

    def counting_nih(hf_token="", max_samples=10, **k):
        calls["nih"] += 1
        return _pairs(max_samples, source="nih")

    monkeypatch.setattr(rx, "padchest_stream", broken_padchest)
    monkeypatch.setattr(rx, "nih_stream", counting_nih)
    run = tmp_path / "run"

    with pytest.raises(SystemExit) as ei:           # failure must NOT look like success
        _run_main(monkeypatch, "--exp", "1,4b", "--max-samples", "4", "--resume-dir", str(run))
    assert ei.value.code == 1
    assert (run / "exp1_results.json").exists() and not (run / "exp4b_results.json").exists()
    assert "error" in json.loads((run / "all_results.json").read_text(encoding="utf-8"))["exp4b"]

    # Mirror is back: resume redoes ONLY Exp 4B, keeps Exp 1, re-applies its threshold
    monkeypatch.setattr(rx, "padchest_stream", lambda hf_token="", max_samples=10, **k: _pairs(max_samples))
    _run_main(monkeypatch, "--exp", "1,4b", "--max-samples", "4", "--resume-dir", str(run))
    assert calls["nih"] == 1, "Exp 1 was recomputed instead of skipped"
    assert (run / "exp4b_results.json").exists()


def test_empty_stream_counts_as_failure(fake_env, monkeypatch, tmp_path):
    monkeypatch.setattr(rx, "padchest_stream", lambda hf_token="", max_samples=10, **k: iter([]))
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, "--exp", "4b", "--max-samples", "4", "--resume-dir", str(tmp_path))


def test_default_results_location_is_dedicated_folder():
    import argparse
    src = Path(rx.__file__).read_text(encoding="utf-8")
    assert 'default="results/experiments"' in src
    sh = (Path(rx.__file__).resolve().parent.parent / "run_pipeline.sh").read_text(encoding="utf-8")
    assert 'EXPERIMENTS_DIR="results/experiments"' in sh and '--resume-dir   "$EXP_RUN_DIR"' in sh
