"""Pure-logic tests for the single-image inference demo (no model, no GPU)."""
import json
from types import SimpleNamespace

from experiments.single_inference import _write_md, run


def _fake_result():
    snip = lambda t: SimpleNamespace(text=t)  # noqa: E731
    return SimpleNamespace(
        diagnosis="ABNORMAL",
        cls_confidence=0.83,
        evidence_used=[1, 3],
        reasoning="Step 1: opacity noted. Step 2: consistent with pneumonia.",
        retrieved=[snip("snippet A"), snip("snippet B")],
        raw_output="raw model text",
    )


class _FakePipe:
    def diagnose(self, path):  # noqa: D401 - mimics MeddiagPipeline.diagnose
        return _fake_result()


def test_run_prints_and_saves(tmp_path, capsys):
    rec = run(_FakePipe(), tmp_path / "x.jpg", tmp_path)
    assert rec["prediction"] == "ABNORMAL"
    assert rec["p_abnormal"] == 0.83
    assert rec["evidence_cited"] == [1, 3]
    assert (tmp_path / "single_inference.json").exists()

    md = (tmp_path / "single_inference.md").read_text(encoding="utf-8")
    assert "ABNORMAL" in md and "pneumonia" in md and "snippet A" in md

    out = capsys.readouterr().out
    assert "SINGLE INFERENCE" in out and "Prediction" in out and "ABNORMAL" in out


def test_write_md_handles_missing_confidence_and_empty_rag(tmp_path):
    rec = {"image": "x.jpg", "prediction": "NORMAL", "p_abnormal": None,
           "evidence_cited": [], "reasoning": "", "retrieved_snippets": []}
    _write_md(rec, tmp_path / "o.md")
    md = (tmp_path / "o.md").read_text(encoding="utf-8")
    assert "N/A" in md and "_(none)_" in md


def test_record_is_json_serialisable():
    rec = run(_FakePipe(), "img.jpg", None)
    back = json.loads(json.dumps(rec))
    assert {"image", "prediction", "p_abnormal", "evidence_cited",
            "reasoning", "retrieved_snippets", "raw_output"} <= set(back)
