"""Pure-logic tests for the single-inference demo writers (no model, no GPU)."""
import json

from experiments.single_inference import _save_figure, _write_markdown


def _recs():
    return [
        {"index": 0, "source": "mimic", "true_label": "ABNORMAL",
         "ground_truth_report": "Findings: opacity in the right lung.",
         "predicted_diagnosis": "ABNORMAL", "p_abnormal": 0.83, "correct": True,
         "reasoning": "Step 1: opacity noted. Step 2: consistent with pneumonia.",
         "evidence_cited": [1, 3], "retrieved_snippets": ["snippet A", "snippet B"],
         "raw_output": "..."},
        {"index": 1, "source": "mimic", "true_label": "NORMAL",
         "ground_truth_report": "No acute findings.",
         "predicted_diagnosis": "ABNORMAL", "p_abnormal": 0.55, "correct": False,
         "reasoning": "Borderline.", "evidence_cited": [],
         "retrieved_snippets": [], "raw_output": "..."},
    ]


def test_write_markdown_contains_key_fields(tmp_path):
    p = tmp_path / "single_inference.md"
    _write_markdown(_recs(), p)
    text = p.read_text(encoding="utf-8")
    assert "Sample 0" in text and "Sample 1" in text
    assert "P(ABNORMAL)=0.830" in text
    assert "correct" in text and "WRONG" in text        # both outcomes rendered
    assert "consistent with pneumonia" in text          # reasoning chain present
    assert "snippet A" in text                           # retrieved snippets present
    assert "_(none — RAG returned nothing)_" in text     # empty-RAG fallback


def test_json_record_shape_is_serialisable():
    # the exact dict shape run_single_inference emits must round-trip through JSON
    blob = json.dumps({"n": 2, "samples": _recs()})
    back = json.loads(blob)
    assert back["n"] == 2
    assert {"true_label", "predicted_diagnosis", "p_abnormal", "reasoning",
            "evidence_cited", "retrieved_snippets"} <= set(back["samples"][0])


def test_save_figure_writes_png(tmp_path):
    from PIL import Image
    items = []
    for r in _recs():
        ip = tmp_path / f"img_{r['index']}.png"
        Image.new("L", (32, 32), color=128).save(ip)
        items.append((ip, r))
    out = tmp_path / "single_inference_examples.png"
    _save_figure(items, out)
    assert out.exists() and out.stat().st_size > 0


def test_save_figure_empty_is_noop(tmp_path):
    out = tmp_path / "none.png"
    _save_figure([], out)
    assert not out.exists()
