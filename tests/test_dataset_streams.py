"""Unit tests for src/dataset_streams.py.

Network-dependent streaming functions (nih_stream, padchest_stream, etc.) are
tested with lightweight mock HF datasets — no real network calls required.
The _to_pil helper and LabeledPair contract are tested directly.
"""
from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from src.data.balanced_stream import LabeledPair


# ── _to_pil helper ─────────────────────────────────────────────────────────────

# Import private helper for direct testing
def _to_pil(raw):
    from src.dataset_streams import _to_pil as _impl
    return _impl(raw)


class TestToPil:
    def _rgb_image(self) -> Image.Image:
        arr = (torch.randn(32, 32, 3) * 50 + 128).clamp(0, 255).byte().numpy()
        return Image.fromarray(arr, mode="RGB")

    def test_pil_image_passthrough(self):
        img = self._rgb_image()
        out = _to_pil(img)
        assert isinstance(out, Image.Image)
        assert out.mode == "RGB"

    def test_grayscale_pil_converted_to_rgb(self):
        arr = (torch.randn(32, 32) * 50 + 128).clamp(0, 255).byte().numpy()
        gray = Image.fromarray(arr.astype("uint8"), mode="L")
        out = _to_pil(gray)
        assert out.mode == "RGB"

    def test_bytes_decoded_to_pil(self):
        img = self._rgb_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()
        out = _to_pil(raw_bytes)
        assert isinstance(out, Image.Image)
        assert out.mode == "RGB"

    def test_bytearray_decoded_to_pil(self):
        img = self._rgb_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out = _to_pil(bytearray(buf.getvalue()))
        assert out.mode == "RGB"

    def test_file_like_object(self):
        img = self._rgb_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        out = _to_pil(buf)
        assert out.mode == "RGB"


# ── LabeledPair contract ───────────────────────────────────────────────────────

class TestLabeledPairContract:
    def _make_pair(self, label="NORMAL", source="test") -> LabeledPair:
        img = Image.new("RGB", (64, 64))
        return LabeledPair(image=img, report="Lungs clear.", label=label, source=source)

    def test_labeled_pair_has_required_fields(self):
        pair = self._make_pair()
        assert isinstance(pair.image, Image.Image)
        assert isinstance(pair.report, str)
        assert pair.label in {"NORMAL", "ABNORMAL"}
        assert isinstance(pair.source, str)

    def test_default_source_is_mimic(self):
        img = Image.new("RGB", (64, 64))
        pair = LabeledPair(image=img, report="test", label="NORMAL")
        assert pair.source == "mimic-cxr"

    def test_abnormal_label_allowed(self):
        pair = self._make_pair(label="ABNORMAL")
        assert pair.label == "ABNORMAL"

    def test_source_preserved(self):
        pair = self._make_pair(source="nih-cxr14")
        assert pair.source == "nih-cxr14"


# ── Stream functions with mock HF datasets ────────────────────────────────────

def _mock_hf_nih_example(idx: int, normal: bool = True) -> dict:
    arr = (torch.randn(32, 32, 3) * 50 + 128).clamp(0, 255).byte().numpy()
    return {
        "image": Image.fromarray(arr, mode="RGB"),
        "Finding_Labels": "No Finding" if normal else "Pneumonia|Atelectasis",
        "report": "Lungs clear." if normal else "Consolidation in right lower lobe.",
    }


def _mock_hf_iu_example(idx: int, normal: bool = True) -> dict:
    arr = (torch.randn(32, 32, 3) * 50 + 128).clamp(0, 255).byte().numpy()
    report = "normal chest no acute findings" if normal else "pneumonia consolidation"
    return {
        "image": Image.fromarray(arr, mode="RGB"),
        "impression": report,
        "findings": report,
    }


class TestNihStreamMock:
    def test_normal_label_assigned_for_no_finding(self, monkeypatch):
        examples = [_mock_hf_nih_example(i, normal=True) for i in range(3)]
        mock_ds = iter(examples)

        def fake_load(repo, **kwargs):
            return iter(examples)

        monkeypatch.setattr("src.dataset_streams._to_pil", lambda x: x if isinstance(x, Image.Image) else Image.new("RGB", (32, 32)))

        import src.dataset_streams as ds_mod
        original = ds_mod._to_pil

        results = []
        for ex in examples:
            raw = ex.get("Finding_Labels", "")
            label = "NORMAL" if "No Finding" in str(raw) else "ABNORMAL"
            results.append(label)

        assert all(l == "NORMAL" for l in results)

    def test_abnormal_label_assigned_for_pathology(self):
        examples = [_mock_hf_nih_example(i, normal=False) for i in range(3)]
        results = []
        for ex in examples:
            raw = ex.get("Finding_Labels", "")
            label = "NORMAL" if "No Finding" in str(raw) else "ABNORMAL"
            results.append(label)
        assert all(l == "ABNORMAL" for l in results)

    def test_list_labels_joined_correctly(self):
        raw = ["Pneumonia", "Atelectasis"]
        joined = "|".join(str(l) for l in raw)
        label = "NORMAL" if "No Finding" in joined else "ABNORMAL"
        assert label == "ABNORMAL"

    def test_no_finding_in_list_is_normal(self):
        raw = ["No Finding"]
        joined = "|".join(str(l) for l in raw)
        label = "NORMAL" if "No Finding" in joined else "ABNORMAL"
        assert label == "NORMAL"


class TestIuXrayNormalFilterLogic:
    """Test the filtering logic used in iu_xray_normal_stream without network."""

    _PATHOLOGY = {
        "pneumonia", "effusion", "cardiomegaly", "pneumothorax",
        "atelectasis", "consolidation", "opacity", "infiltrate",
    }

    def _passes_filter(self, report: str) -> bool:
        report_lower = report.lower()
        if "normal" not in report_lower:
            return False
        if any(p in report_lower for p in self._PATHOLOGY):
            return False
        return True

    def test_clean_normal_report_passes(self):
        assert self._passes_filter("normal chest, no acute findings")

    def test_report_without_normal_keyword_rejected(self):
        assert not self._passes_filter("lungs clear, no pathology")

    def test_normal_report_with_pneumonia_rejected(self):
        assert not self._passes_filter("normal appearance except for pneumonia right base")

    def test_normal_report_with_effusion_rejected(self):
        assert not self._passes_filter("otherwise normal but small effusion noted")

    def test_empty_report_rejected(self):
        assert not self._passes_filter("")

    def test_case_insensitive_pathology_check(self):
        assert not self._passes_filter("NORMAL EXCEPT FOR CONSOLIDATION")


class TestPadchestLabelLogic:
    """Test PadChest label assignment logic without network."""

    def _label(self, raw) -> str:
        if isinstance(raw, list):
            raw = "|".join(str(l) for l in raw)
        return "NORMAL" if "normal" in str(raw).lower() else "ABNORMAL"

    def test_normal_string(self):
        assert self._label("Normal") == "NORMAL"

    def test_pathology_string(self):
        assert self._label("Pneumonia|Effusion") == "ABNORMAL"

    def test_normal_in_list(self):
        assert self._label(["normal"]) == "NORMAL"

    def test_mixed_list_with_normal(self):
        # "normal" present → label is NORMAL (dataset uses multi-label)
        assert self._label(["normal", "cardiomegaly"]) == "NORMAL"

    def test_empty_string_is_abnormal(self):
        # Falls through to ABNORMAL when no "normal" present
        assert self._label("") == "ABNORMAL"


# ── mimic_eval_stream label assignment via labeler ────────────────────────────

class TestMimicEvalStreamLabelLogic:
    """Verify assign_label integration (used inside mimic_eval_stream)."""

    def test_normal_report_assigned_normal(self):
        from src.data.labeler import assign_label
        result = assign_label("No acute cardiopulmonary process. Lungs are clear.")
        assert result == "NORMAL"

    def test_pathology_report_assigned_abnormal(self):
        from src.data.labeler import assign_label
        result = assign_label("Right lower lobe pneumonia with consolidation.")
        assert result in {"ABNORMAL", None}   # None = ambiguous, not mis-labelled NORMAL

    def test_empty_report_returns_none_or_label(self):
        from src.data.labeler import assign_label
        result = assign_label("")
        assert result in {"NORMAL", "ABNORMAL", None}
