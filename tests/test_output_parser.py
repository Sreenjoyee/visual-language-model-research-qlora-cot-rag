"""Unit tests for the output parser.

Runs without GPU, models, or network. These are the first tests in the §19
module-wise testing framework — full coverage will follow as modules land.
"""
from __future__ import annotations

from src.output_parser import parse_output


def test_parses_normal_diagnosis():
    text = (
        "DIAGNOSIS: NORMAL\n"
        "EVIDENCE_USED: 1, 3\n"
        "REASONING:\n"
        "1. Visual observations: clear lung fields.\n"
        "2. Clinical interpretation: nothing abnormal.\n"
        "3. Evidence support: evidence 1 and 3 consistent.\n"
        "4. Justification: no findings.\n"
    )
    out = parse_output(text)
    assert out["diagnosis"] == "NORMAL"
    assert out["evidence_used"] == [1, 3]
    assert "Visual observations" in out["reasoning"]


def test_parses_abnormal_with_none_evidence():
    text = "DIAGNOSIS: ABNORMAL\nEVIDENCE_USED: NONE\nREASONING:\nsomething\n"
    out = parse_output(text)
    assert out["diagnosis"] == "ABNORMAL"
    assert out["evidence_used"] == []


def test_unparseable_when_missing_diagnosis():
    text = "The patient appears fine overall."
    out = parse_output(text)
    assert out["diagnosis"] == "UNPARSEABLE"
    assert out["evidence_used"] == []


def test_case_insensitive_diagnosis():
    text = "diagnosis: abnormal\nEVIDENCE_USED: 2"
    out = parse_output(text)
    assert out["diagnosis"] == "ABNORMAL"
    assert out["evidence_used"] == [2]


def test_bracketed_evidence_ids():
    text = "DIAGNOSIS: NORMAL\nEVIDENCE_USED: [1], [2], [4]\nREASONING: ok"
    out = parse_output(text)
    assert out["evidence_used"] == [1, 2, 4]


def test_extra_whitespace_and_mixed_case_evidence_label():
    text = "DIAGNOSIS: NORMAL\n  Evidence_Used:   1  2  \nREASONING: x"
    out = parse_output(text)
    assert out["diagnosis"] == "NORMAL"
    assert out["evidence_used"] == [1, 2]


def test_reasoning_captures_multiline_block():
    text = (
        "DIAGNOSIS: ABNORMAL\nEVIDENCE_USED: 1\n"
        "REASONING:\nline one\nline two\nline three"
    )
    out = parse_output(text)
    assert "line one" in out["reasoning"]
    assert "line three" in out["reasoning"]