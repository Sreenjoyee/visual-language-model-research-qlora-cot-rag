"""Prompt tests.

These enforce two SRS invariants that are easy to violate by accident:
    - §2: "Do not allow labels in inference prompts."
    - §12: "inference prompt must match training EXACTLY; no dynamic prompt changes."

If someone later adds a keyword argument that accepts a label, these tests
should catch it.
"""
from __future__ import annotations

import inspect

from src.prompts import (
    IMAGE_PLACEHOLDER,
    build_chat_messages,
    build_inference_prompt,
    format_retrieved_evidence,
)


FORBIDDEN_TOKENS = [
    "true label",
    "ground truth",
    "correct answer",
    "actual diagnosis",
    "the answer is",
]


def test_inference_prompt_signature_accepts_only_snippets():
    """Ensure no 'label' or similar parameter has been snuck in."""
    sig = inspect.signature(build_inference_prompt)
    params = set(sig.parameters)
    assert params == {"retrieved_snippets"}, f"Unexpected params: {params}"


def test_inference_prompt_contains_image_placeholder():
    prompt = build_inference_prompt(["Some clinical text about pneumonia."])
    assert IMAGE_PLACEHOLDER in prompt


def test_inference_prompt_has_no_forbidden_hint_tokens():
    prompt = build_inference_prompt(["A retrieved snippet."])
    low = prompt.lower()
    for token in FORBIDDEN_TOKENS:
        assert token not in low, f"Label-hint token in prompt: {token!r}"


def test_empty_evidence_is_flagged_not_silent():
    rendered = format_retrieved_evidence([])
    # If FAISS returns nothing, the prompt must say so explicitly — a silent
    # empty string would hide a critical upstream failure.
    assert "No clinical evidence retrieved" in rendered


def test_evidence_numbering_is_1_indexed_and_stable():
    rendered = format_retrieved_evidence(["alpha", "beta", "gamma"])
    assert "[1] alpha" in rendered
    assert "[2] beta" in rendered
    assert "[3] gamma" in rendered


def test_chat_messages_structure():
    msgs = build_chat_messages(["ev"])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert IMAGE_PLACEHOLDER in msgs[1]["content"]


def test_prompt_is_deterministic():
    """Same input -> same output, byte-for-byte. This is the cross-stage
    consistency guarantee that Stage-2 training must rely on."""
    a = build_inference_prompt(["one", "two"])
    b = build_inference_prompt(["one", "two"])
    assert a == b


def test_diagnosis_output_format_is_specified():
    """The prompt must pin the output schema so the parser has something to
    parse. If someone removes the DIAGNOSIS: line from the template this
    test fails loudly."""
    prompt = build_inference_prompt(["x"])
    assert "DIAGNOSIS:" in prompt
    assert "EVIDENCE_USED:" in prompt
    assert "REASONING:" in prompt