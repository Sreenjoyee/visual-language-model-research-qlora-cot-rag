"""Output parser — isolated so tests don't pull in torch / transformers.

Parses the structured LLM output defined in prompts.py. Never silently
normalizes a broken output into a diagnosis — returns UNPARSEABLE instead,
so upstream logging can flag bad runs. SRS §19.5: no hidden silent failures.
"""
from __future__ import annotations

import re

_DIAG_RE = re.compile(r"DIAGNOSIS:\s*(NORMAL|ABNORMAL)", re.IGNORECASE)
_EVIDENCE_RE = re.compile(r"EVIDENCE_USED:\s*([^\n]+)", re.IGNORECASE)
_REASONING_RE = re.compile(r"REASONING:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_output(text: str) -> dict:
    """Parse the structured output.

    Returns a dict with keys: diagnosis, evidence_used, reasoning.
    diagnosis is one of {"NORMAL", "ABNORMAL", "UNPARSEABLE"}.
    """
    diag_match = _DIAG_RE.search(text)
    diagnosis = diag_match.group(1).upper() if diag_match else "UNPARSEABLE"

    ev_match = _EVIDENCE_RE.search(text)
    evidence_used: list[int] = []
    if ev_match:
        raw = ev_match.group(1).strip()
        if raw.upper() != "NONE":
            for tok in re.split(r"[,\s]+", raw):
                tok = tok.strip().lstrip("[").rstrip("]")
                if tok.isdigit():
                    evidence_used.append(int(tok))

    rs_match = _REASONING_RE.search(text)
    reasoning = rs_match.group(1).strip() if rs_match else ""

    return {
        "diagnosis": diagnosis,
        "evidence_used": evidence_used,
        "reasoning": reasoning,
    }