"""Strict MIMIC-CXR report filtering — SRS §4, §13.

SRS rules implemented here:
  - Device-only findings must be ignored
  - Ambiguous cases must be skipped
  - MIMIC must be strictly cleaned (de-id artifacts, junk text)

These are applied before labeling so the label assignment never sees
bad text, and before training so no contaminated samples enter the loss.
"""
from __future__ import annotations

import re


# ── de-identification artifact patterns ────────────────────────────────────
# MIMIC text contains [** ... **] placeholders from de-identification.
_DEID_RE = re.compile(r"\[\*\*.*?\*\*\]")
# Trailing/leading whitespace normalization
_MULTI_SPACE_RE = re.compile(r"\s+")


# ── device-only report signals (SRS §4) ────────────────────────────────────
# Reports that ONLY discuss tubes, lines, catheters, pacemakers — no findings.
# A report is device-only if it matches one of these and has no other signal.
_DEVICE_PHRASES: tuple[str, ...] = (
    "lines and tubes in",
    "line and tube in",
    "endotracheal tube",
    "nasogastric tube",
    "ng tube",
    "pacemaker lead",
    "defibrillator lead",
    "icd lead",
    "port-a-cath",
    "portacath",
    "central venous catheter",
    "picc line",
    "chest tube",
    "interval placement of",
    "line tip projects",
    "catheter tip projects",
    "tube tip projects",
)

# Standalone "no pneumothorax, no pleural effusion" only (device context)
_DEVICE_ONLY_SINGLE_RE = re.compile(
    r"^(no pneumothorax[.,]?\s*)?(no pleural effusion[.,]?\s*)?(lines? and tubes?[^.]*\.?\s*)+$",
    re.IGNORECASE,
)


# ── ambiguous signals (SRS §4) ─────────────────────────────────────────────
_AMBIGUOUS_PHRASES: tuple[str, ...] = (
    "cannot exclude",
    "cannot be excluded",
    "cannot rule out",
    "cannot be ruled out",
    "may represent",
    "could represent",
    "questionable",
    "ill-defined",
    "indeterminate",
    "clinical correlation",
    "further evaluation",
    "recommend",
    "follow-up",
    "follow up recommended",
    "comparison with prior",
    "cannot assess",
    "limited exam",
    "limited study",
    "motion artifact",
    "poor inspiratory",
    "low lung volumes limit",
)


def clean_mimic_text(text: str) -> str:
    """Remove MIMIC de-id artifacts and normalize whitespace."""
    text = _DEID_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def is_device_only(text: str) -> bool:
    """Return True if the report discusses only hardware/lines with no pathology mention.

    SRS §4: 'Device-only findings must be ignored.'
    """
    lower = text.lower()
    device_count = sum(1 for phrase in _DEVICE_PHRASES if phrase in lower)
    if device_count == 0:
        return False
    # Short reports dominated by device language → device-only
    if len(text.split()) < 25 and device_count >= 1:
        return True
    # Pattern match for pure device-only template reports
    if _DEVICE_ONLY_SINGLE_RE.match(text.strip()):
        return True
    return False


def is_ambiguous(text: str) -> bool:
    """Return True if the report is clinically ambiguous.

    SRS §4: 'Ambiguous cases must be skipped.'
    Ambiguous = hedged language that prevents reliable NORMAL/ABNORMAL assignment.
    """
    lower = text.lower()
    return any(phrase in lower for phrase in _AMBIGUOUS_PHRASES)


def is_too_short(text: str, min_chars: int = 40) -> bool:
    """Filter reports that are too short to be clinically informative."""
    return len(text.strip()) < min_chars


def is_usable(text: str) -> bool:
    """Full usability gate: clean, length, device-only, and ambiguity checks.

    Returns True only if the report should enter training. The filter is
    intentionally strict — false negatives (skipping valid reports) are safer
    than false positives (training on bad labels).
    """
    cleaned = clean_mimic_text(text)
    if is_too_short(cleaned):
        return False
    if is_device_only(cleaned):
        return False
    if is_ambiguous(cleaned):
        return False
    return True
