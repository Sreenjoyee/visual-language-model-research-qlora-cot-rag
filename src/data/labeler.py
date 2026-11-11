"""NORMAL / ABNORMAL label assignment from radiology report text.

SRS §2: "Do not allow labels in inference prompts."
SRS §7 Stage 2: classify NORMAL / ABNORMAL.
SRS §4: no label leakage — labels come from text signals, never from
        ground-truth columns that could contaminate inference prompts.

The labeler is intentionally conservative:
  - Returns None (skip) when the signal is weak or contradictory.
  - NORMAL requires an explicit clear-chest signal — not just absence of
    abnormal terms (absence of evidence ≠ evidence of absence in radiology).
  - ABNORMAL requires at least one explicit pathology term.
"""
from __future__ import annotations

import re

from .filters import is_ambiguous


# ── NORMAL signals ─────────────────────────────────────────────────────────
# These phrases reliably indicate a clean chest X-ray. We require at least
# one STRONG normal signal. Weak signals (e.g. "no change") are insufficient.
_STRONG_NORMAL: tuple[str, ...] = (
    "no acute cardiopulmonary",
    "no acute cardiopulmonary process",
    "no acute cardiopulmonary abnormality",
    "no acute cardiopulmonary disease",
    "lungs are clear",
    "lungs clear",
    "clear lungs bilaterally",
    "normal chest radiograph",
    "normal chest x-ray",
    "normal chest",
    "clear and expanded",
    "no acute intrathoracic",
    "no acute pulmonary",
    "no pneumonia",
    "no consolidation, no effusion",
    "no pleural effusion or pneumothorax",
    "no pneumothorax or pleural effusion",
    "unremarkable chest",
    "within normal limits",
    "no acute findings",
    "no acute abnormality",
)

# Weak normals — only count when NO abnormal signal is present
_WEAK_NORMAL: tuple[str, ...] = (
    "no significant change",
    "stable appearance",
    "stable chest",
    "unchanged",
)


# ── ABNORMAL signals ────────────────────────────────────────────────────────
# Any of these phrases reliably indicate pathology.
_ABNORMAL_TERMS: tuple[str, ...] = (
    "pneumonia",
    "consolidation",
    "infiltrate",
    "infiltration",
    "opacity",
    "opacification",
    "pleural effusion",
    "effusion",
    "pneumothorax",
    "atelectasis",
    "collapse",
    "mass",
    "nodule",
    "nodular",
    "cardiomegaly",
    "enlarged cardiac",
    "enlarged",             # "the heart is mildly enlarged" (IU-Xray probe case labelled NORMAL)
    "enlargement",          # cardiac / hilar / mediastinal enlargement
    "pulmonary edema",
    "edema",
    "congestion",
    "vascular congestion",
    "interstitial",
    "fibrosis",
    "fracture",
    "rib fracture",
    "pleural thickening",
    "pleural plaque",
    "hilar enlargement",
    "hilar prominence",
    "mediastinal widening",
    "mediastinal mass",
    "tracheal deviation",
    "subcutaneous emphysema",
    "emphysema",            # e.g. IU-Xray "findings suggestive of emphysema" (was labelled NORMAL)
    "hyperinflat",          # hyperinflation / hyperinflated lungs (COPD sign)
    "hyperexpan",           # hyperexpanded lungs
    "copd",
    "airspace disease",
    "ground glass",
    "reticular",
    "cavitation",
    "abscess",
    "empyema",
    "hemothorax",
    "pneumomediastinum",
    "lymphadenopathy",
)

# Negation prefixes that flip an abnormal term to normal context
# NegEx-style negation (Chapman et al.): a negation cue negates a pathology term
# that follows it within the SAME sentence and within _NEG_SCOPE_WORDS words,
# unless a termination word ("but", "however", ...) intervenes. Counting words
# (not characters) handles lists like "no visualized pneumothorax or large
# pleural effusion", which a 15-character window missed — so clean reports were
# skipped as "contradictory".
_NEGATION_CUE_RE = re.compile(
    r"\b(no|not|without|absence of|absent|negative for|ruled out|clear of|free of)\b",
    re.IGNORECASE,
)
# Pseudo-negations: "no change in the effusion" means the effusion PERSISTS.
_PSEUDO_NEGATION_RE = re.compile(
    r"\bno\s+(significant\s+|interval\s+|appreciable\s+)?"
    r"(change|increase|decrease|improvement|worsening)\b",
    re.IGNORECASE,
)
_NEG_TERMINATORS = re.compile(
    r"\b(but|however|although|though|except|there is|there are|which|concerning for|new)\b",
    re.IGNORECASE,
)
_NEG_SCOPE_WORDS = 6


def _has_term(text_lower: str, term: str) -> bool:
    return term in text_lower


def _is_negated(text_lower: str, term: str) -> bool:
    """True if EVERY occurrence of `term` is negated (one affirmed mention = present)."""
    start, found = 0, False
    while True:
        idx = text_lower.find(term, start)
        if idx < 0:
            return found
        found = True
        if not _occurrence_negated(text_lower, idx):
            return False
        start = idx + len(term)


def _occurrence_negated(text_lower: str, idx: int) -> bool:
    sent_start = max(text_lower.rfind(".", 0, idx), text_lower.rfind(";", 0, idx)) + 1
    prefix = text_lower[sent_start:idx]
    cues = list(_NEGATION_CUE_RE.finditer(prefix))
    if not cues:
        return False
    cue = cues[-1]
    scope = prefix[cue.end():]
    if _PSEUDO_NEGATION_RE.match(prefix, cue.start()):
        return False
    if _NEG_TERMINATORS.search(scope):
        return False
    return len(scope.split()) <= _NEG_SCOPE_WORDS


def assign_label(text: str) -> str | None:
    """Assign NORMAL, ABNORMAL, or None (skip) from report text.

    Returns:
        'NORMAL'   — report clearly describes a clean chest
        'ABNORMAL' — report contains at least one un-negated pathology term
        None       — signal is weak, contradictory, or ambiguous (skip this sample)
    """
    # Ambiguous language means we can't reliably assign any label — skip first.
    if is_ambiguous(text):
        return None

    lower = text.lower()

    # Check for strong NORMAL signal first
    has_strong_normal = any(_has_term(lower, s) for s in _STRONG_NORMAL)

    # Count un-negated ABNORMAL terms
    abnormal_hits = [
        term for term in _ABNORMAL_TERMS
        if _has_term(lower, term) and not _is_negated(lower, term)
    ]

    if has_strong_normal and not abnormal_hits:
        return "NORMAL"

    if has_strong_normal and abnormal_hits:
        # Contradictory — has both normal phrase and pathology → skip
        return None

    if abnormal_hits:
        return "ABNORMAL"

    # Weak normal signals only count when no abnormal evidence exists
    has_weak_normal = any(_has_term(lower, s) for s in _WEAK_NORMAL)
    if has_weak_normal and not abnormal_hits:
        return "NORMAL"

    # No reliable signal in either direction → skip
    return None
