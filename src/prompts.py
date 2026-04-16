"""Prompt templates — the ONLY place prompts live.

SRS §12: "inference prompt must match training EXACTLY; no dynamic prompt changes."
Every stage and every script must import from here. Do not inline prompt strings
anywhere else in the codebase.

SRS §2: "Do not allow labels in inference prompts." The template APIs below do not
accept a label argument. A training-time variant that appends the label should be
added as a separate function when Stage 2 is built — it must not share the same
function name as the inference variant, to prevent accidental leakage.
"""
from __future__ import annotations

from typing import Sequence


# Placeholder token used for the visual embedding position in the prompt string.
# It is never actually tokenized — pipeline.py splices visual embeddings in at
# the corresponding position when building inputs_embeds.
IMAGE_PLACEHOLDER = "<IMAGE>"

# System prompt: neutral, no disease hints that could bias the classifier.
SYSTEM_PROMPT = (
    "You are a careful radiology assistant. You analyze chest X-rays and provide "
    "a binary classification (NORMAL or ABNORMAL) with step-by-step reasoning. "
    "You must ground your reasoning in the retrieved clinical evidence provided. "
    "If evidence is insufficient, say so rather than speculate."
)


def format_retrieved_evidence(snippets: Sequence[str]) -> str:
    """Render FAISS-retrieved snippets into the prompt.

    Kept deterministic: numbered, no reordering, no deduplication beyond exact
    string equality (caller's responsibility to pass clean snippets).
    """
    if not snippets:
        # SRS §6: retrieval is ALWAYS ON. If this branch triggers in production,
        # something upstream is broken — the pipeline should log loudly.
        return "[No clinical evidence retrieved.]"
    lines = [f"[{i + 1}] {s.strip()}" for i, s in enumerate(snippets)]
    return "\n".join(lines)


def build_inference_prompt(retrieved_snippets: Sequence[str]) -> str:
    """Build the full inference prompt.

    The returned string contains IMAGE_PLACEHOLDER at the position where visual
    embeddings should be spliced. pipeline.py handles the splice.

    No label field. No hints. Structure is fixed across calls — this is the
    contract that Stage-2 training must match byte-for-byte.
    """
    evidence = format_retrieved_evidence(retrieved_snippets)
    # Using Llama-3 chat format. If the tokenizer's chat template differs from
    # this layout, pipeline.py uses tokenizer.apply_chat_template to render it
    # consistently — we don't hardcode the special tokens here.
    user_content = (
        f"Chest X-ray image:\n{IMAGE_PLACEHOLDER}\n\n"
        f"Retrieved clinical evidence:\n{evidence}\n\n"
        "Task: Examine the image and the evidence above. Produce output in EXACTLY "
        "this format:\n"
        "DIAGNOSIS: <NORMAL or ABNORMAL>\n"
        "EVIDENCE_USED: <comma-separated list of evidence numbers you relied on, "
        "or NONE>\n"
        "REASONING:\n"
        "1. Visual observations: <what you see in the image>\n"
        "2. Clinical interpretation: <map observations to clinical meaning>\n"
        "3. Evidence support: <how the retrieved evidence supports or contradicts>\n"
        "4. Justification: <why the final diagnosis follows>\n"
    )
    return user_content


def build_chat_messages(retrieved_snippets: Sequence[str]) -> list[dict]:
    """Return a messages list suitable for tokenizer.apply_chat_template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_inference_prompt(retrieved_snippets)},
    ]