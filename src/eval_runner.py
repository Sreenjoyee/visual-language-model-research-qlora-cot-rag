"""Evaluation inference loop for MEDDIAG.

Runs MeddiagPipeline over a labeled stream of LabeledPair objects, collecting
per-sample ScoredResult objects that include:
  - Predicted label and ground-truth label
  - P(ABNORMAL) extracted from the first generated token's logits
  - Latency and peak VRAM per sample

The probability extraction uses generate(output_scores=True) so no extra
forward pass is needed. Falls back to p=0.5 if logits are unavailable
(e.g., some quantised model / transformers version combinations).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

import torch
import torch.nn.functional as F
from PIL import Image

from .data.balanced_stream import LabeledPair
from .output_parser import parse_output
from .pipeline import DiagnosisResult, MeddiagPipeline
from .prompts import build_chat_messages


@dataclass
class ScoredResult:
    true_label: str           # "NORMAL" or "ABNORMAL"
    pred_label: str           # "NORMAL", "ABNORMAL", or "UNPARSEABLE"
    p_abnormal: float         # model's probability estimate for ABNORMAL [0, 1]
    evidence_used: list[int]
    reasoning: str
    latency_s: float
    vram_peak_gb: float
    source: str
    correct: bool = field(init=False)

    def __post_init__(self) -> None:
        self.correct = (self.true_label == self.pred_label)


def _extract_p_abnormal(scores: tuple, tokenizer) -> float:
    """Extract P(ABNORMAL) from the first-step generate scores.

    scores[0] has shape (1, vocab_size) — logits for the very first token.
    We take softmax over only the first tokens of "NORMAL" and "ABNORMAL".
    """
    first_logits = scores[0][0]   # (vocab_size,)
    norm_ids = tokenizer.encode("NORMAL",   add_special_tokens=False)
    abn_ids  = tokenizer.encode("ABNORMAL", add_special_tokens=False)
    if not norm_ids or not abn_ids:
        return 0.5
    logit_n = first_logits[norm_ids[0]]
    logit_a = first_logits[abn_ids[0]]
    probs = F.softmax(torch.stack([logit_n, logit_a]), dim=0)
    return float(probs[1].item())


@torch.no_grad()
def _diagnose_scored(
    pipeline: MeddiagPipeline,
    image: Image.Image,
    no_rag: bool = False,
) -> tuple[DiagnosisResult, float]:
    """Run one inference step and return (DiagnosisResult, p_abnormal).

    Mirrors pipeline.diagnose() but:
      - Accepts a PIL Image directly (not a file path)
      - Uses return_dict_in_generate + output_scores to capture first-token logits
      - Supports no_rag=True for RAG ablation (empty snippets)
    """
    device = pipeline.llm.device

    # Vision → projector
    pixel_values = pipeline.vision.preprocess(image).to(device)
    vision_tokens = pipeline.vision(pixel_values)
    visual_embeds = pipeline.projector(vision_tokens)

    # Retrieval
    if no_rag:
        snippets: list[str] = []
        retrieved_list = []
    else:
        query = pipeline._build_retrieval_query()
        retrieved = pipeline.retriever.query(query, k=pipeline.config.retrieval_top_k)
        snippets = [r.text for r in retrieved]
        retrieved_list = retrieved

    # Build prompt — identical to inference (SRS §12)
    messages = build_chat_messages(snippets)
    prompt_text = pipeline.llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs_embeds, attention_mask = pipeline._splice_visual(prompt_text, visual_embeds)

    gen_kwargs: dict = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=pipeline.config.max_new_tokens,
        do_sample=pipeline.config.do_sample,
        pad_token_id=pipeline.llm.tokenizer.pad_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if pipeline.config.do_sample:
        gen_kwargs["temperature"] = pipeline.config.temperature

    output = pipeline.llm.model.generate(**gen_kwargs)
    raw_text = pipeline.llm.tokenizer.decode(output.sequences[0], skip_special_tokens=True)

    # Extract probability from first-token logits
    p_abn = 0.5
    if output.scores:
        try:
            p_abn = _extract_p_abnormal(output.scores, pipeline.llm.tokenizer)
        except Exception:
            pass  # fall back to 0.5 — still get binary metrics

    parsed = parse_output(raw_text)
    diag = DiagnosisResult(
        diagnosis=parsed["diagnosis"],
        evidence_used=parsed["evidence_used"],
        reasoning=parsed["reasoning"],
        retrieved=retrieved_list,
        raw_output=raw_text,
    )
    return diag, p_abn


def run_eval_stream(
    pipeline: MeddiagPipeline,
    labeled_stream: Iterator[LabeledPair],
    max_samples: int = 200,
    no_rag: bool = False,
) -> list[ScoredResult]:
    """Evaluate pipeline over a labeled stream.

    Args:
        pipeline:       Loaded MeddiagPipeline (model + FAISS ready).
        labeled_stream: Yields LabeledPair(image, report, label, source).
        max_samples:    Cap total samples evaluated.
        no_rag:         If True, pass empty snippets (RAG ablation mode).

    Returns:
        List of ScoredResult, one per successfully evaluated sample.
    """
    results: list[ScoredResult] = []
    device = pipeline.llm.device
    is_cuda = hasattr(device, "type") and device.type == "cuda"

    for i, pair in enumerate(labeled_stream):
        if i >= max_samples:
            break

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            mem_before = torch.cuda.memory_allocated(device)

        t0 = time.perf_counter()
        try:
            diag, p_abn = _diagnose_scored(pipeline, pair.image, no_rag=no_rag)
        except Exception as exc:
            print(f"  [{i+1:4d}] ERROR: {type(exc).__name__}: {exc} — skipped")
            continue

        latency = time.perf_counter() - t0
        vram_peak_gb = 0.0
        if is_cuda:
            vram_peak_gb = (
                torch.cuda.max_memory_allocated(device) - mem_before
            ) / (1024 ** 3)

        scored = ScoredResult(
            true_label=pair.label,
            pred_label=diag.diagnosis,
            p_abnormal=p_abn,
            evidence_used=diag.evidence_used,
            reasoning=diag.reasoning,
            latency_s=round(latency, 3),
            vram_peak_gb=round(vram_peak_gb, 3),
            source=pair.source,
        )
        results.append(scored)

        tick = "OK" if scored.correct else "XX"
        print(
            f"  [{i+1:4d}] {tick} true={pair.label:<8} pred={diag.diagnosis:<12} "
            f"p_abn={p_abn:.3f}  t={latency:.1f}s"
        )

    return results
