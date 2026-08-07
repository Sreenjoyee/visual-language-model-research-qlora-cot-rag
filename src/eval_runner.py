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

import numpy as np
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
    p_cls: float | None = None   # cls_head probability (None if head unavailable)
    p_lm: float = 0.5            # LM label-token probability (always available)
    ram_gb: float = 0.0          # process resident RAM (RSS) in GB at sample end
    correct: bool = field(init=False)

    def __post_init__(self) -> None:
        self.correct = (self.true_label == self.pred_label)


def _proc_ram_gb() -> float:
    """Resident RAM (RSS) of this process in GB — the CPU-side memory footprint,
    logged alongside VRAM. Returns 0.0 if psutil is unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0


def _extract_p_abnormal(
    scores: tuple, tokenizer, gen_ids: torch.Tensor
) -> float:
    """Extract P(ABNORMAL) from the step where the label token was generated.

    Scans generated token IDs backwards (the label is at the end of the
    reasoning chain) to find the exact step where NORMAL or ABNORMAL was
    emitted, then reads softmax probability from that step's logits.
    Falls back to first-token logits if the label token is not found.
    """
    def candidate_ids(words: list[str]) -> set[int]:
        ids: set[int] = set()
        for w in words:
            toks = tokenizer.encode(w, add_special_tokens=False)
            if toks:
                ids.add(toks[0])
        return ids

    abn_tids = candidate_ids(["ABNORMAL", " ABNORMAL"])
    norm_tids = candidate_ids(["NORMAL",   " NORMAL"])
    if not abn_tids or not norm_tids:
        return 0.5

    gen_list = gen_ids.tolist()
    n_steps = min(len(gen_list), len(scores))

    label_pos: int | None = None
    for pos in range(n_steps - 1, -1, -1):
        tid = gen_list[pos]
        if tid in abn_tids:
            label_pos = pos
            break
        if tid in norm_tids:
            # Guard: skip "NORMAL" if it is the suffix of "ABNORMAL" (prev token = "AB")
            if pos > 0:
                prev = tokenizer.decode([gen_list[pos - 1]], skip_special_tokens=True)
                if prev.upper().endswith("AB"):
                    continue
            label_pos = pos
            break

    step_logits = scores[label_pos][0] if label_pos is not None else scores[0][0]
    logit_n = max(step_logits[t].item() for t in norm_tids)
    logit_a = max(step_logits[t].item() for t in abn_tids)
    probs = F.softmax(torch.tensor([logit_n, logit_a]), dim=0)
    return float(probs[1].item())


@torch.no_grad()
def _diagnose_scored(
    pipeline: MeddiagPipeline,
    image: Image.Image,
    no_rag: bool = False,
    tta_fns: list | None = None,
) -> tuple[DiagnosisResult, float, float | None, float]:
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
        query = pipeline._caption_image(visual_embeds)
        retrieved = pipeline.retriever.query(query, k=pipeline.config.retrieval_top_k)
        snippets = [r.text for r in retrieved]
        retrieved_list = retrieved

    # Classification head — mirrors pipeline.diagnose() exactly.
    # When the head is available it is the authoritative classification signal;
    # LLaMA generation is used only for reasoning text and evidence citations.
    # In no_rag mode (RAG ablation) there are no retrieved embeddings, so the
    # head cannot run and we fall back to text-parsing p_abnormal.
    cls_diagnosis: str | None = None
    cls_confidence: float | None = None
    if pipeline.cls_head is not None and retrieved_list:
        rag_embs = [r.embedding for r in retrieved_list if r.embedding is not None]
        if not rag_embs:
            print("[warn] cls_head loaded but all retrieved embeddings are None — falling back to text-parse")
        if rag_embs:
            rag_tensor = torch.from_numpy(np.stack(rag_embs)).unsqueeze(0).to(device)
            # Prediction-ensemble: average the softmax over the per-checkpoint cls_head
            # ensemble when present (MEDDIAG_CLS_ENSEMBLE), else the single head.
            heads = getattr(pipeline, "cls_head_ensemble", None) or [pipeline.cls_head]

            def _abn_prob(ve) -> float:
                ps = [torch.softmax(h(ve, rag_tensor), dim=-1)[0, 1].item() for h in heads]
                return sum(ps) / len(ps)

            if tta_fns:
                # #5 TTA × ensemble: average cls_head prob over original + augmented
                # views AND over the checkpoint ensemble. Only the cheap vision→
                # projector→head path is re-run (no extra LLM generation).
                confs: list[float] = []
                for fn in [None] + list(tta_fns):
                    aug_img = image if fn is None else fn(image)
                    pv = pipeline.vision.preprocess(aug_img).to(device)
                    ve = pipeline.projector(pipeline.vision(pv))
                    confs.append(_abn_prob(ve))
                cls_confidence = sum(confs) / len(confs)
            else:
                cls_confidence = _abn_prob(visual_embeds)
            cls_diagnosis = (
                "ABNORMAL"
                if cls_confidence >= pipeline.config.classification_threshold
                else "NORMAL"
            )

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

    # Extract probability from the label-token step (not first token)
    p_abn = 0.5
    if output.scores:
        try:
            p_abn = _extract_p_abnormal(
                output.scores, pipeline.llm.tokenizer, output.sequences[0]
            )
        except (KeyError, IndexError, RuntimeError) as e:
            print(f"[warn] p_abnormal extraction failed: {e} — using 0.5 fallback")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in p_abnormal extraction: {e}") from e

    parsed = parse_output(raw_text)
    # cls_head takes precedence over text-parsing for the binary label.
    final_diagnosis = cls_diagnosis if cls_diagnosis is not None else parsed["diagnosis"]
    diag = DiagnosisResult(
        diagnosis=final_diagnosis,
        evidence_used=parsed["evidence_used"],
        reasoning=parsed["reasoning"],
        retrieved=retrieved_list,
        raw_output=raw_text,
        cls_confidence=cls_confidence,
    )
    # Prefer cls_head probability for AUROC/ECE — it is a proper softmax probability
    # rather than a logit-heuristic extracted from the label token.
    final_p_abn = cls_confidence if cls_confidence is not None else p_abn
    # also return the two raw signals: p_cls (cls_head) and p_lm (LM token prob)
    return diag, final_p_abn, cls_confidence, p_abn


def run_eval_stream(
    pipeline: MeddiagPipeline,
    labeled_stream: Iterator[LabeledPair],
    max_samples: int = 200,
    no_rag: bool = False,
    tta_fns: list | None = None,
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
            # Reset so max_memory_allocated tracks THIS sample's ABSOLUTE peak —
            # resident model weights + activations — i.e. the true GPU footprint a
            # deployment needs, not a marginal delta above the model baseline.
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()
        try:
            diag, p_abn, p_cls, p_lm = _diagnose_scored(
                pipeline, pair.image, no_rag=no_rag, tta_fns=tta_fns
            )
        except Exception as exc:
            print(f"  [{i+1:4d}] ERROR: {type(exc).__name__}: {exc} — skipped")
            continue

        latency = time.perf_counter() - t0
        ram_gb = _proc_ram_gb()
        vram_peak_gb = 0.0
        if is_cuda:
            vram_peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        if diag.diagnosis in ("NORMAL", "ABNORMAL"):
            pred_label = diag.diagnosis
        else:
            pred_label = "ABNORMAL" if p_abn > pipeline.config.classification_threshold else "NORMAL"

        scored = ScoredResult(
            true_label=pair.label,
            pred_label=pred_label,
            p_abnormal=p_abn,
            evidence_used=diag.evidence_used,
            reasoning=diag.reasoning,
            latency_s=round(latency, 3),
            vram_peak_gb=round(vram_peak_gb, 3),
            ram_gb=round(ram_gb, 3),
            source=pair.source,
            p_cls=p_cls,
            p_lm=p_lm,
        )
        results.append(scored)

        tick = "OK" if scored.correct else "XX"
        print(
            f"  [{i+1:4d}] {tick} true={pair.label:<8} pred={pred_label:<12} "
            f"p_abn={p_abn:.3f}  t={latency:.1f}s"
        )

    return results
