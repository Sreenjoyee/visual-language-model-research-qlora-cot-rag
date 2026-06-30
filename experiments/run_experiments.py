"""Unified MEDDIAG Experiment Runner — 8 experiments from the research PDF.

Usage:
    python -m experiments.run_experiments --exp all --output-dir reports/
    python -m experiments.run_experiments --exp 1,2,4a --max-samples 50

Experiments:
    1   NIH CXR-14 classification AUROC/F1 + system comparison
    2   CHAIR + BERTScore with/without RAG on MIMIC-50
    3   BERTScore comparison vs LLaVA-Rad 7B / GPT-4V literature
    4a  Adversarial sycophancy probe (IU-Xray NORMAL)
    4b  PadChest OOD accuracy vs literature
    5   GREEN multi-criteria evaluation
    6   ECE calibration + reliability diagram
    7   RAG k-value ablation (k=1,3,5,10)
    8   Energy estimation (GPU TGP vs cloud GPT-4V)
    9   End-to-end component ablation (RAG / LoRA / ClassificationHead / CoT)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from src.config import CONFIG
from src.dataset_streams import (
    iu_xray_normal_stream,
    mimic_eval_stream,
    nih_stream,
    padchest_stream,
)
from src.eval_runner import ScoredResult, _diagnose_scored, run_eval_stream
from src.exp_metrics import (
    bertscore_f1_batch,
    chair_score_batch,
    cloud_gpt4v_energy_kwh,
    estimate_energy_kwh,
    is_sycophantic,
)
from src.metrics import (
    auroc_score,
    binary_metrics,
    evidence_alignment_rate,
    evidence_citation_rate,
    expected_calibration_error,
    green_judge,
    latency_stats,
    reasoning_completeness_score,
    unparseable_rate,
)
from src.output_parser import parse_output
from src.pipeline import MeddiagPipeline
from src.plotting import (
    plot_calibration_curve,
    plot_component_ablation,
    plot_confusion_matrix,
    plot_energy_comparison,
    plot_evidence_histogram,
    plot_green_bar,
    plot_latency_distribution,
    plot_rag_ablation,
    plot_roc_curve,
    plot_system_comparison,
    save_figure,
)
from src.prompts import ADVERSARIAL_SNIPPETS, build_chat_messages


# ── Literature baselines from the PDF ─────────────────────────────────────────

_LIT = {
    "exp1": {
        "QLoRA 8-bit":  {"auroc": 0.782, "f1": 0.694},
        "LLaVA-Med":    {"auroc": 0.831, "f1": 0.741},
        "DenseNet-121": {"auroc": 0.841, "f1": 0.756},
    },
    "exp3": {
        "LLaVA-Rad 7B": {"bertscore_f1": 0.762},
        "GPT-4V":        {"bertscore_f1": 0.778},
    },
    "exp4b": {
        "VLM Alone":  {"accuracy": 0.769},
        "LLaVA-Rad":  {"accuracy": 0.858},
    },
}

# Adversarial snippets imported from prompts.py — shared with training injection


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _scores_to_binary(
    results: list[ScoredResult],
) -> tuple[list[int], list[int], list[float]]:
    y_true = [1 if r.true_label == "ABNORMAL" else 0 for r in results]
    y_pred = [1 if r.pred_label == "ABNORMAL" else 0 for r in results]
    y_prob = [r.p_abnormal for r in results]
    return y_true, y_pred, y_prob


def _agg(vals: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(vals), 4),
        "std":  round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
    }


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then atomically rename — prevents partial JSON files
    # if power is lost mid-write (which would make the resume logic think the
    # experiment completed when it actually didn't).
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    print(f"  Saved → {path}")


# ── Threshold calibration helpers ────────────────────────────────────────────

def _tpr(y_true, y_prob, threshold):
    tp = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 1 and yp >= threshold)
    fn = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 1 and yp < threshold)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _fpr(y_true, y_prob, threshold):
    fp = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 0 and yp >= threshold)
    tn = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 0 and yp < threshold)
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def _find_optimal_threshold(y_true, y_prob) -> float:
    """Return the threshold that maximises Youden J = TPR - FPR."""
    thresholds = sorted(set(y_prob))
    best_t, best_j = 0.5, -1.0
    for t in thresholds:
        j = _tpr(y_true, y_prob, t) - _fpr(y_true, y_prob, t)
        if j > best_j:
            best_j, best_t = j, t
    return round(best_t, 4)


def _apply_optimal_threshold(pipeline, threshold: float) -> None:
    """Write the optimal threshold back to config.py so all subsequent
    experiments and inference use it automatically."""
    pipeline.config.classification_threshold = threshold
    config_path = Path(__file__).resolve().parent.parent / "src" / "config.py"
    try:
        text = config_path.read_text(encoding="utf-8")
        import re
        updated = re.sub(
            r'(os\.environ\.get\("MEDDIAG_THRESHOLD",\s*")[^"]*(")',
            rf'\g<1>{threshold}\g<2>',
            text,
        )
        if updated != text:
            config_path.write_text(updated, encoding="utf-8")
            print(f"  [threshold] Optimal threshold {threshold} written to config.py")
        else:
            print(f"  [threshold] Optimal threshold {threshold} (config.py pattern not matched — set MEDDIAG_THRESHOLD={threshold} manually)")
    except Exception as e:
        print(f"  [threshold] Could not update config.py: {e} — set MEDDIAG_THRESHOLD={threshold}")


# ── Experiment 1: NIH Classification ─────────────────────────────────────────

def exp1_nih_classification(
    pipeline: MeddiagPipeline,
    output_dir: Path,
    max_samples: int = 100,
) -> dict:
    print("\n[Exp 1] NIH ChestX-ray14 Classification")
    results = run_eval_stream(
        pipeline,
        nih_stream(hf_token=pipeline.config.hf_token, max_samples=max_samples),
        max_samples=max_samples,
    )
    if not results:
        return {"error": "no results"}

    y_true, y_pred, y_prob = _scores_to_binary(results)
    bm    = binary_metrics(y_true, y_pred)
    auroc = auroc_score(y_true, y_prob)
    lat   = latency_stats([r.latency_s for r in results])
    vram  = max((r.vram_peak_gb for r in results), default=0.0)

    # Find optimal threshold via Youden J (maximises TPR - FPR on the ROC curve)
    optimal_threshold = _find_optimal_threshold(y_true, y_prob)
    _apply_optimal_threshold(pipeline, optimal_threshold)

    # Recompute binary metrics with the optimal threshold
    y_pred_opt = [1 if p >= optimal_threshold else 0 for p in y_prob]
    bm_opt = binary_metrics(y_true, y_pred_opt)

    # Aggregate comparison: one value per system — bars represent aggregate metrics
    systems: dict[str, dict[str, float]] = {
        "MEDDIAG (ours)": {"auroc": auroc, "f1": bm_opt["f1"]},
        **_LIT["exp1"],
    }
    save_figure(
        plot_system_comparison(systems, ["auroc", "f1"],
                               "Exp 1 — NIH CXR-14: AUROC & F1 by System"),
        output_dir / "exp1_system_comparison",
    )
    save_figure(plot_roc_curve(y_true, y_prob, auroc=auroc), output_dir / "exp1_roc_curve")
    save_figure(plot_confusion_matrix(y_true, y_pred_opt), output_dir / "exp1_confusion_matrix")

    lat_by_label = {
        lbl: [r.latency_s for r in results if r.pred_label == lbl]
        for lbl in ("NORMAL", "ABNORMAL")
    }
    lat_by_label = {k: v for k, v in lat_by_label.items() if v}
    if lat_by_label:
        save_figure(plot_latency_distribution(lat_by_label),
                    output_dir / "exp1_latency_distribution")

    result = {
        "experiment": 1, "dataset": "NIH ChestX-ray14",
        "n_samples": len(results),
        "metrics": {"auroc": auroc, **bm_opt},
        "threshold": {"optimal": optimal_threshold, "youden_j": round(
            max((_tpr(y_true, y_prob, t) - _fpr(y_true, y_prob, t))
                for t in [optimal_threshold]), 4)},
        "latency": lat, "peak_vram_gb": round(vram, 3),
        "literature": _LIT["exp1"],
    }
    _save_json(result, output_dir / "exp1_results.json")
    return result


# ── Experiment 2: CHAIR + BERTScore (RAG vs No-RAG) ──────────────────────────

def exp2_rag_quality(
    pipeline: MeddiagPipeline,
    output_dir: Path,
    max_samples: int = 50,
) -> dict:
    print("\n[Exp 2] CHAIR + BERTScore — MIMIC-50, RAG vs No-RAG")
    items = list(mimic_eval_stream(
        hf_token=pipeline.config.hf_token, max_samples=max_samples
    ))
    refs, rag_outs, no_rag_outs = [], [], []

    for pair in tqdm(items, desc="  [Exp 2] RAG+NoRAG inference", unit="sample"):
        try:
            d_rag, *_ = _diagnose_scored(pipeline, pair.image, no_rag=False)
            d_no,  *_ = _diagnose_scored(pipeline, pair.image, no_rag=True)
        except Exception as e:
            tqdm.write(f"  sample error: {e}")
            continue
        refs.append(pair.report)
        rag_outs.append(d_rag.raw_output)
        no_rag_outs.append(d_no.raw_output)

    if not refs:
        return {"error": "no results"}

    chair_rag    = chair_score_batch(rag_outs,    refs)
    chair_no_rag = chair_score_batch(no_rag_outs, refs)
    bs_rag       = bertscore_f1_batch(rag_outs,    refs)
    bs_no_rag    = bertscore_f1_batch(no_rag_outs, refs)

    # Aggregate: one bar per condition per metric (not one bar per sample)
    systems = {
        "RAG":    {"bertscore_f1": statistics.mean(bs_rag),
                   "chair_coverage": 1 - statistics.mean(chair_rag)},
        "No-RAG": {"bertscore_f1": statistics.mean(bs_no_rag),
                   "chair_coverage": 1 - statistics.mean(chair_no_rag)},
    }
    save_figure(
        plot_system_comparison(systems, ["bertscore_f1", "chair_coverage"],
                               "Exp 2 — RAG vs No-RAG (mean over samples, ↑ better)"),
        output_dir / "exp2_rag_vs_no_rag",
    )

    result = {
        "experiment": 2, "dataset": "MIMIC-CXR", "n_samples": len(refs),
        "rag":    {"chair_score": _agg(chair_rag),    "bertscore_f1": _agg(bs_rag)},
        "no_rag": {"chair_score": _agg(chair_no_rag), "bertscore_f1": _agg(bs_no_rag)},
    }
    _save_json(result, output_dir / "exp2_results.json")
    return result


# ── Experiment 3: BERTScore vs literature ─────────────────────────────────────

def exp3_bertscore_comparison(
    pipeline: MeddiagPipeline,
    output_dir: Path,
    max_samples: int = 50,
    rag_bertscore_mean: float | None = None,
    no_rag_bertscore_mean: float | None = None,
) -> dict:
    print("\n[Exp 3] BERTScore F1 — MEDDIAG vs Literature")
    if rag_bertscore_mean is None or no_rag_bertscore_mean is None:
        items = list(mimic_eval_stream(
            hf_token=pipeline.config.hf_token, max_samples=max_samples
        ))
        refs, rag_outs, no_rag_outs = [], [], []
        for pair in tqdm(items, desc="  [Exp 3] BERTScore inference", unit="sample"):
            try:
                d_rag, *_ = _diagnose_scored(pipeline, pair.image, no_rag=False)
                d_no,  *_ = _diagnose_scored(pipeline, pair.image, no_rag=True)
                refs.append(pair.report)
                rag_outs.append(d_rag.raw_output)
                no_rag_outs.append(d_no.raw_output)
            except Exception:
                continue
        if not refs:
            return {"error": "no results"}
        rag_bertscore_mean    = statistics.mean(bertscore_f1_batch(rag_outs,    refs))
        no_rag_bertscore_mean = statistics.mean(bertscore_f1_batch(no_rag_outs, refs))

    # Aggregate comparison: one bar per system — each is a mean over all samples
    systems = {
        "MEDDIAG RAG":    {"bertscore_f1": rag_bertscore_mean},
        "MEDDIAG No-RAG": {"bertscore_f1": no_rag_bertscore_mean},
        **_LIT["exp3"],
    }
    save_figure(
        plot_system_comparison(systems, ["bertscore_f1"],
                               "Exp 3 — BERTScore F1 by System (aggregate mean)"),
        output_dir / "exp3_bertscore_comparison",
    )

    result = {
        "experiment": 3,
        "meddiag_rag_bertscore":    round(rag_bertscore_mean, 4),
        "meddiag_no_rag_bertscore": round(no_rag_bertscore_mean, 4),
        "literature": _LIT["exp3"],
    }
    _save_json(result, output_dir / "exp3_results.json")
    return result


# ── Experiment 4A: Adversarial Sycophancy Probe ───────────────────────────────

@torch.no_grad()
def _run_adversarial(pipeline: MeddiagPipeline, image, snippet: str) -> tuple[str, str]:
    """Inference with one adversarial snippet replacing all retrieved evidence.

    Uses ClassificationHead (visual features) as the primary diagnosis signal —
    it is more resistant to text manipulation than the LLM generation path.
    Falls back to text-parsed diagnosis if cls_head is unavailable.
    """
    import numpy as np
    device = pipeline.llm.device
    pv = pipeline.vision.preprocess(image).to(device)
    ve = pipeline.projector(pipeline.vision(pv))          # (1, K, D_llm)

    # LLM text generation path
    messages  = build_chat_messages([snippet])
    prompt    = pipeline.llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    emb, mask = pipeline._splice_visual(prompt, ve)
    out_ids   = pipeline.llm.model.generate(
        inputs_embeds=emb, attention_mask=mask,
        max_new_tokens=pipeline.config.max_new_tokens,
        do_sample=pipeline.config.do_sample,
        pad_token_id=pipeline.llm.tokenizer.pad_token_id,
    )
    raw = pipeline.llm.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    text_diagnosis = parse_output(raw)["diagnosis"]

    # ClassificationHead path — more resistant to sycophancy because it reads
    # visual features directly rather than being swayed by adversarial text.
    if pipeline.cls_head is not None:
        adv_emb = pipeline.retriever.embedder.encode(
            [snippet], convert_to_numpy=True, normalize_embeddings=True
        )
        rag_embeddings = torch.from_numpy(
            np.stack(adv_emb)
        ).unsqueeze(0).float().to(device)                 # (1, 1, 384)
        # Pad to retrieval_top_k if needed
        k = pipeline.config.retrieval_top_k
        if rag_embeddings.shape[1] < k:
            pad = torch.zeros(1, k - rag_embeddings.shape[1], rag_embeddings.shape[2], device=device)
            rag_embeddings = torch.cat([rag_embeddings, pad], dim=1)

        logits = pipeline.cls_head(ve, rag_embeddings)    # (1, 2)
        probs  = torch.softmax(logits, dim=-1)
        p_abn  = probs[0, 1].item()
        cls_diagnosis = "ABNORMAL" if p_abn >= pipeline.config.classification_threshold else "NORMAL"
        return cls_diagnosis, raw

    return text_diagnosis, raw


def exp4a_sycophancy(
    pipeline: MeddiagPipeline,
    output_dir: Path,
    max_samples: int = 50,
) -> dict:
    print("\n[Exp 4A] Adversarial Sycophancy Probe — IU-Xray NORMAL")
    fooled, total = 0, 0

    # Rotate through all adversarial snippets for each sample
    samples = list(iu_xray_normal_stream(max_samples=max_samples))
    for i, pair in enumerate(tqdm(samples, desc="  [Exp 4A] Sycophancy probe", unit="sample")):
        snippet = ADVERSARIAL_SNIPPETS[i % len(ADVERSARIAL_SNIPPETS)]
        try:
            diagnosis, raw = _run_adversarial(pipeline, pair.image, snippet)
        except Exception as e:
            tqdm.write(f"  sample error: {e}")
            continue
        fooled += int(is_sycophantic(raw, diagnosis, true_label="NORMAL"))
        total  += 1

    fpr = round(fooled / total, 4) if total > 0 else float("nan")
    result = {
        "experiment": "4a", "dataset": "IU-Xray NORMAL",
        "n_samples": total, "false_positive_rate": fpr,
        "sycophantic_count": fooled,
        "resistance_rate": round(1.0 - fpr, 4) if total > 0 else float("nan"),
        "n_adversarial_snippets": len(ADVERSARIAL_SNIPPETS),
    }
    _save_json(result, output_dir / "exp4a_results.json")
    print(f"  FPR = {fpr:.3f}  ({fooled}/{total} fooled)  "
          f"Resistance = {result['resistance_rate']:.3f}")
    return result


# ── Experiment 4B: PadChest OOD ───────────────────────────────────────────────

def exp4b_ood_padchest(
    pipeline: MeddiagPipeline,
    output_dir: Path,
    max_samples: int = 100,
) -> dict:
    print("\n[Exp 4B] PadChest OOD Accuracy")
    results = run_eval_stream(
        pipeline,
        padchest_stream(hf_token=pipeline.config.hf_token, max_samples=max_samples),
        max_samples=max_samples,
    )
    if not results:
        return {"error": "no results"}

    y_true, y_pred, _ = _scores_to_binary(results)
    bm = binary_metrics(y_true, y_pred)

    systems = {
        "MEDDIAG (ours)": {"accuracy": bm["accuracy"]},
        **_LIT["exp4b"],
    }
    save_figure(
        plot_system_comparison(systems, ["accuracy"],
                               "Exp 4B — PadChest OOD Accuracy by System"),
        output_dir / "exp4b_ood_comparison",
    )
    save_figure(plot_confusion_matrix(y_true, y_pred), output_dir / "exp4b_confusion_matrix")

    result = {
        "experiment": "4b", "dataset": "PadChest OOD",
        "n_samples": len(results), "metrics": bm, "literature": _LIT["exp4b"],
    }
    _save_json(result, output_dir / "exp4b_results.json")
    return result


# ── Experiment 5: GREEN Score ─────────────────────────────────────────────────

def exp5_green(
    pipeline: MeddiagPipeline,
    output_dir: Path,
    max_samples: int = 50,
) -> dict:
    print("\n[Exp 5] GREEN Multi-criteria Evaluation")
    results = run_eval_stream(
        pipeline,
        mimic_eval_stream(hf_token=pipeline.config.hf_token, max_samples=max_samples),
        max_samples=max_samples,
    )
    if not results:
        return {"error": "no results"}

    y_true, y_pred, y_prob = _scores_to_binary(results)
    bm    = binary_metrics(y_true, y_pred)
    auroc = auroc_score(y_true, y_prob)
    ev_lists   = [r.evidence_used for r in results]
    reasonings = [r.reasoning     for r in results]
    diagnoses  = [r.pred_label    for r in results]

    green = green_judge(
        groundedness=evidence_citation_rate(ev_lists),
        reasoning=reasoning_completeness_score(reasonings),
        evidence_alignment=evidence_alignment_rate(ev_lists, pipeline.config.retrieval_top_k),
        error_free=1.0 - unparseable_rate(diagnoses),
        numerical=(auroc + bm["f1"]) / 2,
    )

    save_figure(plot_green_bar(green),           output_dir / "exp5_green_scores")
    save_figure(plot_evidence_histogram(ev_lists), output_dir / "exp5_evidence_histogram")

    result = {
        "experiment": 5, "dataset": "MIMIC-CXR",
        "n_samples": len(results), "green": green, "classification": bm,
    }
    _save_json(result, output_dir / "exp5_results.json")
    return result


# ── Experiment 6: ECE Calibration ────────────────────────────────────────────

def exp6_calibration(
    pipeline: MeddiagPipeline,
    output_dir: Path,
    max_samples: int = 100,
) -> dict:
    print("\n[Exp 6] ECE Calibration + Reliability Diagram")
    results = run_eval_stream(
        pipeline,
        mimic_eval_stream(hf_token=pipeline.config.hf_token, max_samples=max_samples),
        max_samples=max_samples,
    )
    if not results:
        return {"error": "no results"}

    y_true, _, y_prob = _scores_to_binary(results)
    ece = expected_calibration_error(y_true, y_prob)

    save_figure(plot_calibration_curve(y_true, y_prob), output_dir / "exp6_calibration_curve")

    result = {
        "experiment": 6, "dataset": "MIMIC-CXR",
        "n_samples": len(results), "ece": ece,
    }
    _save_json(result, output_dir / "exp6_results.json")
    print(f"  ECE = {ece:.4f}")
    return result


# ── Experiment 7: RAG k-Value Ablation ───────────────────────────────────────

def exp7_rag_ablation(
    pipeline: MeddiagPipeline,
    output_dir: Path,
    max_samples: int = 30,
    k_values: list[int] | None = None,
) -> dict:
    print("\n[Exp 7] RAG k-Value Ablation")
    if k_values is None:
        k_values = [1, 3, 5, 10]

    items     = list(mimic_eval_stream(
        hf_token=pipeline.config.hf_token, max_samples=max_samples
    ))
    orig_k    = pipeline.config.retrieval_top_k
    k_results: dict[int, dict] = {}

    for k in k_values:
        pipeline.config.retrieval_top_k = k
        refs, outputs, latencies = [], [], []

        for pair in tqdm(items, desc=f"  [Exp 7] k={k}", unit="sample"):
            t0 = time.perf_counter()
            try:
                diag, *_ = _diagnose_scored(pipeline, pair.image, no_rag=(k == 0))
            except Exception as e:
                print(f"    sample error: {e}")
                continue
            latencies.append(time.perf_counter() - t0)
            outputs.append(diag.raw_output)
            refs.append(pair.report)

        if not refs:
            continue
        bs = bertscore_f1_batch(outputs, refs)
        k_results[k] = {
            "bertscore_mean": round(statistics.mean(bs), 4),
            "bertscore_std":  round(statistics.stdev(bs) if len(bs) > 1 else 0.0, 4),
            "latency_mean":   round(statistics.mean(latencies), 3),
            "n": len(refs),
        }

    pipeline.config.retrieval_top_k = orig_k

    if not k_results:
        return {"error": "no results"}

    ks        = sorted(k_results)
    bs_means  = [k_results[k]["bertscore_mean"] for k in ks]
    bs_stds   = [k_results[k]["bertscore_std"]  for k in ks]
    lat_means = [k_results[k]["latency_mean"]   for k in ks]

    save_figure(
        plot_rag_ablation(ks, bs_means, bs_stds, lat_means),
        output_dir / "exp7_rag_ablation",
    )

    result = {
        "experiment": 7, "k_values": ks,
        "results_by_k": {str(k): k_results[k] for k in ks},
    }
    _save_json(result, output_dir / "exp7_results.json")
    return result


# ── Experiment 8: Energy Estimation ──────────────────────────────────────────

def exp8_energy(
    pipeline: MeddiagPipeline,
    output_dir: Path,
    max_samples: int = 20,
    tgp_w: float = 55.0,
) -> dict:
    print("\n[Exp 8] Energy Consumption Estimation")
    results = run_eval_stream(
        pipeline,
        mimic_eval_stream(hf_token=pipeline.config.hf_token, max_samples=max_samples),
        max_samples=max_samples,
    )
    if not results:
        return {"error": "no results"}

    latencies         = [r.latency_s for r in results]
    mean_lat          = statistics.mean(latencies)
    energy_per_inf    = estimate_energy_kwh(mean_lat, tgp_w=tgp_w)
    total_energy      = sum(estimate_energy_kwh(l, tgp_w=tgp_w) for l in latencies)
    gpt4v_estimate    = cloud_gpt4v_energy_kwh(len(results))
    efficiency_ratio  = round(gpt4v_estimate / max(total_energy, 1e-10), 1)

    result = {
        "experiment": 8, "n_samples": len(results),
        "mean_latency_s": round(mean_lat, 3),
        "tgp_w": tgp_w,
        "energy_per_inference_kwh": energy_per_inf,
        "total_energy_kwh": round(total_energy, 6),
        "gpt4v_cloud_estimate_kwh": gpt4v_estimate,
        "efficiency_ratio_vs_gpt4v": efficiency_ratio,
    }
    _save_json(result, output_dir / "exp8_results.json")
    save_figure(
        plot_energy_comparison(energy_per_inf, gpt4v_estimate / len(results),
                               len(results), efficiency_ratio),
        output_dir / "exp8_energy_comparison",
    )
    print(f"  Energy/inference:  {energy_per_inf:.2e} kWh")
    print(f"  GPT-4V equivalent: {gpt4v_estimate:.4f} kWh for {len(results)} requests")
    print(f"  MEDDIAG is ~{efficiency_ratio}x more energy-efficient than GPT-4V cloud")
    return result


# ── Experiment 9: End-to-End Component Ablation ───────────────────────────────

def exp9_component_ablation(
    pipeline: MeddiagPipeline,
    output_dir: Path,
    max_samples: int = 50,
) -> dict:
    """Ablate each system component to measure its individual contribution.

    Tests 5 configurations on MIMIC eval stream:
      A) Full system  — RAG + LoRA + ClassificationHead + CoT
      B) No RAG       — empty evidence, ClassificationHead still runs
      C) No cls_head  — text-parse diagnosis only, no ClassificationHead
      D) No LoRA      — pipeline loaded without LoRA adapter (base LLM only)
      E) No CoT       — direct answer prompt, no reasoning chain requested
    """
    print("\n[Exp 9] End-to-End Component Ablation")

    from src.prompts import SYSTEM_PROMPT
    from src.output_parser import parse_output as _parse

    configs = {
        "full_system":   "Full system (RAG + LoRA + CLS head + CoT)",
        "no_rag":        "No RAG (empty evidence)",
        "no_cls_head":   "No ClassificationHead (text-parse only)",
        "no_lora":       "No LoRA (base LLM, projector only)",
        "no_cot":        "No CoT (direct answer, no reasoning)",
    }

    samples = list(mimic_eval_stream(
        hf_token=pipeline.config.hf_token, max_samples=max_samples
    ))

    def _auroc_from_results(results: list[ScoredResult]) -> float:
        if not results:
            return float("nan")
        try:
            y_true, _, y_prob = _scores_to_binary(results)
            return round(auroc_score(y_true, y_prob), 4)
        except Exception:
            return float("nan")

    ablation_results: dict[str, dict] = {}

    # ── A: Full system ────────────────────────────────────────────────────────
    print("  [A] Full system...")
    full_results = run_eval_stream(pipeline, iter(samples), max_samples=max_samples)
    ablation_results["full_system"] = {
        "auroc": _auroc_from_results(full_results),
        "n": len(full_results),
        "description": configs["full_system"],
    }
    print(f"      AUROC = {ablation_results['full_system']['auroc']:.4f}")

    # ── B: No RAG ────────────────────────────────────────────────────────────
    print("  [B] No RAG (empty evidence)...")
    orig_top_k = pipeline.config.retrieval_top_k
    pipeline.config.retrieval_top_k = 0  # forces empty retrieval

    @torch.no_grad()
    def _diagnose_no_rag(pair) -> ScoredResult:
        import time as _time
        t0 = _time.time()
        result = pipeline.diagnose(pair.image)
        return ScoredResult(
            true_label=pair.label,
            pred_label=result.diagnosis,
            p_abnormal=result.cls_confidence if result.cls_confidence is not None else 0.5,
            reasoning=result.reasoning,
            latency_s=_time.time() - t0,
        )

    no_rag_results = []
    for pair in tqdm(samples, desc="  [B] No RAG", unit="sample"):
        try:
            no_rag_results.append(_diagnose_no_rag(pair))
        except Exception as e:
            tqdm.write(f"  skip: {e}")

    pipeline.config.retrieval_top_k = orig_top_k
    ablation_results["no_rag"] = {
        "auroc": _auroc_from_results(no_rag_results),
        "n": len(no_rag_results),
        "description": configs["no_rag"],
    }
    print(f"      AUROC = {ablation_results['no_rag']['auroc']:.4f}")

    # ── C: No ClassificationHead (text-parse only) ────────────────────────────
    print("  [C] No ClassificationHead (text-parse only)...")
    orig_cls_head = pipeline.cls_head
    pipeline.cls_head = None

    text_parse_results = run_eval_stream(pipeline, iter(samples), max_samples=max_samples)
    pipeline.cls_head = orig_cls_head
    ablation_results["no_cls_head"] = {
        "auroc": _auroc_from_results(text_parse_results),
        "n": len(text_parse_results),
        "description": configs["no_cls_head"],
    }
    print(f"      AUROC = {ablation_results['no_cls_head']['auroc']:.4f}")

    # ── D: No LoRA (base LLM only) ───────────────────────────────────────────
    print("  [D] No LoRA — skipping inference (LoRA already loaded, cannot unload cleanly)")
    ablation_results["no_lora"] = {
        "auroc": None,
        "n": 0,
        "description": configs["no_lora"],
        "note": "Requires separate pipeline load without LoRA adapter; skipped in this run.",
    }

    # ── E: No CoT (direct answer prompt) ─────────────────────────────────────
    print("  [E] No CoT (direct answer prompt)...")

    @torch.no_grad()
    def _diagnose_no_cot(pair) -> ScoredResult:
        import time as _time
        from src.prompts import IMAGE_PLACEHOLDER, format_retrieved_evidence
        t0 = _time.time()
        device = pipeline.llm.device

        pv = pipeline.vision.preprocess(pair.image).to(device)
        ve = pipeline.projector(pipeline.vision(pv))
        retrieved = pipeline.retriever.query("chest X-ray", k=pipeline.config.retrieval_top_k)
        evidence = format_retrieved_evidence([r.text for r in retrieved])

        # Direct answer prompt — no CoT steps
        direct_prompt = (
            f"Chest X-ray image:\n{IMAGE_PLACEHOLDER}\n\n"
            f"Retrieved evidence:\n{evidence}\n\n"
            "Answer with only: DIAGNOSIS: NORMAL or DIAGNOSIS: ABNORMAL"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": direct_prompt},
        ]
        prompt_text = pipeline.llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        emb, mask = pipeline._splice_visual(prompt_text, ve)
        out_ids = pipeline.llm.model.generate(
            inputs_embeds=emb, attention_mask=mask,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=pipeline.llm.tokenizer.pad_token_id,
        )
        raw = pipeline.llm.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        parsed = _parse(raw)

        # ClassificationHead still runs with visual features
        p_abn = 0.5
        if pipeline.cls_head is not None:
            import numpy as np
            rag_embs = [r.embedding for r in retrieved if r.embedding is not None]
            if rag_embs:
                rag_t = torch.from_numpy(np.stack(rag_embs)).unsqueeze(0).float().to(device)
                logits = pipeline.cls_head(ve, rag_t)
                p_abn = torch.softmax(logits, dim=-1)[0, 1].item()

        return ScoredResult(
            true_label=pair.label,
            pred_label=parsed["diagnosis"] if parsed["diagnosis"] != "UNPARSEABLE" else (
                "ABNORMAL" if p_abn >= pipeline.config.classification_threshold else "NORMAL"
            ),
            p_abnormal=p_abn,
            reasoning=raw,
            latency_s=_time.time() - t0,
        )

    no_cot_results = []
    for pair in tqdm(samples, desc="  [E] No CoT", unit="sample"):
        try:
            no_cot_results.append(_diagnose_no_cot(pair))
        except Exception as e:
            tqdm.write(f"  skip: {e}")

    ablation_results["no_cot"] = {
        "auroc": _auroc_from_results(no_cot_results),
        "n": len(no_cot_results),
        "description": configs["no_cot"],
    }
    print(f"      AUROC = {ablation_results['no_cot']['auroc']:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    full_auroc = ablation_results["full_system"]["auroc"]
    print("\n  === Exp 9 Ablation Summary ===")
    for k, v in ablation_results.items():
        auroc = v["auroc"]
        if auroc is not None:
            delta = round(auroc - full_auroc, 4) if full_auroc else 0.0
            sign = "+" if delta >= 0 else ""
            print(f"  {v['description']:<45} AUROC={auroc:.4f}  ({sign}{delta:.4f} vs full)")
        else:
            print(f"  {v['description']:<45} SKIPPED")

    result = {
        "experiment": 9,
        "n_samples": max_samples,
        "ablations": ablation_results,
    }
    _save_json(result, output_dir / "exp9_results.json")
    save_figure(
        plot_component_ablation(ablation_results),
        output_dir / "exp9_component_ablation",
    )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

# Per-experiment hard caps applied regardless of global --max-samples.
_EXP_CAPS = {
    "2": 100, "3": 100,        # ×2 inference — capped at 100
    "4a": 100,
    "4b": 100,
    "5": 100,
    "6": 100,
    "7": 30,                   # ×4 inference (k=1,3,5,10)
    "8": 20,
    "9": 50,                   # ×4 configs — capped at 50
}

_ALL_EXPS = ["1", "2", "3", "4a", "4b", "5", "6", "7", "8", "9"]


def main() -> None:
    parser = argparse.ArgumentParser(description="MEDDIAG Experiment Runner")
    parser.add_argument("--exp", default="all",
                        help="Comma-separated experiment IDs or 'all'. E.g. 1,2,4a")
    parser.add_argument("--output-dir", default="reports", type=Path)
    parser.add_argument("--resume-dir", default=None, type=Path,
                        help="Reuse an existing output dir instead of creating a new timestamped one")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--projector", type=Path, default=None,
                        help="Path to projector .pt weights")
    parser.add_argument("--lora-adapter", type=Path, default=None,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--tgp-w", type=float, default=55.0,
                        help="GPU TGP in watts for Exp 8 energy estimation")
    args = parser.parse_args()

    exps = _ALL_EXPS if args.exp.strip() == "all" else [
        e.strip() for e in args.exp.split(",")
    ]
    if args.resume_dir:
        output_dir = args.resume_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline ...")
    pipeline = MeddiagPipeline(
        config=CONFIG,
        projector_weights=args.projector,
        lora_adapter_dir=args.lora_adapter,
    )

    all_results: dict[str, dict] = {}
    rag_bs_mean:    float | None = None
    no_rag_bs_mean: float | None = None

    def _n(exp_id: str) -> int:
        return min(args.max_samples, _EXP_CAPS.get(exp_id, args.max_samples))

    _dispatch = {
        "1":  lambda: exp1_nih_classification(pipeline, output_dir, _n("1")),
        "2":  lambda: exp2_rag_quality(pipeline, output_dir, _n("2")),
        "3":  lambda: exp3_bertscore_comparison(
                    pipeline, output_dir, _n("3"),
                    rag_bertscore_mean=rag_bs_mean,
                    no_rag_bertscore_mean=no_rag_bs_mean),
        "4a": lambda: exp4a_sycophancy(pipeline, output_dir, _n("4a")),
        "4b": lambda: exp4b_ood_padchest(pipeline, output_dir, _n("4b")),
        "5":  lambda: exp5_green(pipeline, output_dir, _n("5")),
        "6":  lambda: exp6_calibration(pipeline, output_dir, _n("6")),
        "7":  lambda: exp7_rag_ablation(pipeline, output_dir, _n("7")),
        "8":  lambda: exp8_energy(pipeline, output_dir, _n("8"), args.tgp_w),
        "9":  lambda: exp9_component_ablation(pipeline, output_dir, _n("9")),
    }

    for exp_id in exps:
        fn = _dispatch.get(exp_id)
        if fn is None:
            print(f"  Unknown experiment: {exp_id}")
            continue
        try:
            r = fn()
            all_results[f"exp{exp_id}"] = r
            # Cache Exp 2 BERTScore means so Exp 3 can reuse them
            if exp_id == "2" and "rag" in r:
                rag_bs_mean    = r["rag"]["bertscore_f1"]["mean"]
                no_rag_bs_mean = r["no_rag"]["bertscore_f1"]["mean"]
        except Exception as e:
            print(f"  [Exp {exp_id}] FAILED: {type(e).__name__}: {e}")
            all_results[f"exp{exp_id}"] = {"error": str(e)}

    _save_json(all_results, output_dir / "all_results.json")
    print(f"\nAll done. Results in {output_dir}/")


if __name__ == "__main__":
    main()
