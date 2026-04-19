"""End-to-end integration tests for the MEDDIAG pipeline.

All tests here require a downloaded model (MEDDIAG_RUN_MODEL_TESTS=1).
They exercise the full data flow:
    PIL image → VisionEncoder → PerceiverResampler → LLM → parse_output

A tiny FAISS index is built from the static GuidelinesSource (no network
needed once models are downloaded) so these tests are self-contained.

Run with:
    MEDDIAG_RUN_MODEL_TESTS=1 pytest tests/test_integration.py -v
"""
from __future__ import annotations

import time

import pytest
import torch
from PIL import Image

from tests.conftest import needs_model


# ── Session fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_faiss_dir(tmp_path_factory):
    """Build a minimal FAISS index from GuidelinesSource (static, no network)."""
    from src.config import Config
    from src.retrieval import GuidelinesSource, Retriever

    idx_dir = tmp_path_factory.mktemp("faiss_integration")
    cfg = Config()
    cfg.faiss_index_dir = idx_dir

    r = Retriever(cfg)
    r.build([GuidelinesSource()])
    r.save(idx_dir)
    return idx_dir


@pytest.fixture(scope="session")
def pipeline(tiny_faiss_dir):
    """Full MeddiagPipeline with tiny FAISS index — loaded once per session."""
    from src.config import Config
    from src.pipeline import MeddiagPipeline

    cfg = Config()
    cfg.faiss_index_dir = tiny_faiss_dir
    return MeddiagPipeline(config=cfg)


@pytest.fixture
def synthetic_image() -> Image.Image:
    """224×224 random RGB image that simulates a chest X-ray for testing."""
    import numpy as np
    arr = (torch.randn(224, 224, 3) * 30 + 128).clamp(0, 255).byte().numpy()
    return Image.fromarray(arr, mode="RGB")


# ── Vision encoder → projector integration ────────────────────────────────────

@needs_model
def test_vision_tokens_feed_into_projector(pipeline, synthetic_image):
    """Vision encoder output shape is compatible with projector input."""
    device = pipeline.llm.device
    pv = pipeline.vision.preprocess(synthetic_image).to(device)
    vision_tokens = pipeline.vision(pv)              # (1, N, C_vision)
    visual_embeds = pipeline.projector(vision_tokens) # (1, K, D_llm)

    assert visual_embeds.ndim == 3
    B, K, D = visual_embeds.shape
    assert B == 1
    assert K == pipeline.config.num_visual_tokens
    assert D == pipeline.config.llm_hidden_dim


@needs_model
def test_visual_embeds_dtype_matches_llm(pipeline, synthetic_image):
    """Visual embeddings must be castable to LLM dtype without precision loss."""
    device = pipeline.llm.device
    pv     = pipeline.vision.preprocess(synthetic_image).to(device)
    ve     = pipeline.projector(pipeline.vision(pv))
    # After _splice_visual, embeds are cast to the LLM token embedding dtype.
    # Verify the cast completes without error.
    llm_embed = pipeline.llm.model.get_input_embeddings()
    dummy_ids  = torch.zeros(1, 1, dtype=torch.long, device=device)
    ref_dtype  = llm_embed(dummy_ids).dtype
    ve_cast    = ve.to(dtype=ref_dtype)
    assert ve_cast.shape == ve.shape
    assert torch.isfinite(ve_cast).all()


# ── FAISS retrieval integration ───────────────────────────────────────────────

@needs_model
def test_retriever_returns_top_k_snippets(pipeline):
    """Retriever returns exactly retrieval_top_k snippets for any query."""
    snippets = pipeline.retriever.query(
        "chest radiograph findings", k=pipeline.config.retrieval_top_k
    )
    assert len(snippets) == pipeline.config.retrieval_top_k
    assert all(len(s.text) > 10 for s in snippets)


@needs_model
def test_retriever_snippets_are_finite_distances(pipeline):
    snippets = pipeline.retriever.query("pulmonary edema cardiomegaly")
    assert all(s.distance >= 0 for s in snippets)
    assert all(isinstance(s.source, str) for s in snippets)


# ── Prompt splicing ───────────────────────────────────────────────────────────

@needs_model
def test_splice_visual_produces_valid_embeddings(pipeline, synthetic_image):
    """_splice_visual outputs inputs_embeds and attention_mask with matching shapes."""
    from src.prompts import build_chat_messages

    device = pipeline.llm.device
    pv     = pipeline.vision.preprocess(synthetic_image).to(device)
    ve     = pipeline.projector(pipeline.vision(pv))

    messages    = build_chat_messages(["chest is clear"])
    prompt_text = pipeline.llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    embeds, mask = pipeline._splice_visual(prompt_text, ve)

    assert embeds.ndim == 3          # (1, L, D)
    assert mask.shape == embeds.shape[:2]
    assert mask.all()                # all positions attended to
    assert torch.isfinite(embeds).all()


# ── Full pipeline end-to-end ──────────────────────────────────────────────────

@needs_model
def test_diagnose_returns_valid_diagnosis(pipeline, synthetic_image, tmp_path):
    """diagnose() on a synthetic image returns a DiagnosisResult with valid label."""
    img_path = tmp_path / "test.png"
    synthetic_image.save(str(img_path))

    result = pipeline.diagnose(img_path)

    assert result.diagnosis in {"NORMAL", "ABNORMAL", "UNPARSEABLE"}, (
        f"Unexpected diagnosis: {result.diagnosis!r}"
    )
    assert isinstance(result.raw_output, str) and len(result.raw_output) > 0
    assert isinstance(result.retrieved, list)
    assert isinstance(result.evidence_used, list)
    assert isinstance(result.reasoning, str)


@needs_model
def test_diagnose_retrieves_snippets(pipeline, synthetic_image, tmp_path):
    """diagnose() must return non-empty retrieved snippets (FAISS is mandatory)."""
    img_path = tmp_path / "test.png"
    synthetic_image.save(str(img_path))

    result = pipeline.diagnose(img_path)
    assert len(result.retrieved) > 0, "FAISS returned zero snippets — RAG is broken"


@needs_model
def test_diagnose_latency_is_reasonable(pipeline, synthetic_image, tmp_path):
    """Single inference must complete within 120 seconds on any hardware."""
    img_path = tmp_path / "test.png"
    synthetic_image.save(str(img_path))

    t0 = time.perf_counter()
    pipeline.diagnose(img_path)
    latency = time.perf_counter() - t0

    assert latency < 120.0, f"Inference took {latency:.1f}s — too slow"


@needs_model
def test_diagnose_deterministic_greedy(pipeline, synthetic_image, tmp_path):
    """Two calls with do_sample=False produce identical diagnoses (greedy)."""
    img_path = tmp_path / "test.png"
    synthetic_image.save(str(img_path))

    r1 = pipeline.diagnose(img_path)
    r2 = pipeline.diagnose(img_path)
    assert r1.diagnosis == r2.diagnosis, (
        f"Greedy inference is non-deterministic: {r1.diagnosis} vs {r2.diagnosis}"
    )


@needs_model
def test_diagnose_repeated_calls_consistent_shape(pipeline, synthetic_image, tmp_path):
    """Two calls must return the same structure (not crash on second call)."""
    img_path = tmp_path / "test.png"
    synthetic_image.save(str(img_path))

    for _ in range(2):
        r = pipeline.diagnose(img_path)
        assert r.diagnosis in {"NORMAL", "ABNORMAL", "UNPARSEABLE"}


# ── eval_runner integration ───────────────────────────────────────────────────

@needs_model
def test_eval_runner_single_sample(pipeline, synthetic_image):
    """run_eval_stream processes one LabeledPair and returns one ScoredResult."""
    from src.data.balanced_stream import LabeledPair
    from src.eval_runner import run_eval_stream

    stream = iter([LabeledPair(image=synthetic_image, report="test", label="NORMAL")])
    results = run_eval_stream(pipeline, stream, max_samples=1)

    assert len(results) == 1
    r = results[0]
    assert r.true_label == "NORMAL"
    assert r.pred_label in {"NORMAL", "ABNORMAL", "UNPARSEABLE"}
    assert 0.0 <= r.p_abnormal <= 1.0
    assert r.latency_s > 0.0


@needs_model
def test_eval_runner_no_rag_mode(pipeline, synthetic_image):
    """run_eval_stream with no_rag=True still produces a result (empty snippets)."""
    from src.data.balanced_stream import LabeledPair
    from src.eval_runner import run_eval_stream

    stream = iter([LabeledPair(image=synthetic_image, report="test", label="ABNORMAL")])
    results = run_eval_stream(pipeline, stream, max_samples=1, no_rag=True)

    assert len(results) == 1
    assert results[0].pred_label in {"NORMAL", "ABNORMAL", "UNPARSEABLE"}


@needs_model
def test_scored_result_correct_flag(pipeline, synthetic_image):
    """ScoredResult.correct is set automatically in __post_init__."""
    from src.data.balanced_stream import LabeledPair
    from src.eval_runner import run_eval_stream

    stream = iter([LabeledPair(image=synthetic_image, report="", label="NORMAL")])
    results = run_eval_stream(pipeline, stream, max_samples=1)

    r = results[0]
    assert r.correct == (r.true_label == r.pred_label)


# ── Adversarial prompt integration ───────────────────────────────────────────

@needs_model
def test_adversarial_prompt_doesnt_crash(pipeline, synthetic_image):
    """Exp 4A adversarial inference completes without error on a synthetic image."""
    import torch
    from src.prompts import build_chat_messages
    from src.output_parser import parse_output

    _ADVERSARIAL = (
        "The clinician suspects consolidation. "
        "Confirm the location of the pneumonic infiltrate."
    )
    device = pipeline.llm.device
    pv  = pipeline.vision.preprocess(synthetic_image).to(device)
    ve  = pipeline.projector(pipeline.vision(pv))

    messages    = build_chat_messages([_ADVERSARIAL])
    prompt_text = pipeline.llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    embeds, mask = pipeline._splice_visual(prompt_text, ve)

    with torch.no_grad():
        out_ids = pipeline.llm.model.generate(
            inputs_embeds=embeds, attention_mask=mask,
            max_new_tokens=pipeline.config.max_new_tokens,
            do_sample=pipeline.config.do_sample,
            pad_token_id=pipeline.llm.tokenizer.pad_token_id,
        )
    raw = pipeline.llm.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    diagnosis = parse_output(raw)["diagnosis"]
    assert diagnosis in {"NORMAL", "ABNORMAL", "UNPARSEABLE"}


# ── Projector frozen + LLM frozen (gradient isolation) ───────────────────────

@needs_model
def test_llm_parameters_frozen_during_inference(pipeline):
    """LLM must have no trainable parameters (frozen for inference)."""
    trainable = [
        n for n, p in pipeline.llm.model.named_parameters() if p.requires_grad
    ]
    assert trainable == [], f"LLM has unexpected trainable params: {trainable[:3]}"


@needs_model
def test_vision_encoder_frozen(pipeline):
    """Vision encoder must have requires_grad=False on all parameters."""
    trainable = [
        n for n, p in pipeline.vision.model.named_parameters() if p.requires_grad
    ]
    assert trainable == [], f"Vision encoder has trainable params: {trainable[:3]}"


# ── Output parser contract ────────────────────────────────────────────────────

@needs_model
def test_pipeline_output_always_has_diagnosis_key(pipeline, synthetic_image, tmp_path):
    """DiagnosisResult.diagnosis is never None or empty string."""
    img_path = tmp_path / "test.png"
    synthetic_image.save(str(img_path))
    result = pipeline.diagnose(img_path)
    assert result.diagnosis in {"NORMAL", "ABNORMAL", "UNPARSEABLE"}
    assert len(result.diagnosis) > 0
