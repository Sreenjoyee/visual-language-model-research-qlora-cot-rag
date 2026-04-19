#!/usr/bin/env bash
# =============================================================================
# MEDDIAG — End-to-End Pipeline Runner
# Usage:  bash run_pipeline.sh [--smoke] [--stage1-only] [--skip-train]
#
#   --smoke        Fast sanity run (100 pairs, 50 eval samples)
#   --stage1-only  Stop after Stage 1; skip LoRA fine-tuning
#   --skip-train   Skip both training stages (use existing checkpoints)
# =============================================================================
set -euo pipefail

# ── Token setup ───────────────────────────────────────────────────────────────
# Reads HF_TOKEN from .env if present, otherwise falls back to the env.
# Never commit your token — .env is in .gitignore.
if [[ -f ".env" ]]; then
    export $(grep -v '^#' .env | xargs)
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[ERROR] HF_TOKEN is not set."
    echo "  Either: export HF_TOKEN=hf_..."
    echo "  Or:     echo 'HF_TOKEN=hf_...' > .env"
    exit 1
fi

# ── Argument parsing ──────────────────────────────────────────────────────────
SMOKE=0
STAGE1_ONLY=0
SKIP_TRAIN=0

for arg in "$@"; do
    case $arg in
        --smoke)       SMOKE=1 ;;
        --stage1-only) STAGE1_ONLY=1 ;;
        --skip-train)  SKIP_TRAIN=1 ;;
        *)
            echo "[ERROR] Unknown argument: $arg"
            echo "  Usage: bash run_pipeline.sh [--smoke] [--stage1-only] [--skip-train]"
            exit 1 ;;
    esac
done

# ── Config ────────────────────────────────────────────────────────────────────
if [[ $SMOKE -eq 1 ]]; then
    MAX_PAIRS_S1=100
    MAX_PAIRS_S2=100
    WARMUP_S1=10
    WARMUP_S2=10
    SAVE_EVERY=50
    EVAL_SAMPLES=50
    ROBUSTNESS_SAMPLES=20
    echo "[info] SMOKE mode — tiny run for end-to-end verification"
else
    MAX_PAIRS_S1=5000
    MAX_PAIRS_S2=2000
    WARMUP_S1=100
    WARMUP_S2=50
    SAVE_EVERY=500
    EVAL_SAMPLES=200
    ROBUSTNESS_SAMPLES=50
fi

PROJECTOR_PATH="models/projector_stage1.pt"
LORA_DIR="models/lora_adapter"
LOG_DIR="logs"
DIAG_DIR="diagnostics"
REPORTS_DIR="reports"

mkdir -p models "$LOG_DIR" "$DIAG_DIR" "$REPORTS_DIR"

# ── Helper ────────────────────────────────────────────────────────────────────
step() { echo; echo "════════════════════════════════════════"; echo "  $1"; echo "════════════════════════════════════════"; }


# ── 0. Dependency check ───────────────────────────────────────────────────────
step "0 / 7  Checking dependencies"
python -m pytest tests/ --ignore=tests/test_integration.py -q --tb=short 2>&1 | tail -4
echo "[ok] All fast tests pass"


# ── 1. FAISS index ────────────────────────────────────────────────────────────
step "1 / 7  Building FAISS retrieval index"
if [[ -f "data/faiss_index/index.faiss" ]]; then
    echo "[skip] FAISS index already exists at data/faiss_index/index.faiss"
elif [[ $SMOKE -eq 1 ]]; then
    # Smoke: guidelines-only index (no MIMIC streaming, builds in ~5s)
    python -m scripts.build_faiss_index --skip-mimic --max-pubmed 20 --max-iuxray 50
    echo "[ok] FAISS index built (smoke — guidelines + small PubMed/IU-Xray)"
else
    python -m scripts.build_faiss_index --max-mimic 20000 --max-pubmed 300 --max-iuxray 2000
    echo "[ok] FAISS index built"
fi


# ── 2. Stage 1 — projector alignment ─────────────────────────────────────────
step "2 / 7  Stage 1 — Projector alignment training"
if [[ $SKIP_TRAIN -eq 1 ]]; then
    echo "[skip] --skip-train set"
elif [[ -f "$PROJECTOR_PATH" ]]; then
    echo "[skip] Projector already exists at $PROJECTOR_PATH"
else
    python -m experiments.stage1_projector \
        --max-pairs    "$MAX_PAIRS_S1" \
        --epochs       1 \
        --lr           1e-4 \
        --warmup-steps "$WARMUP_S1" \
        --grad-accum-steps 4 \
        --save-every   "$SAVE_EVERY" \
        --log-every    25 \
        --save-path    "$PROJECTOR_PATH"
    echo "[ok] Stage 1 done → $PROJECTOR_PATH"
fi


# ── 3. Stage 2 — LoRA classification ─────────────────────────────────────────
step "3 / 7  Stage 2 — LoRA binary classification"
if [[ $SKIP_TRAIN -eq 1 || $STAGE1_ONLY -eq 1 ]]; then
    echo "[skip] skipped by flag"
elif [[ -d "$LORA_DIR" ]]; then
    echo "[skip] LoRA adapter already exists at $LORA_DIR"
else
    python -m experiments.stage2_classification \
        --projector-path   "$PROJECTOR_PATH" \
        --max-pairs        "$MAX_PAIRS_S2" \
        --epochs           2 \
        --lr               2e-4 \
        --warmup-steps     "$WARMUP_S2" \
        --grad-accum-steps 4 \
        --save-every       "$SAVE_EVERY" \
        --log-every        25 \
        --lora-save-dir    "$LORA_DIR"
    echo "[ok] Stage 2 done → $LORA_DIR"
fi


# ── 4. Evaluation ─────────────────────────────────────────────────────────────
step "4 / 7  Evaluation suite"
LORA_ARG=""
if [[ -d "$LORA_DIR" && $STAGE1_ONLY -eq 0 ]]; then
    LORA_ARG="--lora-adapter-dir $LORA_DIR"
fi

python -m experiments.evaluate \
    --projector-path "$PROJECTOR_PATH" \
    $LORA_ARG \
    --max-samples          "$EVAL_SAMPLES" \
    --rag-ablation \
    --robustness \
    --robustness-samples   "$ROBUSTNESS_SAMPLES"
echo "[ok] Evaluation done → $LOG_DIR/eval_report_*.json"


# ── 5. Research experiments (Exp 1–8) ─────────────────────────────────────────
step "5 / 7  Research experiments (1–8)"
LORA_EXP_ARG=""
if [[ -d "$LORA_DIR" && $STAGE1_ONLY -eq 0 ]]; then
    LORA_EXP_ARG="--lora-adapter $LORA_DIR"
fi

python -m experiments.run_experiments \
    --exp all \
    --projector    "$PROJECTOR_PATH" \
    $LORA_EXP_ARG \
    --max-samples  "$EVAL_SAMPLES" \
    --tgp-w        55 \
    --output-dir   "$REPORTS_DIR/"
echo "[ok] Experiments done → $REPORTS_DIR/"


# ── 6. Visualize ──────────────────────────────────────────────────────────────
step "6 / 7  Generating plots"
EVAL_JSON=$(ls -t "$LOG_DIR"/eval_report_*.json 2>/dev/null | head -1 || true)
EVAL_ARG=""
if [[ -n "$EVAL_JSON" ]]; then
    EVAL_ARG="--eval-report $EVAL_JSON"
fi

python -m experiments.visualize \
    $EVAL_ARG \
    --stage1-log  "$LOG_DIR/stage1.jsonl" \
    --stage2-log  "$LOG_DIR/stage2.jsonl" \
    --out-dir     "$DIAG_DIR/" \
    --format      png
echo "[ok] Plots saved → $DIAG_DIR/"


# ── 7. Sanity inference ───────────────────────────────────────────────────────
step "7 / 7  Sanity inference on sample_xray.jpg"
if [[ -f "sample_xray.jpg" ]]; then
    python - <<EOF
from src.pipeline import MedDiagPipeline
from src.config import CONFIG
from PIL import Image

pipe = MedDiagPipeline(
    config=CONFIG,
    projector_path="$PROJECTOR_PATH",
    lora_adapter_dir="$LORA_DIR" if __import__('os').path.isdir("$LORA_DIR") else None,
)
img = Image.open("sample_xray.jpg").convert("RGB")
result = pipe(img)
print("  Prediction :", result["diagnosis"])
print("  Confidence :", result.get("p_abnormal", "n/a"))
print("  Report     :", result["response"][:200], "...")
EOF
else
    echo "[skip] sample_xray.jpg not found; skipping inference demo"
fi


# ── Done ──────────────────────────────────────────────────────────────────────
echo
echo "╔══════════════════════════════════════════╗"
echo "║         MEDDIAG pipeline complete        ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Projector   : $PROJECTOR_PATH"
echo "║  LoRA        : ${LORA_DIR:-none}"
echo "║  Eval report : $LOG_DIR/eval_report_*.json"
echo "║  Plots       : $DIAG_DIR/*.png"
echo "║  Exp reports : $REPORTS_DIR/"
echo "╚══════════════════════════════════════════╝"
