#!/usr/bin/env bash
# =============================================================================
# MEDDIAG — End-to-End Pipeline Runner
# Usage:  bash run_pipeline.sh [--smoke] [--stage1-only] [--skip-train] [--bg]
#
#   --smoke        Fast sanity run (2 pairs, 4 eval samples)
#   --stage1-only  Stop after Stage 1; skip LoRA fine-tuning
#   --skip-train   Skip both training stages (use existing checkpoints)
#   --bg           Run in background; output tailed to logs/run.log
# =============================================================================
set -euo pipefail

# ── Token setup ───────────────────────────────────────────────────────────────
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
BG=0

for arg in "$@"; do
    case $arg in
        --smoke)       SMOKE=1 ;;
        --stage1-only) STAGE1_ONLY=1 ;;
        --skip-train)  SKIP_TRAIN=1 ;;
        --bg)          BG=1 ;;
        *)
            echo "[ERROR] Unknown argument: $arg"
            echo "  Usage: bash run_pipeline.sh [--smoke] [--stage1-only] [--skip-train] [--bg]"
            exit 1 ;;
    esac
done

# ── Background mode ───────────────────────────────────────────────────────────
# Re-launch self without --bg, piping output to logs/run.log.
# The original shell returns immediately; training continues detached.
if [[ $BG -eq 1 ]]; then
    mkdir -p logs
    ARGS_WITHOUT_BG="${@/--bg/}"
    nohup bash "$0" $ARGS_WITHOUT_BG >> logs/run.log 2>&1 &
    BG_PID=$!
    disown $BG_PID  # detach from shell job table so SIGHUP is not forwarded on exit
    echo "[bg] PID $BG_PID — tailing logs/run.log  (Ctrl+C to stop tailing; process keeps running)"
    tail -f logs/run.log
    exit 0
fi

# ── Sleep prevention (Windows) ───────────────────────────────────────────────
# Prevent the PC from sleeping or hibernating while training runs.
# Restored automatically on exit, Ctrl+C, or crash via trap.
_restore_power() {
    powercfg /change standby-timeout-ac 30   2>/dev/null || true
    powercfg /change hibernate-timeout-ac 60 2>/dev/null || true
    echo "[power] Sleep settings restored."
}
trap _restore_power EXIT INT TERM
powercfg /change standby-timeout-ac 0   2>/dev/null && \
powercfg /change hibernate-timeout-ac 0 2>/dev/null && \
echo "[power] Sleep disabled for duration of run." || \
echo "[warn]  Could not disable sleep (run as Admin for reliable prevention)."

# ── Config ────────────────────────────────────────────────────────────────────
if [[ $SMOKE -eq 1 ]]; then
    # 2 pairs: enough to confirm forward+backward runs, not enough to hang on CPU.
    MAX_PAIRS_S1=2
    MAX_PAIRS_S2=2
    WARMUP_S1=1
    WARMUP_S2=1
    SAVE_EVERY=0
    EVAL_SAMPLES=4
    ROBUSTNESS_SAMPLES=2
    echo "[info] SMOKE mode — minimal 2-pair run for pipeline verification (CPU-safe)"
else
    MAX_PAIRS_S1=5000
    MAX_PAIRS_S2=2000
    WARMUP_S1=100
    WARMUP_S2=50
    SAVE_EVERY=500
    EVAL_SAMPLES=200
    ROBUSTNESS_SAMPLES=50
fi

# RTX 3050 4GB: grad_accum_steps=8 keeps per-step activation memory low
GRAD_ACCUM=8

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
FAISS_INDEX_PATH="${MEDDIAG_FAISS_INDEX_DIR:-faiss_index}/index.faiss"
FAISS_META_PATH="${MEDDIAG_FAISS_INDEX_DIR:-faiss_index}/metadata.jsonl"
if [[ $SMOKE -eq 1 ]]; then
    # Smoke: small index, skip if already built from a prior smoke run
    if [[ -f "$FAISS_INDEX_PATH" ]]; then
        echo "[skip] Smoke FAISS index already exists at $FAISS_INDEX_PATH"
    else
        python -m scripts.build_faiss_index --skip-mimic --max-pubmed 20 --max-iuxray 50
        echo "[ok] FAISS index built (smoke — guidelines + small PubMed/IU-Xray)"
    fi
else
    # Full run: always rebuild so a stale smoke index is never used for eval
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
    # Auto-resume from latest mid-training checkpoint if one exists
    RESUME_S1=""
    LATEST_S1_CKPT=$(ls -t models/projector_step*.pt 2>/dev/null | head -1 || true)
    if [[ -n "$LATEST_S1_CKPT" ]]; then
        echo "[resume] Stage 1 checkpoint found: $LATEST_S1_CKPT"
        RESUME_S1="--resume-from $LATEST_S1_CKPT"
    fi

    python -m experiments.stage1_projector \
        --max-pairs        "$MAX_PAIRS_S1" \
        --epochs           1 \
        --lr               1e-4 \
        --warmup-steps     "$WARMUP_S1" \
        --grad-accum-steps "$GRAD_ACCUM" \
        --save-every       "$SAVE_EVERY" \
        --log-every        25 \
        --save-path        "$PROJECTOR_PATH" \
        $RESUME_S1
    echo "[ok] Stage 1 done → $PROJECTOR_PATH"
fi


# ── 3. Stage 2 — LoRA classification ─────────────────────────────────────────
step "3 / 7  Stage 2 — LoRA binary classification"
if [[ $SKIP_TRAIN -eq 1 || $STAGE1_ONLY -eq 1 ]]; then
    echo "[skip] skipped by flag"
elif [[ -d "$LORA_DIR" ]]; then
    echo "[skip] LoRA adapter already exists at $LORA_DIR"
else
    # Auto-resume from latest mid-training LoRA checkpoint if one exists
    RESUME_S2=""
    LATEST_S2_CKPT=$(ls -dt models/lora_step* 2>/dev/null | head -1 || true)
    if [[ -n "$LATEST_S2_CKPT" && -f "$LATEST_S2_CKPT/train_state.pt" ]]; then
        echo "[resume] Stage 2 checkpoint found: $LATEST_S2_CKPT"
        RESUME_S2="--resume-from $LATEST_S2_CKPT"
    fi

    python -m experiments.stage2_classification \
        --projector-path   "$PROJECTOR_PATH" \
        --max-pairs        "$MAX_PAIRS_S2" \
        --epochs           2 \
        --lr               2e-4 \
        --warmup-steps     "$WARMUP_S2" \
        --grad-accum-steps "$GRAD_ACCUM" \
        --save-every       "$SAVE_EVERY" \
        --log-every        25 \
        --lora-save-dir    "$LORA_DIR" \
        $RESUME_S2
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
    MEDDIAG_PROJECTOR="$PROJECTOR_PATH" MEDDIAG_LORA="$LORA_DIR" python - <<'EOF'
import os
from src.pipeline import MeddiagPipeline
from src.config import CONFIG
from PIL import Image

projector_path = os.environ["MEDDIAG_PROJECTOR"]
lora_dir = os.environ["MEDDIAG_LORA"]
pipe = MeddiagPipeline(
    config=CONFIG,
    projector_path=projector_path,
    lora_adapter_dir=lora_dir if os.path.isdir(lora_dir) else None,
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
