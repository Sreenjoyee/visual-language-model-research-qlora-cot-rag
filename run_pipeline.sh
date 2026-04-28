#!/usr/bin/env bash
# =============================================================================
# MEDDIAG — End-to-End Pipeline Runner
# Usage:  bash run_pipeline.sh [--smoke] [--stage1-only] [--skip-train] [--bg] [--resume]
#
#   --smoke        Fast sanity run (2 pairs, 4 eval samples)
#   --stage1-only  Stop after Stage 1; skip LoRA fine-tuning
#   --skip-train   Skip both training stages (use existing checkpoints)
#   --bg           Run in background; output tailed to logs/run.log
#   --resume       Skip stages already completed in a prior run (reads logs/.pipeline_state)
#
# Crash/shutdown recovery:
#   Each stage writes a completion marker to logs/.pipeline_state when it finishes.
#   Re-run with --resume to skip those stages and pick up from the last failure point.
#   To start fully fresh, delete logs/.pipeline_state before running.
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
RESUME=0

for arg in "$@"; do
    case $arg in
        --smoke)       SMOKE=1 ;;
        --stage1-only) STAGE1_ONLY=1 ;;
        --skip-train)  SKIP_TRAIN=1 ;;
        --bg)          BG=1 ;;
        --resume)      RESUME=1 ;;
        *)
            echo "[ERROR] Unknown argument: $arg"
            echo "  Usage: bash run_pipeline.sh [--smoke] [--stage1-only] [--skip-train] [--bg] [--resume]"
            exit 1 ;;
    esac
done

# ── Background mode ───────────────────────────────────────────────────────────
# Re-launch self without --bg via nohup + disown so the process survives terminal close.
if [[ $BG -eq 1 ]]; then
    mkdir -p logs
    ARGS_WITHOUT_BG="${@/--bg/}"
    nohup bash "$0" $ARGS_WITHOUT_BG >> logs/run.log 2>&1 &
    BG_PID=$!
    disown $BG_PID
    echo "[bg] PID $BG_PID — tailing logs/run.log  (Ctrl+C to stop tailing; process keeps running)"
    echo "[bg] Heartbeat: logs/.pipeline_heartbeat updated every 30 s while running"
    tail -f logs/run.log
    exit 0
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECTOR_PATH="models/projector_stage1.pt"
LORA_DIR="models/lora_adapter"
LOG_DIR="logs"
DIAG_DIR="diagnostics"
REPORTS_DIR="reports"
STATE_FILE="$LOG_DIR/.pipeline_state"
LOCK_FILE="$LOG_DIR/.pipeline.lock"
HEARTBEAT_FILE="$LOG_DIR/.pipeline_heartbeat"

mkdir -p models "$LOG_DIR" "$DIAG_DIR" "$REPORTS_DIR"

# ── Lock — prevent concurrent duplicate runs ──────────────────────────────────
if [[ -f "$LOCK_FILE" ]]; then
    EXISTING_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "[error] Pipeline already running (PID $EXISTING_PID). Aborting."
        echo "        Monitor: tail -f $LOG_DIR/run.log"
        exit 1
    fi
    echo "[warn] Stale lock found (PID ${EXISTING_PID:-?} is dead). Removing."
    rm -f "$LOCK_FILE"
fi
echo $$ > "$LOCK_FILE"

# Tee all output to logs/run.log so `tail -f logs/run.log` always works.
# stdbuf -oL forces line-buffered tee; PYTHONUNBUFFERED=1 stops Python from buffering.
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
exec > >(stdbuf -oL tee -a "$LOG_DIR/run.log") 2>&1

# ── Heartbeat — lets you verify a bg run is still alive ──────────────────────
_heartbeat() {
    while true; do
        date -Iseconds > "$HEARTBEAT_FILE" 2>/dev/null || true
        sleep 30
    done
}
_heartbeat &
HEARTBEAT_PID=$!

# ── Sleep prevention + cleanup trap (Windows) ─────────────────────────────────
_cleanup() {
    powercfg /change standby-timeout-ac 30   2>/dev/null || true
    powercfg /change hibernate-timeout-ac 60 2>/dev/null || true
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    rm -f "$LOCK_FILE" "$HEARTBEAT_FILE"
    echo "[power] Sleep settings restored. Lock released."
}
trap _cleanup EXIT INT TERM

powercfg /change standby-timeout-ac 0   2>/dev/null && \
powercfg /change hibernate-timeout-ac 0 2>/dev/null && \
echo "[power] Sleep disabled for duration of run." || \
echo "[warn]  Could not disable sleep (run as Admin for reliable prevention)."

# ── Stage state helpers ───────────────────────────────────────────────────────
# Each stage writes its name to STATE_FILE on success.
# --resume consults this file to skip already-completed stages.
_stage_done() { [[ $RESUME -eq 1 ]] && grep -qxF "$1" "$STATE_FILE" 2>/dev/null; }
_mark_done()  {
    echo "$1" >> "$STATE_FILE"
    echo "[state] Stage '$1' complete → $STATE_FILE"
}

# ── Network helpers ───────────────────────────────────────────────────────────
# Uses Python (guaranteed present) so it works on both Git Bash and WSL.
_net_ok() {
    python -c "
import urllib.request, sys
try:
    urllib.request.urlopen('https://huggingface.co', timeout=8)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

# Blocks until HuggingFace is reachable or max_wait seconds elapse.
wait_for_network() {
    local max_wait=300 interval=30 waited=0
    _net_ok && return 0
    echo "[net] HuggingFace unreachable — waiting up to ${max_wait}s for connectivity..."
    while ! _net_ok; do
        waited=$((waited + interval))
        if [[ $waited -ge $max_wait ]]; then
            echo "[warn] Network still down after ${max_wait}s — proceeding (may use local HF cache)."
            return 0
        fi
        echo "[net] Still unreachable. Retrying in ${interval}s... (${waited}s / ${max_wait}s elapsed)"
        sleep "$interval"
    done
    echo "[net] Network restored."
}

# Retries a command up to 4 times with doubling backoff; waits for network between attempts.
retry_net() {
    local max=4 delay=60 attempt=1
    until "$@"; do
        [[ $attempt -ge $max ]] && { echo "[error] Command failed after $max attempts: $*"; return 1; }
        echo "[retry] Attempt $attempt/$max failed — waiting ${delay}s then checking network..."
        sleep "$delay"
        wait_for_network
        delay=$((delay * 2))
        attempt=$((attempt + 1))
    done
}

# ── Config ────────────────────────────────────────────────────────────────────
if [[ $SMOKE -eq 1 ]]; then
    MAX_PAIRS_S1=2
    MAX_PAIRS_S2=2
    WARMUP_S1=1
    WARMUP_S2=1
    SAVE_EVERY=0
    EVAL_SAMPLES=4
    ROBUSTNESS_SAMPLES=2
    GRAD_ACCUM=1  # smoke: 2 pairs × 3 epochs = 6 steps; accum=1 ensures updates fire
    echo "[info] SMOKE mode — minimal 2-pair run for pipeline verification (CPU-safe)"
else
    MAX_PAIRS_S1=10000
    MAX_PAIRS_S2=4000
    WARMUP_S1=200
    WARMUP_S2=150
    SAVE_EVERY=500
    EVAL_SAMPLES=200
    ROBUSTNESS_SAMPLES=50
    GRAD_ACCUM=8  # RTX 3050 4GB: keeps per-step activation memory low
fi

# ── Helper ────────────────────────────────────────────────────────────────────
step() { echo; echo "════════════════════════════════════════"; echo "  $1"; echo "════════════════════════════════════════"; }


# ── 0. Dependency check ───────────────────────────────────────────────────────
step "0 / 7  Checking dependencies"
if _stage_done "step0"; then
    echo "[skip] Step 0 already done (--resume)"
else
    python -m pytest tests/ --ignore=tests/test_integration.py -q --tb=short 2>&1 | tail -4
    echo "[ok] All fast tests pass"
    _mark_done "step0"
fi


# ── 1. FAISS index ────────────────────────────────────────────────────────────
step "1 / 7  Building FAISS retrieval index"
if _stage_done "step1"; then
    echo "[skip] Step 1 already done (--resume)"
else
    FAISS_INDEX_PATH="${MEDDIAG_FAISS_INDEX_DIR:-faiss_index}/index.faiss"
    if [[ $SMOKE -eq 1 ]]; then
        if [[ -f "$FAISS_INDEX_PATH" ]]; then
            echo "[skip] Smoke FAISS index already exists at $FAISS_INDEX_PATH"
        else
            wait_for_network
            retry_net python -m scripts.build_faiss_index --skip-mimic --max-pubmed 20 --max-iuxray 50
            echo "[ok] FAISS index built (smoke — guidelines + small PubMed/IU-Xray)"
        fi
    elif [[ $SKIP_TRAIN -eq 1 && -f "$FAISS_INDEX_PATH" ]]; then
        echo "[skip] --skip-train set and FAISS index exists at $FAISS_INDEX_PATH"
    else
        wait_for_network
        retry_net python -m scripts.build_faiss_index --max-mimic 20000 --max-pubmed 300 --max-iuxray 2000
        echo "[ok] FAISS index built"
    fi
    # Verify index files are non-empty — catches power-cut mid-write
    if [[ ! -s "${MEDDIAG_FAISS_INDEX_DIR:-faiss_index}/index.faiss" || \
          ! -s "${MEDDIAG_FAISS_INDEX_DIR:-faiss_index}/meta.jsonl" ]]; then
        echo "[error] FAISS files missing or empty after build — aborting"
        exit 1
    fi
    _mark_done "step1"
fi


# ── 2. Stage 1 — projector alignment ─────────────────────────────────────────
step "2 / 7  Stage 1 — Projector alignment training"
if _stage_done "step2"; then
    echo "[skip] Step 2 already done (--resume)"
elif [[ $SKIP_TRAIN -eq 1 ]]; then
    echo "[skip] --skip-train set"
elif [[ -f "$PROJECTOR_PATH" ]]; then
    echo "[skip] Projector already exists at $PROJECTOR_PATH"
    _mark_done "step2"
else
    RESUME_S1=""
    LATEST_S1_CKPT=$(ls -t models/projector_step*.pt 2>/dev/null | head -1 || true)
    if [[ -n "$LATEST_S1_CKPT" ]]; then
        if python -c "import torch; torch.load('$LATEST_S1_CKPT', map_location='cpu')" 2>/dev/null; then
            echo "[resume] Stage 1 checkpoint valid: $LATEST_S1_CKPT"
            RESUME_S1="--resume-from $LATEST_S1_CKPT"
        else
            echo "[warn] Stage 1 checkpoint corrupted (power loss?): $LATEST_S1_CKPT — starting fresh"
        fi
    fi

    wait_for_network
    python -m experiments.stage1_projector \
        --max-pairs        "$MAX_PAIRS_S1" \
        --epochs           2 \
        --lr               1e-4 \
        --warmup-steps     "$WARMUP_S1" \
        --grad-accum-steps "$GRAD_ACCUM" \
        --save-every       "$SAVE_EVERY" \
        --log-every        25 \
        --save-path        "$PROJECTOR_PATH" \
        $RESUME_S1
    echo "[ok] Stage 1 done → $PROJECTOR_PATH"
    _mark_done "step2"
fi


# ── 3. Stage 2 — LoRA classification ─────────────────────────────────────────
step "3 / 7  Stage 2 — LoRA binary classification"
if _stage_done "step3"; then
    echo "[skip] Step 3 already done (--resume)"
elif [[ $SKIP_TRAIN -eq 1 || $STAGE1_ONLY -eq 1 ]]; then
    echo "[skip] skipped by flag"
elif [[ -d "$LORA_DIR" && -f "models/cls_head.pt" ]]; then
    echo "[skip] LoRA adapter and cls_head.pt already exist — Stage 2 up to date"
    _mark_done "step3"
else
    RESUME_S2=""
    LATEST_S2_CKPT=$(ls -dt models/lora_step* 2>/dev/null | head -1 || true)
    if [[ -n "$LATEST_S2_CKPT" && -f "$LATEST_S2_CKPT/train_state.pt" ]]; then
        if python -c "import torch; torch.load('$LATEST_S2_CKPT/train_state.pt', map_location='cpu')" 2>/dev/null; then
            echo "[resume] Stage 2 checkpoint valid: $LATEST_S2_CKPT"
            RESUME_S2="--resume-from $LATEST_S2_CKPT"
        else
            echo "[warn] Stage 2 checkpoint corrupted (power loss?): $LATEST_S2_CKPT — starting fresh"
        fi
    fi

    wait_for_network
    python -m experiments.stage2_classification \
        --projector-path   "$PROJECTOR_PATH" \
        --max-pairs        "$MAX_PAIRS_S2" \
        --epochs           3 \
        --lr               2e-4 \
        --warmup-steps     "$WARMUP_S2" \
        --grad-accum-steps "$GRAD_ACCUM" \
        --save-every       "$SAVE_EVERY" \
        --log-every        25 \
        --lora-save-dir    "$LORA_DIR" \
        $RESUME_S2
    echo "[ok] Stage 2 done → $LORA_DIR"
    _mark_done "step3"
fi


# ── 4. Evaluation ─────────────────────────────────────────────────────────────
step "4 / 7  Evaluation suite"
if _stage_done "step4"; then
    echo "[skip] Step 4 already done (--resume)"
else
    LORA_ARG=""
    if [[ -d "$LORA_DIR" && $STAGE1_ONLY -eq 0 ]]; then
        LORA_ARG="--lora-adapter-dir $LORA_DIR"
    fi

    wait_for_network
    retry_net python -m experiments.evaluate \
        --projector-path "$PROJECTOR_PATH" \
        $LORA_ARG \
        --max-samples          "$EVAL_SAMPLES" \
        --rag-ablation \
        --robustness \
        --robustness-samples   "$ROBUSTNESS_SAMPLES"
    echo "[ok] Evaluation done → $LOG_DIR/eval_report_*.json"
    _mark_done "step4"
fi


# ── 5. Research experiments (Exp 1–8) ─────────────────────────────────────────
step "5 / 7  Research experiments (1–8)"
if _stage_done "step5"; then
    echo "[skip] Step 5 already done (--resume)"
else
    LORA_EXP_ARG=""
    if [[ -d "$LORA_DIR" && $STAGE1_ONLY -eq 0 ]]; then
        LORA_EXP_ARG="--lora-adapter $LORA_DIR"
    fi

    LATEST_EXP_DIR=$(ls -dt "$REPORTS_DIR"/[0-9]* 2>/dev/null | head -1 || true)
    EXP_RESUME_ARGS=""
    EXP_IDS="all"

    if [[ -n "$LATEST_EXP_DIR" ]]; then
        # Build list of experiments that still need to run by checking each result file.
        ALL_EXP_IDS=("1" "2" "3" "4a" "4b" "5" "6" "7" "8")
        PENDING=()
        DONE=()
        for eid in "${ALL_EXP_IDS[@]}"; do
            result_file="$LATEST_EXP_DIR/exp${eid}_results.json"
            if [[ -f "$result_file" ]]; then
                DONE+=("$eid")
            else
                PENDING+=("$eid")
            fi
        done

        if [[ ${#DONE[@]} -gt 0 ]]; then
            DONE_STR=$(IFS=,; echo "${DONE[*]}")
            echo "[resume] Experiments already done: $DONE_STR"
        fi

        if [[ ${#PENDING[@]} -eq 0 ]]; then
            echo "[skip] All experiments already done — marking step5 complete"
            _mark_done "step5"
            # fall through to the _stage_done check on next resume
        else
            PENDING_STR=$(IFS=,; echo "${PENDING[*]}")
            echo "[resume] Experiments pending: $PENDING_STR — resuming into $LATEST_EXP_DIR"
            EXP_RESUME_ARGS="--resume-dir $LATEST_EXP_DIR"
            EXP_IDS="$PENDING_STR"
        fi
    fi

    wait_for_network
    retry_net python -m experiments.run_experiments \
        --exp          "$EXP_IDS" \
        --projector    "$PROJECTOR_PATH" \
        $LORA_EXP_ARG \
        --max-samples  "$EVAL_SAMPLES" \
        --tgp-w        55 \
        --output-dir   "$REPORTS_DIR/" \
        $EXP_RESUME_ARGS
    echo "[ok] Experiments done → $REPORTS_DIR/"
    _mark_done "step5"
fi


# ── 6. Visualize ──────────────────────────────────────────────────────────────
step "6 / 7  Generating plots"
if _stage_done "step6"; then
    echo "[skip] Step 6 already done (--resume)"
else
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
    _mark_done "step6"
fi


# ── 7. Sanity inference ───────────────────────────────────────────────────────
step "7 / 7  Sanity inference on sample_xray.jpg"
if _stage_done "step7"; then
    echo "[skip] Step 7 already done (--resume)"
else
    if [[ -f "sample_xray.jpg" ]]; then
        MEDDIAG_PROJECTOR="$PROJECTOR_PATH" MEDDIAG_LORA="$LORA_DIR" python - <<'EOF'
import os
from pathlib import Path
from src.pipeline import MeddiagPipeline
from src.config import CONFIG

projector_path = os.environ["MEDDIAG_PROJECTOR"]
lora_dir = os.environ["MEDDIAG_LORA"]
pipe = MeddiagPipeline(
    config=CONFIG,
    projector_weights=Path(projector_path),
    lora_adapter_dir=Path(lora_dir) if os.path.isdir(lora_dir) else None,
)
result = pipe.diagnose("sample_xray.jpg")
print("  Prediction  :", result.diagnosis)
print("  Confidence  :", f"{result.cls_confidence:.3f} (cls_head)" if result.cls_confidence is not None else "N/A (text-parse fallback)")
print("  Reasoning   :", result.reasoning[:200], "...")
EOF
    else
        echo "[skip] sample_xray.jpg not found; skipping inference demo"
    fi
    _mark_done "step7"
fi


# ── Done ──────────────────────────────────────────────────────────────────────
echo
echo "╔══════════════════════════════════════════╗"
echo "║         MEDDIAG pipeline complete        ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Projector   : $PROJECTOR_PATH"
echo "║  LoRA        : ${LORA_DIR:-none}"
echo "║  Cls head    : models/cls_head.pt"
echo "║  Eval report : $LOG_DIR/eval_report_*.json"
echo "║  Plots       : $DIAG_DIR/*.png"
echo "║  Exp reports : $REPORTS_DIR/"
echo "╚══════════════════════════════════════════╝"
