# RunPod Setup — Resume Stage 2 Training from Step 2000

This guide lets you resume LoRA fine-tuning from checkpoint **step 2000** on a cloud GPU
(RTX 3090 / A5000 / A6000 recommended — needs at least 16 GB VRAM).

---

## 1. Rent a Pod on RunPod

1. Go to [runpod.io](https://runpod.io) → **Pods** → **Deploy**
2. Choose **GPU Pod** (not Serverless)
3. Recommended GPU: **RTX 3090** (~$0.13/hr), **RTX A5000**, or **RTX 4090**
4. Template: **RunPod PyTorch 2.1** (has CUDA + Python pre-installed)
5. Disk: set **Container Disk** to at least **60 GB** (for model weights + HuggingFace cache)
6. Click **Deploy** → wait for status to turn green → click **Connect → Start Web Terminal**

---

## 2. Clone the Repo

```bash
git clone https://github.com/Sreenjoyee/visual-language-model-research-qlora-cot-rag.git
cd visual-language-model-research-qlora-cot-rag
```

---

## 3. Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57.6 peft==0.19.1 bitsandbytes==0.49.2 \
    datasets accelerate sentence-transformers faiss-cpu \
    scikit-learn pillow requests python-dotenv bert-score
```

---

## 4. Set Environment Variables

Create a `.env` file with your API keys:

```bash
cat > .env << 'EOF'
HF_TOKEN=<ask_sujal_for_hf_token>
SEMANTIC_SCHOLAR_API_KEY=<ask_sujal_for_s2_key>
HF_HUB_DISABLE_SYMLINKS_WARNING=1
EOF
```

> **Note:** Ask Sujal for the actual `HF_TOKEN` and `SEMANTIC_SCHOLAR_API_KEY` values — they are
> secret and not stored in this repo. Sujal: the keys are in your local `.env` file.

---

## 5. Upload Model Weights from Google Drive

The checkpoint weights are too large for GitHub. Download them from Google Drive
(ask Sujal for the share link) and place them in the correct folders.

**Files needed** (Sujal will share a zip on Google Drive):

| File in Drive | Where to put it on RunPod |
|---|---|
| `projector_stage1.pt` | `models/projector_stage1.pt` |
| `cls_head.pt` | `models/cls_head.pt` |
| `lora_step2000/adapter_model.safetensors` | `models/lora_step2000/adapter_model.safetensors` |
| `lora_step2000/train_state.pt` | `models/lora_step2000/train_state.pt` |
| `faiss_index/` (whole folder) | `faiss_index/` |

### Quick download via `gdown` (if Sujal shares a Google Drive folder link):

```bash
pip install gdown
# Replace FOLDER_ID with the ID from the share link
gdown --folder "https://drive.google.com/drive/folders/FOLDER_ID" -O .
```

Or use the RunPod file manager (Files tab in the pod dashboard) to drag-and-drop files.

### Verify files are in place:

```bash
ls -lh models/lora_step2000/
# Should show: adapter_config.json, adapter_model.safetensors, README.md, train_state.pt
ls -lh models/projector_stage1.pt models/cls_head.pt
ls faiss_index/
```

---

## 6. Fix Pipeline State to Resume from Step 2

The pipeline tracks which stages are done. Tell it stages 0, 1, and 2 are already complete
(FAISS index built, Stage 1 projector trained, Stage 2 already at step 2000):

```bash
mkdir -p logs
printf "step0\nstep1\nstep2\n" > logs/.pipeline_state
```

> **Why step2?** The Stage 2 checkpoint at step 2000 is a mid-training checkpoint,
> NOT a completed stage. Writing "step2" tells the pipeline to skip the stage2 block
> entirely — instead you will run Stage 2 directly with `--resume-from` (see below).

---

## 7. Resume Stage 2 Training Directly

Instead of running the full pipeline, run Stage 2 directly so you can control it:

```bash
python -m experiments.stage2_classification \
    --projector-path    models/projector_stage1.pt \
    --max-pairs         500 \
    --epochs            3 \
    --lr                2e-4 \
    --warmup-steps      50 \
    --grad-accum-steps  8 \
    --save-every        250 \
    --log-every         25 \
    --lora-save-dir     models/lora_adapter \
    --resume-from       models/lora_step2000
```

Training will print:
```
[stage2] Resumed from models/lora_step2000  (step=2000, epoch=0)
```

It will save checkpoints every 250 steps: `models/lora_step2250/`, `lora_step2500/`, etc.
Full training is 7500 steps; at step 2000 you have ~5500 steps left (~3–5 hours on a 3090).

---

## 8. Run Full Pipeline After Training Completes

Once Stage 2 finishes (the script exits normally), run the evaluation and experiments:

```bash
# Update pipeline state: stages 0-3 done
printf "step0\nstep1\nstep2\nstep3\n" > logs/.pipeline_state

# Run everything from stage 4 onward (evaluation + experiments)
bash run_pipeline.sh --resume
```

---

## 9. Download Results

After experiments finish, download `reports/` and `logs/` back to your local machine:

```bash
# On your local machine (PowerShell):
scp -r root@<RUNPOD_IP>:<PORT>:/workspace/visual-language-model-research-qlora-cot-rag/reports ./
```

Or use the RunPod file manager to download the `reports/` folder.

---

## Tips

- **Screen session** — wrap the training command in `screen` so it survives disconnect:
  ```bash
  screen -S training
  # run the python command above
  # Detach: Ctrl+A then D
  # Reattach: screen -r training
  ```
- **Stop/resume** — if the pod crashes, just re-run the same `--resume-from models/lora_step2000`
  command. The pipeline will auto-detect the latest checkpoint (e.g. `lora_step2500/`) and
  resume from there automatically.
- **VRAM** — on a 3090 (24 GB) you won't hit OOM. If you use a 16 GB GPU, keep
  `--grad-accum-steps 8` and `max_target_tokens=500` unchanged.
- **Cost estimate** — ~5 hours on RTX 3090 at $0.13/hr ≈ **$0.65 total**.
