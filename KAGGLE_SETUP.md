# Kaggle Setup — Resume Stage 2 Training from Step 2000

Kaggle gives **30 GPU hours/week for free** (P100 16 GB or T4 16 GB).
Sessions last up to **12 hours** before auto-disconnect.
At estimated ~6–10 hours per run on a P100.

---

## Kaggle vs RunPod — Key Differences

| | RunPod | Kaggle |
|---|---|---|
| Cost | ~$0.13/hr | Free (30 hr/wk) |
| GPU (typical) | RTX 3090 (24 GB) | P100 or T4 (16 GB) |
| Session limit | Until you stop | 12 hrs/session |
| File system | Persistent `/workspace/` | Ephemeral `/kaggle/working/` — auto-saved on exit |
| Secrets | `.env` file | Kaggle Secrets (Add-ons menu) |
| Internet | Yes | Yes (must enable per notebook) |

---

## 1. One-Time Kaggle Setup

### 1a. Create a Kaggle account
Go to [kaggle.com](https://www.kaggle.com) and sign up.

### 1b. Add your secrets
1. Open any notebook → **Add-ons** (top menu) → **Secrets**
2. Add two secrets:

| Label | Value |
|---|---|
| `HF_TOKEN` | Your HuggingFace access token (needs LLaMA-3.2 access) |
| `SEMANTIC_SCHOLAR_API_KEY` | Your Semantic Scholar key |

These persist across all your notebooks — you only do this once.

### 1c. Upload model weights as a Kaggle Dataset

The checkpoint files are too large for GitHub. Upload them once as a private Kaggle Dataset so every session can load them instantly.

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Name it `meddiag-weights` (private)
3. Upload a zip containing:
   ```
   projector_stage1.pt
   cls_head.pt
   lora_step2000/
       adapter_config.json
       adapter_model.safetensors
       train_state.pt
   faiss_index/
       ...all files...
   ```
4. Click **Create** — Kaggle processes and stores it permanently.

> **Alternative:** Use `gdown` each session to re-download from Google Drive.
> This wastes ~10 minutes per session. The Kaggle Dataset approach is much faster.

---

## 2. Create the Training Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Click **File** → **Import Notebook** → upload `MEDDIAG_Kaggle_Training.ipynb` from this repo
3. In the right panel:
   - **Accelerator** → `GPU P100` (or `T4 x2` if available — uses only cuda:0)
   - **Internet** → `On`
4. Click **Add Data** → search for `meddiag-weights` (your dataset from step 1c)
   - This mounts it at `/kaggle/input/meddiag-weights/`
5. In **Add-ons** → **Secrets** → toggle on `HF_TOKEN` and `SEMANTIC_SCHOLAR_API_KEY`

---

## 3. Run Training

Open `MEDDIAG_Kaggle_Training.ipynb` and run cells top to bottom.

The training cell will stream output directly. It blocks until completion or session timeout.
Kaggle auto-saves everything in `/kaggle/working/` when the session ends.

---

## 4. Resume After Session Timeout (Multi-Session Workflow)

After a session ends (auto or manual):

1. Go to your notebook → **Output** tab → find the saved files
2. Click **Create Dataset from Output** → name it `meddiag-ckpt-stepXXXX`
3. Start a **new session** of the same notebook
4. **Add Data** → add `meddiag-ckpt-stepXXXX` as input
5. The notebook will auto-detect the latest checkpoint from that input dataset and resume

The notebook's resume cell handles this automatically — just run all cells as normal.

---

## 5. Training Time Estimate

| Stage | Steps | Estimated Time (P100) |
|---|---|---|
| Remaining from step 2000 | ~5 500 | 6–10 hours |
| Full run if starting fresh | ~7 500 | 9–14 hours |

With 30 hrs/week you should finish in **1 week of normal usage** even with re-downloads.

---

## Tips

- **Don't close the browser tab while training** — Kaggle may suspend idle sessions.
  Open the notebook in a tab and leave it visible.
- **Early checkpoint saving** — the script saves every 250 steps.
  Even a 4-hour session completes ~1 000 steps before disconnect.
- **P100 vs T4** — P100 is generally faster for this model. Pick it when available.
- **bitsandbytes on Kaggle** — uses CUDA 12.x. The pinned version (0.49.2) is verified
  compatible. Do not upgrade without testing.
- **After all training** — download the `reports/` and `logs/` folders from the
  Output tab, or save them to your Kaggle Dataset for permanent storage.
