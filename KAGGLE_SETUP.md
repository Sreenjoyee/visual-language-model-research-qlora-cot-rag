# Kaggle Setup — background training on a T4 (laptop can be off)

All training runs on Kaggle through **`kaggle_train_fresh.ipynb`**. It trains the full
pipeline from scratch (FAISS → Stage-1 projector → Stage-2 LoRA + cls_head → eval →
experiments → plots) under a **hard 6 GB VRAM cap**. It resumes automatically across runs.

## Why the laptop can be off

Kaggle has two ways to run a notebook:

| | Interactive session (running cells) | **Commit run (Save Version → Save & Run All)** |
|---|---|---|
| Needs the browser open | Yes; stops when the tab closes or sits idle | **No.** Runs on Kaggle's servers |
| Max length | 12 h | 12 h |
| Use for | quick debugging | **all training** |

The notebook stops itself at 11 h (`TIME_BUDGET_H`), uploads its checkpoints, and exits
before Kaggle's 12 h kill. To continue training, start another commit run.

## One-time setup

1. **Secrets** (Add-ons → Secrets, then tick each one for this notebook):
   - `HF_TOKEN`: HuggingFace token with LLaMA-3.2 access
   - `KAGGLE_KEY`: Kaggle API token (`KGAT_…`) for the account in `KAGGLE_USERNAME`
   - `SEMANTIC_SCHOLAR_API_KEY`: optional
2. **Import** `kaggle_train_fresh.ipynb` (File → Import Notebook).
3. **Settings:** Accelerator = **GPU T4**, Internet = **On**.
4. **Add Input** (avoids HuggingFace Xet 403s on Kaggle):
   - Model: Llama 3.2 → 3B-Instruct (transformers)
   - `raddar/chest-xrays-indiana-university`
   - `paultimothymooney/chest-xray-pneumonia`
   - optional: an old dataset containing `faiss_index/`. This is the text retrieval index,
     not trained weights. Reusing it skips about 1 h of index building.
5. Make sure the GitHub repo has the latest commit. The notebook clones it and refuses to
   train if `src/config.py` lacks the 6 GB cap (`MEDDIAG_VRAM_CAP_GB`).

## Each training run

**Save Version → Save & Run All (Commit) → Save.** Then you can close the tab and shut the
laptop down.

- **First run:** the state dataset (`meddiag-fresh-t4`) doesn't exist yet, so the run starts
  from scratch. The first upload creates the dataset.
- **Later runs:** the notebook downloads the **latest** dataset version through the API, so
  you don't need to attach or update it. Training resumes from the last checkpoint.
- Checkpoints upload every 20 min. A crash or kill loses at most one checkpoint interval
  (250 steps in Stage-2, 500 in Stage-1).
- **Done** when the log prints `ALL STAGES COMPLETE`.

Watch progress from any device: Your Work → the notebook → **Versions** → the running
version. The `vram_gb` field in the stage logs is the peak VRAM and must stay under 6.

## The 6 GB constraint on a 16 GB T4

The notebook sets `MEDDIAG_VRAM_CAP_GB=6`. When any process imports `src/config.py`, this
calls `torch.cuda.set_per_process_memory_fraction`, so a step that would need more than
6 GB fails with CUDA OOM instead of quietly using the T4's extra memory. Before training
starts, cell 6 confirms the cap by trying a 7 GB allocation in a child process; that
allocation must fail.

## Quota

Kaggle gives about 30 GPU-hours per week, which is 2–3 runs. Restarting completely: set
`FORCE_FRESH = True` for one run, then set it back to `False`.
