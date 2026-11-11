# Kaggle Setup — training runs on Kaggle, your laptop can be off

All training runs on Kaggle with **`kaggle_train_fresh.ipynb`**. It trains the MEDDIAG
pipeline from scratch, exactly as described in the paper:
1. FAISS knowledge base over MIMIC-CXR reports
2. Stage-1 projector
3. Stage-2 QLoRA + classification head
4. Evaluation, all 9 experiments, and plots

GPU memory is held to a **hard 6 GB VRAM budget**. Runs resume and chain automatically.

**It works from any laptop and any Kaggle account.** Nothing is tied to a person:
- the Kaggle username and the notebook's name are detected from your API key
- the model and datasets download from HuggingFace
- checkpoints go to a dataset in whichever account runs it

The laptop is only needed to click **Save & Run All** once.

The step-by-step instructions (plain language) are at the top of the notebook itself.
Short version:

1. **Import** the notebook on kaggle.com (any name). Settings: **GPU T4 x2**, **Internet On**.
2. **Secrets:** `HF_TOKEN` and `KAGGLE_KEY` (see `secrets.md`). Tick both for this notebook.
   Your HuggingFace account must have accepted the
   [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) licence.
3. **Attach nothing.** The model and all datasets come from HuggingFace.
4. Click **Save Version → Save & Run All (Commit) → Save**, then close the laptop.

## Why the laptop can be off

| | Interactive session (▶ running cells) | **Commit run (Save & Run All)** |
|---|---|---|
| Needs the browser open | yes; stops when the tab closes or sits idle | **no**, runs on Kaggle's servers |
| Max length | 12 h | 12 h |

Each run stops itself at 11 h (`TIME_BUDGET_H`) and uploads everything to
`<your-username>/meddiag-fresh-t4`. It then waits until that upload has finished processing
and **starts the next run by itself**. The next run downloads the latest save and continues
from the last checkpoint. Checkpoints also upload every 20 minutes, so a crash loses at most
~20 minutes of work.

The chain stops on its own in three cases:
- **Everything is done:** the log says `ALL STAGES COMPLETE`.
- **The weekly GPU quota ran out** (~30 h/week). Click Save & Run All once after the weekly reset.
- **A run made no progress**, i.e. something is broken. Read the error at the end of its log.

## Where the results are

In the `meddiag-fresh-t4` dataset:
- `results/experiments/<date_time>/`: Exp 1–9 results (`expN_results.json` + plots), kept in a separate folder
- `lora_adapter/`, `cls_head.pt`, `projector_stage1.pt`: the trained model
- `eval_report_*.json`, `diagnostics/`: the evaluation report and plots

## Held to the paper

| Paper | Where it is enforced |
|---|---|
| ≤ 6 GB VRAM (hard ceiling, even on a 16 GB T4) | `MEDDIAG_VRAM_CAP_GB` → `src/config.py`; cell 6 proves it before training |
| Frozen DenseNet-121, 8 visual tokens × 3072, NF4 LLaMA-3.2-3B + bf16 | `src/config.py`, `src/llm.py` |
| Knowledge base = MIMIC-CXR report text, ≤ 20k, MiniLM + IndexFlatL2, k = 5 | `scripts/build_faiss_index.py --kb paper` |
| Head: concat(3072 visual, 384 RAG) → 512 → GELU → 2, mean pooling | `src/classification_head.py` (`arch="paper"`) |
| Class-weighted CE NORMAL 2.0 / ABNORMAL 1.0, λ 0.65/0.35, LoRA 16/32/0.05 q,k,v,o, LR 2e-4 (input_norm 2e-5), AdamW 0.01, warm-up 100 + cosine, clip 1.0, adversarial P = 0.20 | `experiments/stage2_classification.py`, `run_pipeline.sh` |
| Stage-2 trained on MIMIC-CXR only (IU X-ray stays out-of-distribution, RQ4) | `MEDDIAG_S2_EXTRA_SOURCES` empty by default |
| Held-out in-distribution evaluation | MIMIC rows ≥ 27,000 reserved for evaluation (`src/data/mimic_split.py`) |
| Exp 6: raw ECE + temperature-scaled ECE on a held-out calibration split | `experiments/run_experiments.py` |

`tests/test_paper_compliance.py` pins all of these.
