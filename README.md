# MEDDIAG

Clean-room rebuild of a chest X-ray multimodal radiology pipeline, per the SRS.

## What this is (today)

A runnable **inference** path:

Image → frozen ViT-B/16 → Perceiver Resampler → FAISS (MIMIC reports) → LLaMA-3.2-3B (4-bit NF4) → structured diagnosis + reasoning

## What this is not (yet)

- **Trained.** The projector is randomly initialized until you run Stage 1. Inferences will be nonsense medically; the pipeline is plumbing-correct but semantically untrained.
- **Clinically validated.** Do not use on real patient data outside research.
- **Feature-complete per the SRS.** Stages 2 & 3, the full evaluation suite, GREEN eval, Radiopaedia/MedPix/guidelines retrieval, and most of the §19 testing framework are deferred — see `STATUS.md`.

## Requirements

- Windows + Git Bash (or any POSIX shell)
- Python 3.12
- NVIDIA GPU with ≥ 6 GB VRAM recommended. The code now uses a ViT-B/16 vision encoder and the LLM loader will fail loudly if GPU memory is insufficient.
- Note: if your GPU compute capability is not supported by the installed PyTorch CUDA build, the code falls back to CPU loading. This is slower but avoids the unsupported-device crash seen on RTX 5060 with the current PyTorch wheel.
- HuggingFace token with access to:
  - `meta-llama/Llama-3.2-3B-Instruct`
  - A MIMIC-CXR mirror (see note below)

## Setup

```bash
# Clone, then:
cd meddiag
python -m venv .venv
source .venv/Scripts/activate          # Git Bash on Windows
# OR: .venv\Scripts\activate           # cmd/PowerShell

# Torch must be installed with the right CUDA wheel for YOUR system first:
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

cp .env.example .env
# edit .env and set HF_TOKEN
```

## Build the FAISS index (one-time)

FAISS is mandatory (SRS §6). Inference will not run without an index.

```bash
python -m scripts.build_faiss_index --max-snippets 20000
```

This streams MIMIC-CXR report text via HuggingFace (no images, no local dataset cache) and writes `faiss_index/index.faiss` + `faiss_index/meta.jsonl`.

**Default dataset: `itsanmolgupta/mimic-cxr-dataset`** — a public mirror, 30,633 rows, no credentials required. Columns: `image`, `findings`, `impression`.

**If you have PhysioNet / credentialed access** and want a different mirror, override via env var:

```bash
export MEDDIAG_MIMIC_REPO=your/preferred/mirror
export MEDDIAG_MIMIC_SPLIT=train
```

Column names are auto-detected from `impression` / `findings` / `report` (text) and `image` / `jpg` / `dicom` (image). If your mirror uses different column names, the loader will raise with the actual column list so you can update `mimic_text_columns` / `mimic_image_columns` in `src/config.py`.

## Run inference

```bash
# Use a real path — the one below is just an example. Forward slashes work in Git Bash on Windows.
python -m scripts.infer --image ./some_xray.jpg

# JSON output:
python -m scripts.infer --image ./some_xray.jpg --json
```

With an untrained projector (no `--projector-weights`), expect outputs that are structurally correct (the `DIAGNOSIS:` / `EVIDENCE_USED:` / `REASONING:` format) but medically meaningless. The code also now supports a CPU fallback if your GPU is unsupported by the installed PyTorch build; inference may still run on CPU but will be slow.

## Run tests

```bash
pip install pytest                     # if not already via requirements.txt
python -m pytest tests/ -v
```

The current tests cover the output parser and prompt templates — both run without GPU or network. More tests land as §19 is implemented.

## Project layout

```
meddiag/
├── src/
│   ├── config.py          # .env loading; all paths and model ids
│   ├── prompts.py         # SINGLE source of truth for prompt text
│   ├── output_parser.py   # structured-output parser (no torch dep)
│   ├── vision.py          # frozen ViT-B/16, spatial tokens
│   ├── projector.py       # Perceiver Resampler → LLaMA embedding space
│   ├── llm.py             # 4-bit NF4 loader; fails on silent CPU offload
│   ├── retrieval.py       # FAISS IndexFlatL2 + MiniLM; MIMIC source
│   ├── pipeline.py        # orchestrator; inputs_embeds-only forward
│   └── data/mimic_stream.py
├── scripts/
│   ├── build_faiss_index.py
│   └── infer.py
├── tests/
├── experiments/           # (empty) — training scripts go here
├── green_eval/            # (empty) — multi-judge eval
├── models/                # (empty) — projector / LoRA checkpoints
├── logs/                  # (empty) — training logs
├── diagnostics/           # (empty) — plots
├── requirements.txt
├── .env.example
├── .gitignore
└── STATUS.md              # what's done, what's next
```

## Design rules in force

From SRS §2 / §5 / §6 / §12:

- `inputs_embeds` + `attention_mask` only in the multimodal forward. Never `input_ids`.
- No image storage. No dataset caching. Streaming only.
- FAISS always on in the inference path.
- Prompts live in `src/prompts.py` only. Do not inline prompt strings anywhere else.
- `HF_TOKEN` via env vars only.
- No silent CPU offload — the LLM loader asserts every parameter is on GPU.

## Troubleshooting

**`HF_TOKEN is not set`** — Copy `.env.example` to `.env` and add your token.

**`Parameter '...' is on CPU — silent offload detected`** — Your GPU budget is too tight for the model. Raise `MEDDIAG_MAX_VRAM_GB` in `.env` or use a bigger GPU.

**`FAISS index not found`** — Run `python -m scripts.build_faiss_index` first.

**Streaming error on index build** — `src/data/mimic_stream.py` provides a retry wrapper. If errors persist, it's usually an HF auth issue (MIMIC requires credentialed access) or a wrong dataset repo id — see the MIMIC mirror note above.

**Unsupported GPU compute capability / CUDA device error** — If PyTorch does not support your GPU, the loader will fall back to CPU and continue. The code now detects this and avoids the unsupported-device crash seen on RTX 5060 with the current wheel.

## License

Research use only. Do not deploy on patient data without appropriate review and regulatory clearance.