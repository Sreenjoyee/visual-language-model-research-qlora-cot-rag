"""Central configuration.

Loads environment via python-dotenv and exposes a single Config dataclass.
No hardcoded secrets. All paths resolved relative to the project root.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present. Absence is fine — env vars may be set by the shell.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- Secrets / auth ---
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))

    # --- Hardware ---
    device: str = field(default_factory=lambda: os.environ.get("MEDDIAG_DEVICE", "cuda:0"))
    # Note: "4GB inference" in the SRS is aspirational. We measure actual usage
    # rather than hard-crash below a threshold. This value is passed to
    # transformers `max_memory` to prevent silent CPU offload.
    max_vram_gb: float = field(
        default_factory=lambda: float(os.environ.get("MEDDIAG_MAX_VRAM_GB", "6.0"))
    )

    # --- Model IDs ---
    vision_model_id: str = "google/vit-base-patch16-224"
    llm_model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    embedder_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Image preprocessing ---
    image_size: int = 224

    # --- Projector (Perceiver Resampler) ---
    # ViT-B/16 last_hidden_state: (B, 197, 768) at 224x224 input (14x14 patches + 1 cls token, 768 channels)
    vision_hidden_dim: int = 768
    num_visual_tokens: int = 8           # compressed visual tokens passed to LLM
    projector_num_heads: int = 8
    projector_num_layers: int = 2
    # LLaMA-3.2-3B hidden size is 3072; kept as a config to avoid magic numbers but
    # pipeline.py verifies this matches the actual loaded model at runtime.
    llm_hidden_dim: int = 3072

    # --- Retrieval ---
    faiss_index_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("MEDDIAG_FAISS_INDEX_DIR", str(PROJECT_ROOT / "faiss_index"))
        )
    )
    embedder_dim: int = 384              # MiniLM-L6-v2
    retrieval_top_k: int = 3

    # --- MIMIC-CXR dataset ---
    # Default: itsanmolgupta/mimic-cxr-dataset — a public mirror of MIMIC-CXR
    # report text + 512x512 images, verified reachable without credentialed access.
    # If you have PhysioNet credentials, swap to your approved mirror via the env
    # variable. Column names are also configurable because different mirrors use
    # different layouts ('impression' vs 'report' vs 'findings_section').
    mimic_dataset_repo: str = field(
        default_factory=lambda: os.environ.get(
            "MEDDIAG_MIMIC_REPO", "itsanmolgupta/mimic-cxr-dataset"
        )
    )
    mimic_split: str = field(
        default_factory=lambda: os.environ.get("MEDDIAG_MIMIC_SPLIT", "train")
    )
    # Candidate column names, checked in order. First match wins. If none match,
    # loaders raise with the actual available keys — no silent skip.
    mimic_text_columns: tuple[str, ...] = ("impression", "findings", "report")
    mimic_image_columns: tuple[str, ...] = ("image", "jpg", "dicom")

    # --- Inference ---
    max_new_tokens: int = 384
    # Greedy by default to make diagnosis reproducible. SRS §17 flags greedy as an
    # open question; switch to sampling in evaluation configs, not here.
    do_sample: bool = False
    temperature: float = 0.0

    # --- Paths ---
    logs_dir: Path = PROJECT_ROOT / "logs"
    models_dir: Path = PROJECT_ROOT / "models"
    diagnostics_dir: Path = PROJECT_ROOT / "diagnostics"

    def validate(self, require_token: bool = True) -> None:
        """Fail fast on missing critical config.

        Args:
            require_token: If True (default), raise when HF_TOKEN is missing.
                LLaMA-3.2 is gated and needs it. Set False for operations that
                only touch public datasets / open models (e.g. building FAISS
                against the public MIMIC mirror).
        """
        if require_token and not self.hf_token:
            raise RuntimeError(
                "HF_TOKEN is not set. LLaMA-3.2-3B-Instruct is a gated model and "
                "requires a HuggingFace token with accepted access. Copy "
                ".env.example to .env and add your token, or export HF_TOKEN "
                "in your shell."
            )
        for d in (self.logs_dir, self.models_dir, self.diagnostics_dir, self.faiss_index_dir):
            d.mkdir(parents=True, exist_ok=True)


# Module-level singleton for convenience. Import and call .validate() at entry points.
CONFIG = Config()