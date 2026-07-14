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


def _vision_hidden_dim_default() -> int:
    """Feature dim of the configured vision backbone (kept in sync with the model)."""
    override = os.environ.get("MEDDIAG_VISION_HIDDEN_DIM")
    if override:
        return int(override)
    model = os.environ.get("MEDDIAG_VISION_MODEL", "xrv:densenet121-res224-all")
    if "densenet121" in model:
        return 1024          # TorchXRayVision DenseNet-121 (default encoder)
    return 1024


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
    # SRS §5: frozen lightweight vision encoder, outputs spatial tokens.
    # Default: TorchXRayVision DenseNet-121 ("xrv:densenet121-res224-all", ~8M,
    # pretrained on the union of NIH/CheXpert/MIMIC/PadChest → (B,1024,7,7)),
    # chosen for cross-dataset (OOD) generalisation. An "xrv:<weights>" tag loads a
    # TorchXRayVision backbone; any other value loads a HuggingFace AutoModel.
    vision_model_id: str = field(
        default_factory=lambda: os.environ.get("MEDDIAG_VISION_MODEL", "xrv:densenet121-res224-all")
    )
    # Env-overridable so it can point at a locally-attached copy (e.g. a Kaggle Model
    # mounted under /kaggle/input) — needed when HF's Xet CDN is unreachable/403s.
    llm_model_id: str = field(
        default_factory=lambda: os.environ.get("MEDDIAG_LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    )
    embedder_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Image preprocessing ---
    image_size: int = 224

    # --- Projector (Perceiver Resampler) ---
    # DenseNet-121 final feature map: 1024 channels.
    # Spatial map 7×7=49 tokens after reshape from (B,1024,7,7) → (B,49,1024).
    # Token count N is read dynamically — projector cross-attention handles any N.
    # (num_heads applies to the llm_dim attention space: 3072/8=384, not vision_dim.)
    # SRS §19.2 projector output: (B, 8, 3072).
    vision_hidden_dim: int = field(default_factory=_vision_hidden_dim_default)
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
    retrieval_top_k: int = 5

    # --- MIMIC-CXR dataset ---
    # Default: itsanmolgupta/mimic-cxr-dataset — a public mirror of MIMIC-CXR
    # report text + 512x512 images, verified reachable without credentialed access.
    # If you have PhysioNet credentials, swap to your approved mirror via the env
    # variable. Column names are also configurable because different mirrors use
    # different layouts ('impression' vs 'report' vs 'findings_section').
    mimic_dataset_repo: str = field(
        default_factory=lambda: os.environ.get(
            "MEDDIAG_MIMIC_REPO", "power2004/mimic-cxr-dataset"
        )
    )
    mimic_split: str = field(
        default_factory=lambda: os.environ.get("MEDDIAG_MIMIC_SPLIT", "train")
    )
    # Candidate column names, checked in order. First match wins. If none match,
    # loaders raise with the actual available keys — no silent skip.
    mimic_text_columns: tuple[str, ...] = ("impression", "findings", "report")
    mimic_image_columns: tuple[str, ...] = ("image", "jpg", "dicom")

    # --- Classification threshold ---
    # Starting point for the ClassificationHead softmax output (proper binary
    # probability in [0,1], balanced training). 0.5 is the natural midpoint for
    # a balanced classifier — recalibrate via Youden J on the Exp 1 ROC curve
    # after training and update MEDDIAG_THRESHOLD accordingly.
    # Previous value (0.3486) was for the old LLaMA logit-heuristic and is invalid here.
    classification_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("MEDDIAG_THRESHOLD", "0.5")
        )
    )

    # --- Classification head ---
    # Weight balancing classification loss vs LM generation loss in Stage 2.
    # total_loss = cls_alpha * cls_loss + (1 - cls_alpha) * lm_loss
    cls_alpha: float = field(
        default_factory=lambda: float(os.environ.get("MEDDIAG_CLS_ALPHA", "0.65"))
    )
    # Recall-oriented classification loss (Run 4). Defaults reproduce the plain
    # label-smoothed CE used through Run 3.
    #   cls_pos_weight  > 1.0 up-weights the ABNORMAL class (raises recall/sensitivity
    #                   by making false negatives costlier). e.g. 2.0-3.0.
    #   cls_focal_gamma > 0.0 enables focal loss (down-weights easy examples so the
    #                   model focuses on the hard/confidently-missed abnormals). e.g. 2.0.
    cls_pos_weight: float = field(
        default_factory=lambda: float(os.environ.get("MEDDIAG_CLS_POS_WEIGHT", "1.0"))
    )
    cls_focal_gamma: float = field(
        default_factory=lambda: float(os.environ.get("MEDDIAG_CLS_FOCAL_GAMMA", "0.0"))
    )
    # Dimension of the hidden layer in the classification MLP.
    cls_hidden_dim: int = 512

    # --- Run 4: vision-encoder fine-tuning + train-time augmentation (AUROC levers) ---
    # Defaults keep the encoder frozen and augmentation off (Run-3 behavior).
    #   vision_finetune         unfreeze the DenseNet-121 encoder so features adapt
    #                           further to the training set. ~10x below LoRA LR.
    #   vision_lr               LR for the unfrozen encoder params.
    #   vision_finetune_blocks  how many trailing encoder blocks to unfreeze (<=0 = all).
    #   vision_augment          light train-time aug (small rotation + brightness/contrast).
    vision_finetune: bool = field(
        default_factory=lambda: os.environ.get("MEDDIAG_VISION_FINETUNE", "").lower() in ("1", "true", "yes")
    )
    vision_lr: float = field(
        default_factory=lambda: float(os.environ.get("MEDDIAG_VISION_LR", "1e-5"))
    )
    vision_finetune_blocks: int = field(
        default_factory=lambda: int(os.environ.get("MEDDIAG_VISION_FT_BLOCKS", "2"))
    )
    vision_augment: bool = field(
        default_factory=lambda: os.environ.get("MEDDIAG_VISION_AUGMENT", "").lower() in ("1", "true", "yes")
    )

    # --- Inference ---
    max_new_tokens: int = 512
    # Greedy by default to make diagnosis reproducible. SRS §17 flags greedy as an
    # open question; switch to sampling in evaluation configs, not here.
    do_sample: bool = False
    temperature: float = 0.0

    # --- External API ---
    pubmed_email: str = "meddiag-research@noreply.local"  # NCBI requests an email

    # --- Debug ---
    debug_vision: bool = field(
        default_factory=lambda: bool(os.environ.get("MEDDIAG_DEBUG_VISION"))
    )

    # --- Paths ---
    logs_dir: Path = PROJECT_ROOT / "logs"
    models_dir: Path = PROJECT_ROOT / "models"
    diagnostics_dir: Path = PROJECT_ROOT / "diagnostics"

    @property
    def cls_head_path(self) -> Path:
        return self.models_dir / "cls_head.pt"

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