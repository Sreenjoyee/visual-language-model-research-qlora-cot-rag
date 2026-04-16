"""FAISS retrieval — MiniLM embeddings, IndexFlatL2, CPU only.

SRS §6: "FAISS is NOT optional. It is a core clinical grounding system."

This pass supports MIMIC-CXR report snippets only. Loader hooks for Radiopaedia,
MedPix, and guideline sources are defined as abstract methods so they can be
implemented later without touching the index format.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Config


INDEX_FILENAME = "index.faiss"
META_FILENAME = "meta.jsonl"


@dataclass
class RetrievedSnippet:
    text: str
    source: str
    distance: float


class KnowledgeSource(ABC):
    """Abstract knowledge source. Implementations yield (text, source_tag) pairs."""

    name: str = "unknown"

    @abstractmethod
    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        """Yield (text, source_tag). Caller writes to the index; source never changes."""


class MimicReportsSource(KnowledgeSource):
    """Stream MIMIC-CXR report impressions via HuggingFace datasets.

    Notes on licensing: MIMIC-CXR requires credentialed access. We pass HF_TOKEN
    and stream — nothing is cached to disk. The SRS explicitly forbids local
    dataset storage.
    """

    name = "mimic-cxr"

    def __init__(self, config: Config, max_snippets: int | None = None):
        self.config = config
        self.max_snippets = max_snippets

    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        # Lazy import so tests don't need the datasets package just to import this file.
        from datasets import load_dataset

        repo = self.config.mimic_dataset_repo
        try:
            ds = load_dataset(
                repo,
                split=self.config.mimic_split,
                streaming=True,
                token=self.config.hf_token or None,
            )
        except Exception as e:
            # Wrap with an actionable message. The original traceback still chains
            # so debugging info isn't lost — we just add guidance above it.
            raise RuntimeError(
                f"Could not load MIMIC dataset '{repo}' "
                f"(split='{self.config.mimic_split}'). "
                f"Underlying error: {type(e).__name__}: {e}\n"
                f"Fixes:\n"
                f"  1. Verify the dataset id exists on HuggingFace.\n"
                f"  2. If it's gated, ensure HF_TOKEN is set and you have access.\n"
                f"  3. Override via env var, e.g.\n"
                f"     export MEDDIAG_MIMIC_REPO=itsanmolgupta/mimic-cxr-dataset\n"
                f"  4. See README 'MIMIC mirror' section for known-working options."
            ) from e

        text_cols = self.config.mimic_text_columns
        count = 0
        schema_checked = False
        for example in ds:
            # One-time: detect which text column this dataset actually uses, and
            # fail loudly if none match. Better than silently yielding nothing.
            if not schema_checked:
                available = list(example.keys())
                if not any(c in example for c in text_cols):
                    raise RuntimeError(
                        f"None of the expected text columns {list(text_cols)} are "
                        f"present in dataset '{repo}'. Available columns: {available}. "
                        f"Set MEDDIAG_MIMIC_TEXT_COLUMNS in config.py or override "
                        f"the column list."
                    )
                schema_checked = True

            text = ""
            for col in text_cols:
                val = example.get(col)
                if val:
                    text = str(val).strip()
                    if text:
                        break
            if not text:
                continue
            yield text, self.name
            count += 1
            if self.max_snippets is not None and count >= self.max_snippets:
                break


# Hooks for sources we are deferring this pass (SRS §6 multi-source requirement).
# Each is a stub that raises if used — so nothing silently returns empty text.
class RadiopaediaSource(KnowledgeSource):
    name = "radiopaedia"

    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        raise NotImplementedError(
            "Radiopaedia source not implemented in this build. "
            "Requires ToS-compliant data acquisition — see SRS §6."
        )


class MedPixSource(KnowledgeSource):
    name = "medpix"

    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        raise NotImplementedError("MedPix source not implemented in this build.")


class GuidelinesSource(KnowledgeSource):
    name = "guidelines"

    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        raise NotImplementedError(
            "Clinical guidelines (ACR/WHO) source not implemented in this build."
        )


class Retriever:
    """FAISS IndexFlatL2 over MiniLM-normalized embeddings."""

    def __init__(self, config: Config):
        self.config = config
        self.embedder = SentenceTransformer(config.embedder_model_id, device="cpu")
        # Confirm embedder dim matches config — fail loudly if model changes.
        actual_dim = self.embedder.get_sentence_embedding_dimension()
        if actual_dim != config.embedder_dim:
            raise RuntimeError(
                f"Embedder dim {actual_dim} != config.embedder_dim {config.embedder_dim}. "
                "Update config or embedder model."
            )
        self.index: faiss.Index | None = None
        self.meta: list[dict] = []

    # ---- Build ----

    def build(self, sources: Sequence[KnowledgeSource], batch_size: int = 64) -> None:
        """Build a fresh IndexFlatL2 from one or more knowledge sources."""
        self.index = faiss.IndexFlatL2(self.config.embedder_dim)
        self.meta = []
        buffer_texts: list[str] = []
        buffer_sources: list[str] = []

        def flush():
            if not buffer_texts:
                return
            emb = self.embedder.encode(
                buffer_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ).astype(np.float32)
            assert self.index is not None
            self.index.add(emb)
            for t, s in zip(buffer_texts, buffer_sources):
                self.meta.append({"text": t, "source": s})
            buffer_texts.clear()
            buffer_sources.clear()

        for src in sources:
            for text, source_tag in src.iter_snippets():
                buffer_texts.append(text)
                buffer_sources.append(source_tag)
                if len(buffer_texts) >= batch_size:
                    flush()
        flush()

        if self.index.ntotal == 0:
            raise RuntimeError("FAISS index is empty after build — no snippets ingested.")

    # ---- Persist ----

    def save(self, directory: Path | None = None) -> None:
        directory = directory or self.config.faiss_index_dir
        directory.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            raise RuntimeError("No index to save.")
        faiss.write_index(self.index, str(directory / INDEX_FILENAME))
        with open(directory / META_FILENAME, "w", encoding="utf-8") as f:
            for row in self.meta:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load(self, directory: Path | None = None) -> None:
        directory = directory or self.config.faiss_index_dir
        idx_path = directory / INDEX_FILENAME
        meta_path = directory / META_FILENAME
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {directory}. "
                f"Run `python -m scripts.build_faiss_index` first."
            )
        self.index = faiss.read_index(str(idx_path))
        self.meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))
        if self.index.ntotal != len(self.meta):
            raise RuntimeError(
                f"Index/meta mismatch: {self.index.ntotal} vs {len(self.meta)}."
            )

    # ---- Query ----

    def query(self, text: str, k: int | None = None) -> list[RetrievedSnippet]:
        if self.index is None:
            raise RuntimeError("Retriever has no loaded index. Call .load() or .build().")
        k = k or self.config.retrieval_top_k
        if not text.strip():
            # Empty query is a bug upstream — do not silently return neighbors of noise.
            raise ValueError("Empty query passed to Retriever.query.")
        vec = self.embedder.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        distances, indices = self.index.search(vec, k)
        out: list[RetrievedSnippet] = []
        for dist, idx in zip(distances[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self.meta):
                continue
            row = self.meta[idx]
            out.append(RetrievedSnippet(text=row["text"], source=row["source"], distance=float(dist)))
        return out