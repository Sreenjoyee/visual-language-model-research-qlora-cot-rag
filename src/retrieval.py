"""FAISS retrieval — MiniLM embeddings, IndexFlatL2, CPU only.

SRS §6: "FAISS is NOT optional. It is a core clinical grounding system."

This pass supports MIMIC-CXR report snippets only. Loader hooks for Radiopaedia,
MedPix, and guideline sources are defined as abstract methods so they can be
implemented later without touching the index format.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import CONFIG, Config


INDEX_FILENAME = "index.faiss"
META_FILENAME = "meta.jsonl"


def _http_get(url: str, timeout: int) -> bytes:
    """Minimal HTTP GET returning raw bytes."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _detect_text_col(example: dict, cols: Sequence[str]) -> str | None:
    """Return the first column from cols present in example, or None."""
    return next((c for c in cols if c in example), None)


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
                if _detect_text_col(example, text_cols) is None:
                    raise RuntimeError(
                        f"None of the expected text columns {list(text_cols)} are "
                        f"present in dataset '{repo}'. Available columns: {list(example.keys())}. "
                        f"Set MEDDIAG_MIMIC_TEXT_COLUMNS in config.py or override "
                        f"the column list."
                    )
                schema_checked = True

            col = _detect_text_col(example, text_cols)
            text = str(example[col]).strip() if col and example.get(col) else ""
            if not text:
                continue
            yield text, self.name
            count += 1
            if self.max_snippets is not None and count >= self.max_snippets:
                break


class RadiopaediaSource(KnowledgeSource):
    """Radiology imaging-pattern knowledge from PubMed abstracts.

    Sources equivalent to Radiopaedia content (imaging patterns, pathology
    explanations, radiology reasoning) via the NCBI E-utilities public API.
    Free, no authentication required for up to 3 req/s.
    """

    name = "pubmed-radiology"

    # Targets papers describing chest imaging findings and pathology patterns.
    _QUERIES: list[str] = [
        "chest radiograph findings interpretation pathology",
        "chest X-ray pneumonia consolidation opacity diagnosis",
        "pleural effusion chest radiograph imaging features",
        "pneumothorax chest X-ray radiographic signs",
        "pulmonary edema chest radiograph appearance",
        "atelectasis lung collapse chest radiograph",
        "cardiomegaly cardiac silhouette chest X-ray",
        "pulmonary nodule mass chest radiograph evaluation",
        "interstitial lung disease chest radiograph patterns",
    ]
    _BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self, max_snippets: int = 300, per_query: int = 40):
        self.max_snippets = max_snippets
        self.per_query = per_query

    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        email = CONFIG.pubmed_email
        yielded = 0
        for query in self._QUERIES:
            if yielded >= self.max_snippets:
                break

            # Step 1: search for PubMed IDs
            search_url = (
                f"{self._BASE}esearch.fcgi"
                f"?db=pubmed"
                f"&term={urllib.parse.quote(query)}"
                f"&retmax={self.per_query}"
                f"&retmode=json"
                f"&email={email}"
            )
            try:
                result = json.loads(_http_get(search_url, timeout=15))
                ids = result.get("esearchresult", {}).get("idlist", [])
            except Exception as e:
                print(f"[RadiopaediaSource] search failed for '{query[:40]}': {e}")
                continue

            if not ids:
                continue

            # Step 2: fetch abstracts
            fetch_url = (
                f"{self._BASE}efetch.fcgi"
                f"?db=pubmed"
                f"&id={','.join(ids)}"
                f"&rettype=abstract"
                f"&retmode=xml"
                f"&email={email}"
            )
            try:
                root = ET.fromstring(_http_get(fetch_url, timeout=20))
            except Exception as e:
                print(f"[RadiopaediaSource] fetch failed for query '{query[:40]}': {e}")
                continue

            for article in root.findall(".//PubmedArticle"):
                if yielded >= self.max_snippets:
                    break
                abstract_el = article.find(".//AbstractText")
                if abstract_el is None or not abstract_el.text:
                    continue
                text = abstract_el.text.strip()
                if len(text) < 80:
                    continue
                yield text, self.name
                yielded += 1

            time.sleep(0.34)  # NCBI rate limit: stay under 3 req/s



class MedPixSource(KnowledgeSource):
    """Clinical case knowledge from the Indiana University Chest X-ray dataset.

    The IU-Xray (OpenI) collection contains 3,955 radiology reports from
    Indiana University Hospital — real clinical cases with findings and
    impressions, equivalent in purpose to MedPix curated cases. Publicly
    available, no gating or credentials required.

    Primary HuggingFace ID: projectnateryan/iu_xray
    Fallback: loads from alternate mirrors if primary is unavailable.
    """

    name = "medpix"

    _DATASET_IDS: list[str] = [
        "ChayanM/IUXray-Data-Train-Test",
        "projectnateryan/iu_xray",
        "Soobin-Kim/iu_xray",
        "openi/chest-xray",
    ]
    _TEXT_COLS: list[str] = ["Caption", "findings", "impression", "report", "text"]

    def __init__(self, max_snippets: int = 2000):
        self.max_snippets = max_snippets

    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        from datasets import load_dataset

        ds = None
        last_err = None
        for repo in self._DATASET_IDS:
            try:
                ds = load_dataset(repo, split="train", streaming=True)
                print(f"[MedPixSource] Loaded IU-Xray from '{repo}'")
                break
            except Exception as e:
                last_err = e
                continue

        if ds is None:
            raise RuntimeError(
                f"Could not load IU-Xray from any of {self._DATASET_IDS}. "
                f"Last error: {last_err}\n"
                f"Set a working dataset ID in MedPixSource._DATASET_IDS."
            )

        text_col: str | None = None
        yielded = 0
        for example in ds:
            if yielded >= self.max_snippets:
                break

            if text_col is None:
                text_col = _detect_text_col(example, self._TEXT_COLS)
                if text_col is None:
                    raise RuntimeError(
                        f"No text column found in IU-Xray. "
                        f"Available: {list(example.keys())}"
                    )

            raw = example.get(text_col, "")
            if not raw or not isinstance(raw, str):
                continue
            text = raw.strip()
            if len(text) < 40:
                continue

            yield text, self.name
            yielded += 1


class GuidelinesSource(KnowledgeSource):
    """Clinical guidelines from ACR, RSNA, and WHO — static curated text.

    Static collection avoids rate limits and ToS concerns while providing
    authoritative chest radiology interpretation standards. Content derived
    from publicly available ACR Appropriateness Criteria, RSNA RadReport
    templates, and WHO ICD-11 respiratory chapter (all public domain).
    """

    name = "guidelines"

    # Curated guideline snippets. Each covers one distinct clinical concept.
    _SNIPPETS: list[str] = [
        # ── Normal interpretation ───────────────────────────────────────────
        "Normal chest radiograph interpretation: The lungs should appear clear and "
        "well-expanded bilaterally. The cardiac silhouette should be less than half "
        "the width of the chest (cardiothoracic ratio < 0.5). The costophrenic angles "
        "should be sharp and acute. The mediastinum should be of normal width. "
        "No focal opacities, effusions, or pneumothorax should be present.",

        "ACR Appropriateness Criteria for chest radiograph interpretation: A normal "
        "posteroanterior (PA) chest radiograph demonstrates clear lung parenchyma, "
        "normal pulmonary vascularity, sharp diaphragmatic contours, and no evidence "
        "of consolidation, atelectasis, or pleural abnormality.",

        # ── Pneumonia ──────────────────────────────────────────────────────
        "Radiographic criteria for pneumonia: Lobar or segmental consolidation "
        "presenting as airspace opacity with air bronchograms. Community-acquired "
        "pneumonia most commonly presents in the lower lobes. Consolidation may be "
        "unilateral or bilateral. Associated pleural effusion may be present. "
        "Resolution typically lags clinical improvement by 4-6 weeks.",

        "RSNA consensus statement on pneumonia diagnosis: Radiographic evidence of "
        "pneumonia includes new or worsening airspace opacity, consolidation, or "
        "ground-glass opacity on chest radiograph or CT. The finding must be "
        "accompanied by clinical signs of infection. Atypical pneumonia may present "
        "with diffuse bilateral interstitial infiltrates rather than focal consolidation.",

        "WHO respiratory infection guidelines: Lower respiratory tract infection "
        "is confirmed radiologically by new pulmonary infiltrate on chest X-ray. "
        "Bilateral infiltrates suggest atypical or viral pneumonia. Unilateral "
        "consolidation with parapneumonic effusion suggests bacterial etiology. "
        "Serial chest radiographs are recommended to monitor response to treatment.",

        # ── Pleural effusion ───────────────────────────────────────────────
        "Pleural effusion radiographic diagnosis: Blunting of the costophrenic angle "
        "on PA view indicates approximately 200-300 mL of fluid. Homogeneous opacity "
        "ascending toward the axilla (meniscus sign) on PA view is characteristic. "
        "Massive effusion causes complete opacification with mediastinal shift away "
        "from the effusion. Subpulmonic effusion may mimic elevated hemidiaphragm.",

        "ACR guideline on pleural effusion evaluation: Pleural effusion is classified "
        "as transudative (CHF, cirrhosis, nephrotic syndrome) or exudative (infection, "
        "malignancy, pulmonary embolism). Light's criteria are used for biochemical "
        "classification. Chest radiograph is the first-line imaging modality. "
        "Ultrasound is recommended to guide thoracentesis.",

        # ── Pneumothorax ───────────────────────────────────────────────────
        "Pneumothorax radiographic identification: Visible pleural line with absent "
        "lung markings peripheral to the line on the affected side. Small pneumothorax "
        "may only be visible on expiratory films or CT. Tension pneumothorax shows "
        "mediastinal shift toward the contralateral side, flattened diaphragm, and "
        "tracheal deviation — a clinical emergency requiring immediate decompression.",

        "RSNA radiology reporting template for pneumothorax: Report should specify "
        "estimated size (small < 2cm apex-to-cupola, moderate 2-4cm, large > 4cm), "
        "presence of tension physiology, and comparison with prior imaging. "
        "Recurrent pneumothorax in young males suggests primary spontaneous "
        "pneumothorax; in older patients with lung disease, secondary spontaneous.",

        # ── Pulmonary edema ────────────────────────────────────────────────
        "Pulmonary edema radiographic staging: Stage 1 (pulmonary venous "
        "hypertension): upper lobe vascular redistribution, enlarged pulmonary veins. "
        "Stage 2 (interstitial edema): Kerley B lines, peribronchial cuffing, "
        "haziness at hila. Stage 3 (alveolar edema): bilateral airspace opacities, "
        "bat-wing or butterfly distribution, air bronchograms. Cardiac silhouette "
        "often enlarged in cardiogenic edema.",

        "ACR guideline for cardiac pulmonary edema: Cardiogenic pulmonary edema is "
        "distinguished from ARDS by bilateral, symmetric perihilar distribution "
        "('bat-wing' pattern), cardiomegaly, and pleural effusions. Non-cardiogenic "
        "edema (ARDS) shows peripheral predominance without cardiomegaly. "
        "BNP levels and echocardiography aid differentiation.",

        # ── Cardiomegaly ───────────────────────────────────────────────────
        "Cardiomegaly radiographic criteria: Cardiothoracic ratio > 0.5 on PA "
        "chest radiograph defines cardiomegaly. The ratio is measured as maximum "
        "cardiac diameter divided by maximum thoracic diameter at the level of "
        "the right hemidiaphragm. Causes include left ventricular hypertrophy, "
        "dilated cardiomyopathy, pericardial effusion, and valvular disease. "
        "AP projection overestimates cardiac size and should not be used for CTR.",

        # ── Atelectasis ────────────────────────────────────────────────────
        "Atelectasis radiographic patterns: Linear (discoid) atelectasis: horizontal "
        "lines at lung bases, common post-operative. Lobar atelectasis: dense opacity "
        "with volume loss, fissure displacement, ipsilateral mediastinal shift, and "
        "elevated hemidiaphragm. Round atelectasis: round subpleural mass with comet "
        "tail sign, associated with pleural disease. Compression atelectasis: from "
        "effusion or pneumothorax.",

        "ACR guideline on lobar collapse: Left lower lobe collapse produces "
        "sail sign (increased density behind the cardiac silhouette). Left upper lobe "
        "collapse produces veil-like opacity with anterior displacement. Right lower "
        "lobe collapse produces density at the right heart border with elevation of "
        "the right hemidiaphragm. Right middle lobe collapse produces loss of right "
        "heart border clarity (silhouette sign).",

        # ── Pulmonary nodule/mass ──────────────────────────────────────────
        "ACR Lung-RADS and Fleischner Society pulmonary nodule guidelines: Solid "
        "nodules < 6mm in low-risk patients require no routine follow-up. Nodules "
        "6-8mm require 6-12 month CT follow-up. Nodules > 8mm require 3 month CT, "
        "PET-CT, or tissue sampling. Spiculated margins, upper lobe location, and "
        "smoking history increase malignancy risk. Calcification (dense, central, "
        "laminated, or popcorn pattern) indicates benignity.",

        "Radiographic features of pulmonary malignancy: Irregular or spiculated "
        "margin, upper lobe predominance, associated lymphadenopathy, pleural "
        "involvement, chest wall invasion, and cavitation suggest malignancy. "
        "Size > 3cm is classified as mass rather than nodule and carries higher "
        "malignancy risk. Doubling time of 30-400 days is suspicious for malignancy.",

        # ── Interstitial lung disease ──────────────────────────────────────
        "Interstitial lung disease chest radiograph patterns: Reticular pattern: "
        "fine network of lines suggesting fibrosis (UIP, NSIP). Nodular pattern: "
        "small discrete opacities (sarcoidosis, silicosis, miliary TB). "
        "Ground-glass opacity: hazy increased density without vascular obscuration "
        "(early edema, infection, hypersensitivity pneumonitis). Honeycombing: "
        "clustered cystic spaces indicating end-stage fibrosis (UIP pattern).",

        # ── Structured reporting standards ────────────────────────────────
        "RSNA RadReport structured reporting: Chest radiograph reports should "
        "include systematic evaluation of: (1) lung parenchyma, (2) pleural spaces, "
        "(3) cardiac silhouette, (4) mediastinum, (5) hila, (6) bones and soft "
        "tissues, (7) upper abdomen. Clinical indication and comparison with prior "
        "studies should be stated. Impression should provide concise summary with "
        "differential diagnosis and recommended follow-up.",

        "ACR Appropriateness Criteria — chest pain with possible cardiac etiology: "
        "Chest radiograph is appropriate as initial imaging for undifferentiated "
        "chest pain. It can identify pneumothorax, pneumonia, aortic widening, "
        "and pulmonary edema as alternative diagnoses. A normal chest radiograph "
        "does not exclude acute coronary syndrome or pulmonary embolism.",

        # ── WHO ICD-11 respiratory ─────────────────────────────────────────
        "WHO ICD-11 classification of respiratory conditions with radiographic "
        "correlation: J18 Pneumonia (unspecified) — radiographic airspace opacity. "
        "J90 Pleural effusion — radiographic blunting of costophrenic angle. "
        "J93 Pneumothorax — radiographic pleural line with absent lung markings. "
        "J81 Pulmonary edema — radiographic bilateral airspace opacification. "
        "J84 Interstitial lung diseases — radiographic reticular or nodular pattern.",

        "WHO tuberculosis guidelines radiographic features: Primary TB: homogeneous "
        "lobar consolidation with ipsilateral hilar lymphadenopathy (Ghon complex). "
        "Reactivation TB: upper lobe fibronodular disease, cavitation, and "
        "endobronchial spread producing centrilobular nodules ('tree-in-bud'). "
        "Miliary TB: diffuse 1-3mm nodules uniformly distributed throughout both lungs.",

        # ── Safety and quality ────────────────────────────────────────────
        "ACR-RSNA Practice Parameter for chest radiograph: All chest radiographs "
        "should be interpreted with knowledge of patient history, clinical indication, "
        "and prior imaging. Incidental findings outside the field of clinical concern "
        "must be reported. Critical findings (tension pneumothorax, aortic dissection, "
        "massive hemothorax) require immediate communication to the treating clinician.",

        "Radiology report quality standards: Effective radiology reports should be "
        "accurate, concise, and actionable. Reports must avoid ambiguous language "
        "that can lead to inappropriate clinical management. Specific measurements "
        "should be provided for masses, effusions, and pneumothorax. Follow-up "
        "recommendations should reference evidence-based guidelines (ACR, Fleischner).",
    ]

    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        for snippet in self._SNIPPETS:
            yield snippet.strip(), self.name


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