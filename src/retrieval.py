"""FAISS retrieval — MiniLM embeddings, IndexFlatL2, CPU only.

SRS §6: "FAISS is NOT optional. It is a core clinical grounding system."

This pass supports MIMIC-CXR report snippets only. Loader hooks for Radiopaedia,
MedPix, and guideline sources are defined as abstract methods so they can be
implemented later without touching the index format.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import CONFIG, Config


INDEX_FILENAME = "index.faiss"
META_FILENAME = "meta.jsonl"


def _http_get(url: str, timeout: int = 10) -> bytes:
    """Minimal HTTP GET returning raw bytes. Hard timeout prevents DNS hangs."""
    import socket
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    finally:
        socket.setdefaulttimeout(old_timeout)


def _detect_text_col(example: dict, cols: Sequence[str]) -> str | None:
    """Return the first column from cols present in example, or None."""
    return next((c for c in cols if c in example), None)


@dataclass
class RetrievedSnippet:
    text: str
    source: str
    distance: float
    # Raw embedding reconstructed from the FAISS index — used by ClassificationHead.
    # None when the index doesn't support reconstruction (shouldn't happen with
    # IndexFlatL2, but guarded so old code paths don't break).
    embedding: np.ndarray | None = field(default=None, repr=False)


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
    """PubMed abstracts via Semantic Scholar API.

    Replaces NCBI E-utilities with Semantic Scholar — covers all PubMed papers,
    free, no authentication, different servers so unaffected by NCBI outages.
    Rate limit: ~1 req/s without API key.
    """

    name = "pubmed-radiology"

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
    _BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, max_snippets: int = 300, per_query: int = 100):
        self.max_snippets = max_snippets
        self.per_query = min(per_query, 100)  # Semantic Scholar max is 100

    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        yielded = 0
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        if api_key:
            print("[RadiopaediaSource] Semantic Scholar: API key found — using authenticated requests")

        for query in self._QUERIES:
            if yielded >= self.max_snippets:
                break

            url = (
                f"{self._BASE}"
                f"?query={urllib.parse.quote(query)}"
                f"&fields=abstract"
                f"&limit={self.per_query}"
            )
            try:
                req = urllib.request.Request(url, headers={"x-api-key": api_key} if api_key else {})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
            except Exception as e:
                if "429" in str(e):
                    print("[RadiopaediaSource] Semantic Scholar rate limited — skipping source")
                    return
                print(f"[RadiopaediaSource] Semantic Scholar failed '{query[:40]}': {e}")
                continue

            for paper in data.get("data", []):
                if yielded >= self.max_snippets:
                    break
                abstract = (paper.get("abstract") or "").strip()
                if len(abstract) < 80:
                    continue
                yield abstract, self.name
                yielded += 1

            time.sleep(1.0)  # 1 req/s with API key


class OpenAlexSource(KnowledgeSource):
    """PubMed abstracts via OpenAlex API.

    OpenAlex is a fully open, no-key-required index of 200M+ scholarly papers.
    Covers all PubMed content. Rate limit: 100K req/day — effectively unlimited.
    Abstracts stored as inverted index; reconstructed to plain text here.
    """

    name = "openalex"

    _QUERIES: list[str] = [
        "chest radiograph normal findings interpretation",
        "chest X-ray pneumonia consolidation diagnosis",
        "pleural effusion chest radiograph features",
        "pneumothorax chest X-ray diagnosis",
        "pulmonary edema chest radiograph cardiogenic",
        "atelectasis chest radiograph lobar collapse",
        "cardiomegaly cardiothoracic ratio chest X-ray",
        "interstitial lung disease chest radiograph",
        "pulmonary nodule chest radiograph malignancy",
        "chest X-ray abnormal findings classification",
    ]
    _BASE = "https://api.openalex.org/works"

    def __init__(self, max_snippets: int = 300, per_query: int = 50):
        self.max_snippets = max_snippets
        self.per_query = per_query

    @staticmethod
    def _reconstruct_abstract(inv_index: dict) -> str:
        if not inv_index:
            return ""
        positions: dict[int, str] = {}
        for word, pos_list in inv_index.items():
            for pos in pos_list:
                positions[pos] = word
        return " ".join(positions[i] for i in sorted(positions))

    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        yielded = 0
        for query in self._QUERIES:
            if yielded >= self.max_snippets:
                break
            url = (
                f"{self._BASE}"
                f"?search={urllib.parse.quote(query)}"
                f"&filter=open_access.is_oa:true"
                f"&per-page={self.per_query}"
                f"&select=abstract_inverted_index,title"
                f"&mailto={CONFIG.pubmed_email}"
            )
            try:
                data = json.loads(_http_get(url, timeout=15))
            except Exception as e:
                print(f"[OpenAlexSource] failed '{query[:40]}': {e}")
                continue

            for work in data.get("results", []):
                if yielded >= self.max_snippets:
                    break
                abstract = self._reconstruct_abstract(
                    work.get("abstract_inverted_index") or {}
                )
                if len(abstract) < 80:
                    continue
                yield abstract, self.name
                yielded += 1

            time.sleep(0.5)


class CrossRefSource(KnowledgeSource):
    """Radiology abstracts via CrossRef REST API.

    CrossRef indexes 140M+ scholarly works. Free, no API key required.
    Rate limit is generous — polite crawling (1 req/s) is well within limits.
    """

    name = "crossref"

    _QUERIES: list[str] = [
        "chest radiograph interpretation normal abnormal",
        "chest X-ray pneumonia diagnosis radiology",
        "pleural effusion radiograph clinical features",
        "pulmonary edema chest X-ray cardiogenic",
        "atelectasis radiograph diagnosis management",
    ]
    _BASE = "https://api.crossref.org/works"

    def __init__(self, max_snippets: int = 200, per_query: int = 50):
        self.max_snippets = max_snippets
        self.per_query = per_query

    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        yielded = 0
        for query in self._QUERIES:
            if yielded >= self.max_snippets:
                break
            url = (
                f"{self._BASE}"
                f"?query={urllib.parse.quote(query)}"
                f"&rows={self.per_query}"
                f"&select=abstract"
                f"&mailto={CONFIG.pubmed_email}"
            )
            try:
                data = json.loads(_http_get(url, timeout=15))
                items = data.get("message", {}).get("items", [])
            except Exception as e:
                print(f"[CrossRefSource] failed '{query[:40]}': {e}")
                continue

            for item in items:
                if yielded >= self.max_snippets:
                    break
                abstract = (item.get("abstract") or "").strip()
                # CrossRef wraps abstracts in JATS XML tags — strip them
                import re
                abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
                if len(abstract) < 80:
                    continue
                yield abstract, self.name
                yielded += 1

            time.sleep(1.0)


class EuropePMCSource(KnowledgeSource):
    """Radiology abstracts from Europe PMC REST API.

    Drop-in alternative to RadiopaediaSource (NCBI E-utilities). Same radiology
    queries, different endpoint — free, no authentication, and accessible from
    networks that block or throttle NCBI. Rate limit is generous (~10 req/s).
    """

    name = "europepmc-radiology"

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
    _BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, max_snippets: int = 300, per_query: int = 40):
        self.max_snippets = max_snippets
        self.per_query = per_query

    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        yielded = 0
        for query in self._QUERIES:
            if yielded >= self.max_snippets:
                break
            url = (
                f"{self._BASE}"
                f"?query={urllib.parse.quote(query)}"
                f"&resultType=core&format=json"
                f"&pageSize={self.per_query}"
            )
            try:
                data = json.loads(_http_get(url, timeout=15))
                results = data.get("resultList", {}).get("result", [])
            except Exception as e:
                print(f"[EuropePMCSource] search failed for '{query[:40]}': {e}")
                continue
            for article in results:
                if yielded >= self.max_snippets:
                    break
                abstract = article.get("abstractText", "").strip()
                if not abstract or len(abstract) < 80:
                    continue
                yield abstract, self.name
                yielded += 1
            time.sleep(0.15)


class HFPubMedQASource(KnowledgeSource):
    """Radiology-filtered QA contexts from the PubMedQA dataset (HuggingFace).

    qiaojin/PubMedQA is public and streams through HuggingFace — no external
    API call needed. Each example contains a research question and supporting
    abstract sentences; we concatenate the context sentences and keep only
    chest/radiology-relevant ones.
    """

    name = "pubmedqa-radiology"

    _KEYWORDS = frozenset({
        "chest", "radiograph", "x-ray", "pulmonary", "lung", "pleural",
        "pneumonia", "consolidation", "effusion", "opacity", "atelectasis",
        "pneumothorax", "cardiomegaly", "mediastinum", "thorax", "thoracic",
        "bronchial", "tracheal", "diaphragm", "costophrenic",
    })

    def __init__(self, max_snippets: int = 300):
        self.max_snippets = max_snippets

    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        from datasets import load_dataset
        try:
            ds = load_dataset(
                "qiaojin/PubMedQA", "pqa_labeled", split="train", streaming=True
            )
        except Exception as e:
            print(f"[HFPubMedQASource] Could not load PubMedQA: {e}")
            return

        yielded = 0
        for example in ds:
            if yielded >= self.max_snippets:
                break
            question = str(example.get("question", "")).lower()
            if not any(kw in question for kw in self._KEYWORDS):
                continue
            contexts = example.get("context", {}).get("contexts", [])
            if not contexts:
                continue
            text = " ".join(str(c) for c in contexts if c).strip()
            if len(text) < 80:
                continue
            yield text, self.name
            yielded += 1


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


class RadiopaediaArticleSource(KnowledgeSource):
    """Full-text radiology articles via Europe PMC REST API.

    Replaces NCBI PMC full-text fetcher with Europe PMC — identical content
    (Europe PMC mirrors PubMed Central), different servers, no NCBI dependency.
    Returns full article text for open-access papers. Free, no auth required.
    """

    name = "europepmc-fulltext"

    _QUERIES: list[str] = [
        "chest radiograph normal interpretation systematic review",
        "pneumonia chest X-ray radiographic diagnosis findings",
        "pleural effusion chest radiograph imaging diagnosis",
        "pneumothorax chest radiograph diagnosis management",
        "pulmonary edema chest radiograph cardiogenic diagnosis",
        "cardiomegaly chest radiograph cardiothoracic ratio",
        "atelectasis chest radiograph lobar collapse",
        "interstitial lung disease chest radiograph patterns",
        "pulmonary nodule chest radiograph malignancy evaluation",
        "chest X-ray interpretation education radiology",
    ]

    _SEARCH_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    _FULLTEXT_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    _MIN_LEN = 100
    _PER_QUERY = 3

    def _search_ids(self, query: str) -> list[str]:
        url = (
            f"{self._SEARCH_BASE}"
            f"?query={urllib.parse.quote(query)}+OPEN_ACCESS:Y"
            f"&resultType=lite&format=json&pageSize={self._PER_QUERY}"
        )
        try:
            data = json.loads(_http_get(url, timeout=15))
            return [
                r["pmcid"] for r in data.get("resultList", {}).get("result", [])
                if r.get("pmcid")
            ]
        except Exception as e:
            print(f"[EuropePMCFullText] search failed '{query[:40]}': {e}")
            return []

    def _fetch_paragraphs(self, pmcid: str) -> list[str]:
        url = f"{self._FULLTEXT_BASE}/{pmcid}/fullTextXML"
        try:
            raw = _http_get(url, timeout=20)
            root = ET.fromstring(raw)
        except Exception as e:
            print(f"[EuropePMCFullText] fetch failed {pmcid}: {e}")
            return []

        paragraphs = []
        for p in root.findall(".//p"):
            text = "".join(p.itertext()).strip()
            if len(text) >= self._MIN_LEN:
                paragraphs.append(text)
        return paragraphs

    def iter_snippets(self) -> Iterator[tuple[str, str]]:
        seen: set[str] = set()
        for query in self._QUERIES:
            ids = self._search_ids(query)
            time.sleep(1.0)
            for pmcid in ids:
                if pmcid in seen:
                    continue
                seen.add(pmcid)
                for para in self._fetch_paragraphs(pmcid):
                    yield para, self.name
                time.sleep(1.0)


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
            try:
                for text, source_tag in src.iter_snippets():
                    buffer_texts.append(text)
                    buffer_sources.append(source_tag)
                    if len(buffer_texts) >= batch_size:
                        flush()
            except Exception as e:
                print(f"[Retriever] source {src.name} failed, skipping: {type(e).__name__}: {e}")
            finally:
                flush()

        if self.index.ntotal == 0:
            raise RuntimeError("FAISS index is empty — all sources failed. Check network.")

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
            # Reconstruct the stored embedding so ClassificationHead can use it
            # directly without re-encoding. IndexFlatL2 supports reconstruct().
            try:
                embedding = self.index.reconstruct(int(idx))
            except AttributeError:
                embedding = None  # index type doesn't support reconstruct
            except Exception as e:
                print(f"[retriever] reconstruct({idx}) failed: {type(e).__name__}: {e}")
                embedding = None
            out.append(RetrievedSnippet(
                text=row["text"],
                source=row["source"],
                distance=float(dist),
                embedding=embedding,
            ))
        return out