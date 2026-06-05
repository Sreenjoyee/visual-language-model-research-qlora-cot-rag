# MEDDIAG: Multimodal Chest Radiograph Diagnosis via Perceiver-Resampled QLoRA with Retrieval-Augmented Chain-of-Thought Reasoning

**Authors:** [Author Names]  
**Affiliation:** [Institution]  
**Date:** June 5, 2026  
**Status:** Publication-Ready Draft (Sections 1–3, Skeleton for 4–8)

---

## PAPER STRUCTURAL OUTLINE

### Section 1: Abstract & Introduction
- **Abstract (~200 words):** Motivation, problem statement, approach, key results, impact
- **1.1 Background & Motivation:** Clinical radiology workload, need for interpretable automated systems
- **1.2 Problem Statement & Research Gap:** Current VLM limitations in medical domains
- **1.3 Proposed Approach (MEDDIAG):** System overview
- **1.4 Contributions:** Bulleted list (5–7 key contributions)
- **1.5 Paper Organization**

### Section 2: Related Work (Placeholders)
- **2.1 Vision-Language Models & Parameter-Efficient Fine-Tuning**
  - Foundation VLMs (CLIP, LLaVA, BLIP)
  - LoRA and QLoRA (Hu et al., Dettmers et al.)
  - Medical VLM fine-tuning (LLaVA-Med, CXR-VLM)
  
- **2.2 Retrieval-Augmented Generation (RAG)**
  - RAG framework (Lewis et al.)
  - Multimodal RAG extensions
  - Medical knowledge base integration
  
- **2.3 Chain-of-Thought (CoT) Prompting**
  - CoT reasoning in vision-language tasks
  - Structured output generation
  - Medical reasoning templates

- **2.4 Medical Image Analysis & Explainability**
  - Chest X-ray classification benchmarks
  - Explainability in diagnostic systems
  - Grounding language in visual features

### Section 3: Methodology & System Architecture
- **3.1 System Overview:** Dataflow diagram reference, high-level architecture
- **3.2 Vision Encoding Module:** EfficientNet-B0, output specification
- **3.3 Perceiver Resampler (Projector):** Cross-attention architecture, equations
- **3.4 Retrieval-Augmented Generation:** FAISS indexing, query mechanism, knowledge sources
- **3.5 Prompt Engineering & CoT Template:** Inference prompt structure, 4-step reasoning
- **3.6 Classification Head:** Binary classifier, parallel to LLM path
- **3.7 Training Procedure:** Stage 1 & Stage 2 objectives, loss functions, data pipeline
- **3.8 QLoRA Fine-Tuning Details:** NF4 quantization, LoRA adapter configuration, OOM recovery

### Section 4: Experimental Setup & Results
- **4.1 Datasets:** MIMIC-CXR, IU-Xray, Pneumonia-Xray, PadChest
- **4.2 Baseline Models:** Comparison with DenseNet-121, QLoRA-8bit, LLaVA-Med, GPT-4V
- **4.3 Evaluation Metrics:** AUROC, F1, BERTScore, CHAIR, GREEN, ECE, latency
- **4.4 Experimental Results (Exp 1–9):** Detailed tables, graphs, analysis
  - Exp 1: In-distribution classification (NIH ChestX-ray14)
  - Exp 2: RAG vs No-RAG ablation
  - Exp 3: Text quality vs literature
  - Exp 4A/4B: Adversarial robustness & OOD generalization
  - Exp 5–9: Calibration, RAG k-ablation, energy efficiency, component ablation

### Section 5: Analysis & Discussion
- **5.1 What Works:** Text generation quality, energy efficiency, modular architecture
- **5.2 Why Classification Fails:** Feature discrimination problem, visual-language misalignment
- **5.3 RAG Paradox:** Retrieved evidence confuses rather than helps
- **5.4 Calibration Breakdown:** Why confidence scores are inverted
- **5.5 Adversarial Vulnerability:** Sycophancy as a fundamental limitation
- **5.6 Out-of-Distribution Failure:** Why OOD generalization collapses
- **5.7 Implications for Medical Deployment:** Trustworthiness, regulatory concerns

### Section 6: Limitations
- Classification accuracy insufficient for clinical use (AUROC 0.532)
- Calibration broken (ECE 0.34, should be <0.1)
- Sycophancy to adversarial prompts (100% attack success)
- RAG quality issues (k=1 optimal suggests poor retrieval)
- Limited dataset diversity (primarily English chest X-rays)

### Section 7: Future Work
- Improve visual-language alignment (architectural modifications)
- Investigate classification head design
- Develop robust retrieval mechanism (hybrid BM25+semantic)
- Post-hoc calibration or temperature scaling
- Adversarial training on grounded evidence
- Clinical validation on held-out datasets

### Section 8: Conclusion
- Summary of MEDDIAG system and design choices
- Key findings (successes and failures)
- Broader impact on interpretable medical AI
- Call for future work on trustworthiness

---

---

# SECTION 1: INTRODUCTION

## 1.1 Background & Motivation

The interpretation of chest radiographs remains a cornerstone of diagnostic radiology, with over 500 million chest X-rays (CXRs) acquired annually worldwide. Despite advances in deep learning, the deployment of automated diagnostic systems in clinical practice faces several critical barriers: (1) computational cost—state-of-the-art vision-language models (VLMs) typically require 20–40 GB of memory for inference, limiting accessibility in resource-constrained settings; (2) lack of interpretability—end-to-end neural networks provide predictions without clinical evidence, reducing trust among radiologists; and (3) generalization failure—models trained on public benchmarks frequently degrade on out-of-distribution data from different institutions or imaging protocols.

Recent advances in parameter-efficient fine-tuning (PEFT) techniques, particularly Low-Rank Adaptation (LoRA) and its quantized variant (QLoRA), have demonstrated that large language models can be adapted to downstream tasks with <5% of the parameters of full fine-tuning, reducing memory requirements from 40 GB to 6 GB or less. Simultaneously, Retrieval-Augmented Generation (RAG) has emerged as a method to ground language model outputs in external knowledge bases, reducing hallucinations and improving factuality. Chain-of-Thought (CoT) prompting has proven effective at eliciting multi-step reasoning, particularly when combined with structured output templates.

## 1.2 Problem Statement & Research Gap

Despite the individual promise of QLoRA, RAG, and CoT, their integration into a unified medical VLM system remains largely unexplored. Existing work treats these components in isolation:
- **QLoRA alone:** Fine-tuning language models but not addressing vision-language alignment or medical evidence grounding
- **RAG alone:** Improving factuality but not adapting the base model to medical domain specifics
- **CoT alone:** Eliciting structured reasoning but without grounding in retrieved clinical evidence

The core research question driving this work is: **Can we jointly optimize vision-language alignment, medical evidence grounding, and cost-efficient computation into a single deployable system that is both accurate and trustworthy?**

Specifically, we identify three key gaps in existing literature:

1. **Vision-Language Alignment in Medical Domains:** Current VLMs (LLaVA, LLaVA-Med) freeze the vision encoder and project features via linear layers or simple MLPs. For medical images, where subtle patterns carry diagnostic significance, this alignment is insufficient. A trainable cross-attention resampler (Perceiver) may better compress high-dimensional vision tokens into disease-relevant semantic space.

2. **Grounding Explanations in Retrievable Evidence:** While RAG has improved factuality in general-domain text generation, its application to multimodal medical reasoning is underdeveloped. Specifically, how should retrieved evidence be integrated with visual features? When does RAG help vs. harm? Does the model learn to weight visual vs. textual evidence appropriately?

3. **Reliability Under Adversarial and Out-of-Distribution Conditions:** Medical systems must be robust to distribution shift (new institutions, protocols) and resistant to prompt-based attacks (e.g., adversarial snippets). Current VLMs lack systematic evaluation in these threat models.

## 1.3 Proposed Approach: MEDDIAG

We introduce **MEDDIAG**, a resource-efficient multimodal chest radiograph diagnostic system that integrates:

1. **Perceiver Resampler Bridge:** A learnable 2-layer cross-attention module that projects frozen EfficientNet-B0 vision features (1280-dim, 49 tokens) to LLM embedding space (3072-dim, 8 tokens). Unlike linear projectors, the Perceiver can selectively attend to disease-relevant visual patterns, improving alignment with LLM semantic space.

2. **Multi-Source FAISS Retrieval:** Nine pluggable knowledge sources (MIMIC-CXR, Semantic Scholar, OpenAlex, CrossRef, Europe PMC, PubMedQA, IU-Xray, ACR/RSNA guidelines) indexed via L2 distance in MiniLM-L6-v2 embedding space (384-dim).

3. **Structured Chain-of-Thought Prompting:** A 4-step diagnostic template (visual observations → clinical interpretation → evidence support → justification) that enforces explicit reasoning and grounds predictions in retrieved snippets.

4. **QLoRA Fine-Tuning:** 4-bit NF4 quantization with double quantization, enabling training on 6 GB VRAM. LoRA adapters (r=16, α=32) target attention projections in LLaMA-3.2-3B, achieving domain adaptation with <2% additional parameters.

5. **Multi-Task Training:** A two-stage pipeline—Stage 1 aligns the projector via next-token prediction on MIMIC reports; Stage 2 jointly optimizes classification accuracy and diagnostic explanation quality via a weighted loss: $\mathcal{L} = 0.65 \cdot \mathcal{L}_{\text{cls}} + 0.35 \cdot \mathcal{L}_{\text{lm}}$.

**System Deployment Context:**
- **Inference VRAM:** 5.5–6.2 GB (6 GB budget enforced)
- **Inference Latency:** ~16–27 seconds per image (single GPU)
- **Energy:** 9.7× more efficient than GPT-4V cloud APIs
- **Deployment Target:** Resource-constrained clinical settings, edge devices, regions with limited cloud access

## 1.4 Core Contributions

This work makes the following research contributions:

1. **Architectural Innovation:** Design and validate a Perceiver Resampler as a trainable bridge between frozen domain-specific vision encoders and general-purpose language models, demonstrating that selective cross-attention improves alignment compared to linear projectors.

2. **Integrated QLoRA+RAG+CoT Framework:** Demonstrate a unified system that combines parameter-efficient fine-tuning, evidence-grounded retrieval, and structured reasoning, with end-to-end training on medical data.

3. **Comprehensive Empirical Evaluation (9 experiments):**
   - **Exp 1:** In-distribution classification on NIH ChestX-ray14 (200 images, AUROC 0.532)
   - **Exp 2–3:** RAG quality ablation and text-generation benchmarking (BERTScore 0.763, competitive with LLaVA-Rad)
   - **Exp 4A–4B:** Adversarial robustness (100% sycophancy) and OOD generalization (50% accuracy on PadChest)
   - **Exp 5–6:** Structural quality (GREEN composite 0.93) and calibration (ECE 0.34, problematic)
   - **Exp 7:** RAG k-ablation revealing that k=1 is optimal (context confusion at k≥3)
   - **Exp 8:** Energy efficiency (9.7× vs. GPT-4V)
   - **Exp 9:** Component ablation study

4. **Honest Failure Analysis:** Unlike prior work that emphasizes positive results, we systematically document where and why MEDDIAG fails—classification accuracy barely above random, vulnerability to adversarial prompts, poor calibration—and propose mechanistic explanations grounded in experimental data.

5. **Resource Efficiency & Deployment Viability:** Demonstrate that competitive text-generation quality can be achieved within a 6 GB VRAM budget, 9.7× more energy-efficient than cloud alternatives, enabling local deployment in resource-constrained settings.

6. **Public Reproducibility:** Full code, architecture diagrams, and experimental logs are released. Multi-source data pipeline (OpenAlex, CrossRef, guidelines) enables offline deployment without API dependencies.

## 1.5 Paper Organization

The remainder of this paper is organized as follows:

- **Section 2 (Related Work):** Surveys parameter-efficient VLMs, RAG, CoT prompting, and medical image analysis.
- **Section 3 (Methodology):** Provides detailed technical descriptions of each component (vision encoder, projector, FAISS retrieval, prompts, training objectives) with formal equations.
- **Section 4 (Experimental Setup & Results):** Describes datasets, baselines, and comprehensive evaluation across 9 experiments.
- **Section 5 (Discussion):** Analyzes successes (text generation, efficiency) and failures (classification, calibration, adversarial robustness), with mechanistic explanations.
- **Section 6 (Limitations):** Acknowledges dataset constraints, clinical applicability concerns, and scope boundaries.
- **Section 7 (Future Work):** Proposes directions for improving classification, calibration, and robustness.
- **Section 8 (Conclusion):** Summarizes findings and implications for medical AI deployment.

---

---

# SECTION 3: METHODOLOGY & SYSTEM ARCHITECTURE

## 3.1 System Overview

MEDDIAG is a unified inference and training system for binary chest X-ray classification (NORMAL vs. ABNORMAL) with interpretable diagnostic explanations. The system decomposes into five primary components:

1. **Vision Encoder ($f_{\text{vis}}$):** Frozen EfficientNet-B0 or ViT-B/16, extracts spatial feature tokens from input images.
2. **Perceiver Resampler ($f_{\text{proj}}$):** Learnable 2-layer cross-attention module that compresses and aligns vision tokens to language model embedding space.
3. **Retrieval-Augmented Generator (RAG):** FAISS index + MiniLM-L6-v2 embeddings, retrieves clinical evidence snippets.
4. **Language Model ($f_{\text{llm}}$):** LLaMA-3.2-3B-Instruct, 4-bit quantized, fine-tuned via LoRA for diagnostic text generation.
5. **Classification Head ($f_{\text{cls}}$):** Learnable binary classifier, parallel to LLM path, predicts NORMAL/ABNORMAL confidence.

**Inference Dataflow:**
$$
\text{Input Image} \xrightarrow{f_{\text{vis}}} \text{Vision Tokens} \xrightarrow{f_{\text{proj}}} \text{Visual Embeds} \xrightarrow{\text{Caption}} \text{Query Embedding}
$$
$$
\text{Query Embedding} \xrightarrow{\text{FAISS}} \text{Retrieved Snippets} \xrightarrow{\text{Prompt}} \text{LLM Input} \xrightarrow{f_{\text{llm}}} \text{Diagnostic Text}
$$
$$
\text{Visual Embeds + RAG Embeds} \xrightarrow{f_{\text{cls}}} \text{Classification Logits} \xrightarrow{\text{Softmax}} \text{Confidence Scores}
$$

**Mathematical Notation:**
- $x \in \mathbb{R}^{3 \times 224 \times 224}$: Input image
- $Z = f_{\text{vis}}(x) \in \mathbb{R}^{N \times d_v}$: Vision tokens (N=49, $d_v$=1280)
- $E_{\text{vis}} = f_{\text{proj}}(Z) \in \mathbb{R}^{K \times d_{\text{llm}}}$: Projected visual embeddings (K=8, $d_{\text{llm}}$=3072)
- $\mathcal{R} = \{(r_i, e_i^{\text{rag}})\}_{i=1}^{k}$: Retrieved snippets and embeddings
- $y_{\text{pred}} = \arg\max f_{\text{llm}}(\mathbf{E}_{\text{prompt}})$: Predicted diagnostic class
- $p_{\text{abnormal}} = \sigma(f_{\text{cls}}(E_{\text{vis}}, \mathcal{E}^{\text{rag}}))$: Classification confidence

---

## 3.2 Vision Encoding Module

The vision encoder compresses raw chest X-ray images into high-dimensional spatial feature tokens. Two architectures are supported; experiments primarily use EfficientNet-B0.

### 3.2.1 EfficientNet-B0 Backbone

**Architecture Parameters:**
- Input: 224×224 RGB image (grayscale converted to 3-channel via replication)
- Output: 49 feature tokens at 1280 dimensions
- Pretraining: ImageNet-1K (frozen during all training stages)

**Forward Process:**
$$
x_{\text{rgb}} = \text{Expand}(x_{\text{gray}}) \in \mathbb{R}^{3 \times 224 \times 224}
$$
$$
x_{\text{norm}} = \frac{x_{\text{rgb}} - \mu_{\text{ImageNet}}}{\sigma_{\text{ImageNet}}}
$$
$$
Z_{\text{conv}} = f_{\text{EfficientNet}}(x_{\text{norm}}) \in \mathbb{R}^{1280 \times 7 \times 7}
$$
$$
Z = \text{Reshape}(Z_{\text{conv}}) \in \mathbb{R}^{49 \times 1280}
$$

**Rationale for Freezing:**
Vision encoders trained on natural images (ImageNet) capture low-to-mid-level features (edges, textures, local patterns) that transfer well to medical images. Freezing reduces memory during fine-tuning (projector training), prevents overfitting on limited medical data, and maintains established feature distributions learned from 1M+ images.

### 3.2.2 Alternative: ViT-B/16

As an ablation, experiments also tested google/vit-base-patch16-224:
- Input: 224×224 image
- Patch size: 16×16 → 196 patches + 1 class token = 197 tokens
- Output dimension: 768
- Preprocessing: Different (no explicit patch projection in our code)

**Note:** EfficientNet-B0 is primary; ViT-B/16 tested for completeness but not featured in final results.

---

## 3.3 Perceiver Resampler: Trainable Projector

The Perceiver Resampler serves as a learned bridge between the frozen vision encoder (1280-dim spatial tokens) and the language model (3072-dim embedding space). Unlike linear projectors $f(Z) = ZW$ used in prior VLMs, the Perceiver uses cross-attention to selectively compress disease-relevant visual patterns.

### 3.3.1 Architecture

**Learned Components:**
- **Latent Queries:** $Q^{(0)} = \{q_k\}_{k=1}^{K} \in \mathbb{R}^{K \times d_{\text{llm}}}$, initialized as $\mathcal{N}(0, 0.02)$, with K=8 queries and $d_{\text{llm}}$=3072.
- **Vision Projection:** $W_v \in \mathbb{R}^{1280 \times 3072}$, projects spatial features to LLM space.
- **Cross-Attention Heads:** 8 attention heads, each attending across all 49 vision tokens.
- **Feed-Forward Networks:** 2 FFN blocks (3072 → 12288 → 3072) with GELU activation.

### 3.3.2 Forward Computation

**Input:** $Z \in \mathbb{R}^{B \times 49 \times 1280}$ (batch of vision tokens)

**Step 1: Project Vision Features**
$$
V = \text{LayerNorm}(Z \cdot W_v) \in \mathbb{R}^{B \times 49 \times 3072}
$$

**Step 2: Stacked Cross-Attention Blocks** (2 layers, $\ell = 1, 2$)

For each block $\ell$:

$$
Q^{(\ell)}_{\text{norm}} = \text{LayerNorm}(Q^{(\ell-1)})
$$
$$
V_{\text{norm}} = \text{LayerNorm}(V)
$$
$$
\text{Attn}^{(\ell)} = \text{MultiheadAttention}(Q^{(\ell)}_{\text{norm}}, V_{\text{norm}}, V_{\text{norm}}, \text{num\_heads}=8)
$$
$$
Q^{(\ell)} := Q^{(\ell-1)} + \text{Attn}^{(\ell)}
$$
$$
Q^{(\ell)} := Q^{(\ell)} + \text{FFN}(\text{LayerNorm}(Q^{(\ell)}))
$$

where MultiheadAttention applies:
$$
\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_8) W^O
$$
$$
\text{head}_i = \text{softmax}\left(\frac{Q W_i^Q (K W_i^K)^T}{\sqrt{d_k}}\right) V W_i^V
$$

with $d_k = 3072/8 = 384$ per head.

**Step 3: Output Normalization and dtype Casting**
$$
E_{\text{vis}} = \text{LayerNorm}(Q^{(2)}) \in \mathbb{R}^{B \times 8 \times 3072}
$$
$$
E_{\text{vis}} = \text{cast\_to\_bf16}(E_{\text{vis}})
$$

**Output:** $E_{\text{vis}} \in \mathbb{R}^{B \times 8 \times 3072 \text{ (bfloat16)}}$

### 3.3.3 Training Dynamics

**Stage 1 (Projector Alignment):**
- Trainable: All Perceiver parameters
- Frozen: Vision encoder, LLM
- Objective: Next-token prediction on MIMIC report text (cross-entropy loss)
- Learning rate: $\eta_1 = 1 \times 10^{-4}$

**Stage 2 (Classification Fine-Tuning):**
- Trainable: Only `input_norm` (LayerNorm of vision projection) at reduced LR
- Frozen: Learned latent queries, attention weights
- Reasoning: Preserve Stage 1 alignment; allow domain-specific scaling of features
- Learning rate: $\eta_2 = 1 \times 10^{-5}$ (10× lower than LoRA)

**Rationale:** The Perceiver acts as a bottleneck (8 tokens) that the LLM must interpret. Over-fine-tuning risks destroying learned alignments. Freezing most parameters during Stage 2 ensures the LLM's LoRA adapters can focus on disease reasoning rather than adapting to shifted feature distributions.

---

## 3.4 Retrieval-Augmented Generation (RAG)

RAG retrieves external clinical evidence from a FAISS index, providing the language model with grounding documents. This section formalizes the retrieval mechanism and knowledge sources.

### 3.4.1 FAISS Indexing

**Index Type:** `IndexFlatL2` (CPU-only, L2 Euclidean distance)
**Embedding Model:** Sentence-transformers/all-MiniLM-L6-v2
- Dimension: 384
- Pre-trained on 215M sentence pairs (AllNLI, Parabank, etc.)
- Frozen at inference (not fine-tuned on medical data)

**Index Compilation:**

Let $\mathcal{D} = \{(s_i, m_i)\}_{i=1}^{N}$ be the knowledge base, where $s_i$ is a snippet text and $m_i$ is metadata (source, URL).

$$
e_i = f_{\text{embed}}(s_i) \in \mathbb{R}^{384}
$$

FAISS stores:
$$
\mathcal{I} = \{(e_i, i, m_i)\}_{i=1}^{N}
$$

**Index Size:** ~24,000 snippets across 9 sources (nominal, varies by sources included)

### 3.4.2 Retrieval at Inference

**Query Generation:**

Given visual embeddings $E_{\text{vis}}$, generate a text caption via LLM:

$$
c = f_{\text{llm}}^{\text{caption}}(E_{\text{vis}}, \text{prompt}_{\text{caption}}, \text{max\_tokens}=64)
$$

The caption prompt instructs the LLM to describe key imaging findings (e.g., "Consolidation visible in right lower lobe").

**Query Embedding:**
$$
q = f_{\text{embed}}(c) \in \mathbb{R}^{384}
$$

**Top-k Retrieval:**
$$
\text{distances}, \text{indices} = \mathcal{I}.\text{search}(q, k)
$$

$$
\mathcal{R} = \{(s_{\text{indices}[i]}, e_{\text{indices}[i]}, \text{distances}[i])\}_{i=1}^{k}
$$

Default $k=5$ (configurable). Snippets ranked by L2 distance (lower is more similar).

**Output:** List of `RetrievedSnippet` objects containing:
- `text`: Snippet content (200–500 tokens)
- `source`: Source tag (e.g., "mimic-cxr", "guidelines", "semantic-scholar")
- `distance`: L2 distance from query embedding
- `embedding`: Reconstructed vector from FAISS

### 3.4.3 Knowledge Sources

Nine pluggable sources, enabled/disabled via config:

| Source | Size | Method | Free? | Notes |
|--------|------|--------|-------|-------|
| MIMIC-CXR | 20K | HuggingFace streaming | ✓* | Public mirror; credentialed access larger |
| Semantic Scholar | 300 | PubMed + API | ✗ (key req'd) | Academic paper abstracts, radiology-focused |
| OpenAlex | 300 | Free API | ✓ | 200M+ paper index, no auth |
| CrossRef | 500 | Free API | ✓ | 140M+ paper index, 1 req/s polite |
| Europe PMC | 300 | Free API | ✓ | PubMed alternative, full-text capable |
| PubMedQA HF | 300 | HuggingFace dataset | ✓ | Chest-filtered QA pairs |
| IU-Xray | 2K | HuggingFace dataset | ✓ | Indiana University CXR reports (3,955 total) |
| Europe PMC Articles | 30 | Full-text API | ✓ | ~10 full-text radiology papers |
| Guidelines | 22 | Static text | ✓ | ACR, RSNA, WHO curated snippets (offline) |

*MIMIC-CXR public mirror available; full dataset requires credentialed access through PhysioNet.

**Rationale for Multi-Source:**
- Reduces dependence on single API (resilience to rate limits, API changes)
- Combines different evidence types (clinical reports, guidelines, literature abstracts)
- Enables offline deployment (Guidelines + IU-Xray + MIMIC mirror)
- Covers diverse clinical contexts and knowledge formats

### 3.4.4 RAG Quality & Ablations

**Exp 2 Finding:** RAG paradoxically *worsens* text generation quality (BERTScore 0.741 with RAG vs. 0.763 without). Possible explanations:

1. **Context Confusion:** Retrieved snippets introduce conflicting information, forcing LLM to balance visual + textual signals.
2. **Context Length Constraints:** Max context ~2000 tokens; RAG consumes tokens, reducing space for reasoning.
3. **Low Retrieval Quality:** FAISS with general-domain embeddings (MiniLM-L6-v2) may not capture medical semantic nuances.

**Exp 7 Finding:** k=1 optimal (BERTScore 0.7146), k≥3 degraded (0.6846–0.6895). This suggests:
- First snippet useful; rest are noise.
- FAISS retrieval precision low for k>1.
- Model better off with focused evidence than confused by multiple options.

**Recommendation for practitioners:** Use k=1 or consider hybrid retrieval (BM25 + semantic embedding) to improve precision.

---

## 3.5 Prompt Engineering & Chain-of-Thought Template

The prompt template is the **single source of truth** for model behavior, applied identically at training and inference.

### 3.5.1 Inference Prompt Structure

**System Prompt (constant):**
```
You are a careful radiology assistant. You analyze chest X-rays and provide 
a binary classification (NORMAL or ABNORMAL) with step-by-step reasoning. 
You must ground your reasoning in the retrieved clinical evidence provided. 
If evidence is insufficient, say so rather than speculate.
```

**User Content (formatted with retrieved evidence):**
```
Chest X-ray image:
<IMAGE>

Retrieved clinical evidence:
[1] {snippet_0_text}
[2] {snippet_1_text}
[3] {snippet_2_text}
[4] {snippet_3_text}
[5] {snippet_4_text}

Task: Examine the image and the evidence above. Produce output in EXACTLY this format:
DIAGNOSIS: <NORMAL or ABNORMAL>
EVIDENCE_USED: <comma-separated list of evidence numbers you relied on, or NONE>
REASONING:
1. Visual observations: <what you see in the image>
2. Clinical interpretation: <map observations to clinical meaning>
3. Evidence support: <quote or paraphrase the exact text from retrieved evidence snippets>
4. Justification: <why the final diagnosis follows from observations + evidence>
```

**Chain-of-Thought Enforcement:** The 4-step reasoning template (visual → interpretation → evidence → justification) enforces structured output, measurable via the GREEN metric (Groundedness, Reasoning, Alignment, Error-free, Numerical).

### 3.5.2 Training Target Templates

During Stage 2, ground-truth diagnostic texts are generated from diverse templates (5 NORMAL variants, 8 ABNORMAL variants) to prevent overfitting to a single style.

**Example NORMAL Template (Variant 1):**
```
DIAGNOSIS: NORMAL
EVIDENCE_USED: 1, 2
REASONING:
1. Visual observations: The lungs are clear bilaterally without focal consolidation, 
   pleural effusion, or pneumothorax. The cardiac silhouette is normal in size and contour. 
   The mediastinum is unremarkable.
2. Clinical interpretation: Absence of acute cardiopulmonary process.
3. Evidence support: Reference [1] states "A normal chest radiograph demonstrates 
   clear bilateral lungs, normal cardiomediastinal silhouette, and no acute findings." 
   Reference [2] provides ACR guidelines confirming these criteria.
4. Justification: All vital structures appear normal. No pathological findings identified. 
   Diagnosis: NORMAL.
```

**Example ABNORMAL Template (Consolidation):**
```
DIAGNOSIS: ABNORMAL
EVIDENCE_USED: 1, 2, 3
REASONING:
1. Visual observations: Right lower lobe consolidation (RLL) is visible on the frontal view, 
   with air bronchograms present. The infiltrate measures approximately 3–4 cm in dimension 
   and shows homogeneous opacity.
2. Clinical interpretation: Consolidation is consistent with pneumonia or other acute 
   pulmonary process.
3. Evidence support: Evidence [1] describes radiographic criteria for pneumonia: "Lobar or 
   segmental consolidation with air bronchograms." Evidence [2] discusses differential 
   diagnoses. Evidence [3] provides epidemiological context.
4. Justification: Clinical and radiographic features support diagnosis of acute consolidation, 
   likely pneumonia. Recommend clinical correlation. Diagnosis: ABNORMAL.
```

**Adversarial Injection (Stage 2 only):** With 20% probability, the last retrieved snippet is replaced with adversarial text designed to push NORMAL images toward ABNORMAL predictions (e.g., "Consolidation suspected" for a normal chest). Model learns to resist sycophancy by observing label mismatches.

### 3.5.3 Output Parsing

Structured output is extracted via regex from model-generated text:

```python
DIAGNOSIS_REGEX = r"DIAGNOSIS:\s*(NORMAL|ABNORMAL)"
EVIDENCE_REGEX = r"EVIDENCE_USED:\s*([^\n]+)"
REASONING_REGEX = r"REASONING:\s*(.+)"
```

If parsing fails, model returns `{diagnosis: "UNPARSEABLE", evidence_used: [], reasoning: ""}`.

---

## 3.6 Classification Head

Parallel to the language modeling path, a learnable binary classification head predicts NORMAL/ABNORMAL confidence. This head is trained only during Stage 2, allowing a direct diagnostic decision without waiting for LLM token generation (optional at inference).

### 3.6.1 Architecture

**Input:**
- Perceiver output: $E_{\text{vis}} \in \mathbb{R}^{B \times 8 \times 3072}$
- RAG embeddings: $\mathcal{E}^{\text{rag}} = \{e_i^{\text{rag}}\}_{i=1}^{k} \in \mathbb{R}^{B \times k \times 384}$

**Processing:**

Step 1: Normalize and pool Perceiver tokens
$$
v = \text{LayerNorm}\left(\frac{1}{K}\sum_{k=1}^{K} e_k^{\text{vis}}\right) \in \mathbb{R}^{B \times 3072}
$$

Step 2: Project and pool RAG embeddings
$$
e_i^{\text{rag, proj}} = W_{\text{rag}} \cdot e_i^{\text{rag}} \in \mathbb{R}^{3072}
$$
$$
r = \text{LayerNorm}\left(\frac{1}{k}\sum_{i=1}^{k} e_i^{\text{rag, proj}}\right) \in \mathbb{R}^{B \times 3072}
$$

Step 3: Concatenate and pass through MLP
$$
x = [v \oplus r] \in \mathbb{R}^{B \times 6144}
$$
$$
h = \text{GELU}(\text{LayerNorm}(W_1 x + b_1)) \in \mathbb{R}^{B \times 512}
$$
$$
\text{logits} = W_2 h + b_2 \in \mathbb{R}^{B \times 2}
$$

Step 4: Temperature scaling (calibration)
$$
p_{\text{abnormal}} = \sigma\left(\frac{\text{logits}[1]}{\tau}\right)
$$

where $\tau$ is a learnable temperature parameter initialized to $\tau = 1.0$ and clamped to $\tau \geq 0.1$.

**Output:** Classification logits and softmax probabilities

### 3.6.2 Training

**Stage 1:** Not present (projector training only)

**Stage 2:** Trained end-to-end with LoRA and input_norm.

Loss: Class-weighted cross-entropy
$$
\mathcal{L}_{\text{cls}} = -\sum_{c \in \{0, 1\}} w_c \cdot y_c \log \sigma(\text{logits}_c)
$$

where $w_0 = 2.0$ (NORMAL) and $w_1 = 1.0$ (ABNORMAL) to counteract class imbalance and prevent model from defaulting to abnormal predictions.

**Inference:** Optional. Can use classification head prediction directly (fast) or combined with LLM output (slower but more interpretable).

---

## 3.7 Training Procedure

MEDDIAG uses a two-stage training pipeline: Stage 1 aligns the projector via next-token prediction; Stage 2 jointly optimizes classification and explanation via multi-task learning.

### 3.7.1 Stage 1: Projector Alignment (Next-Token Prediction)

**Objective:** Train the Perceiver Resampler to align vision tokens with LLM embedding space such that the LLM can generate coherent diagnostic reports.

**Data:** MIMIC-CXR balanced pairs (image + report text), max 5,000 samples.

**Loss Function:**

Standard cross-entropy over report tokens only:
$$
\mathcal{L}_{\text{lm}}^{\text{Stage 1}} = -\frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \log P_\theta(x_t | x_{<t}, E_{\text{vis}})
$$

where $\mathcal{T}$ is the set of report token positions and $x_{<t}$ represents context tokens (prompt + visual embeddings + preceding report tokens).

**Label Masking:** All positions except report tokens have label = -100, preventing loss contamination from prompt/visual tokens.

**Forward Process Per Sample:**

1. Encode image via frozen vision encoder: $Z = f_{\text{vis}}(x)$
2. Project to LLM space: $E_{\text{vis}} = f_{\text{proj}}(Z)$
3. Tokenize prompt prefix (before `<IMAGE>`): `[left_tokens]`
4. Embed left tokens via LLM embedding layer
5. Insert visual embeddings: `[left_embeds | E_vis | ...]`
6. Tokenize and embed right prompt: `[right_tokens | report_tokens]`
7. Concatenate: `[left_embeds | E_vis | right_embeds | report_embeds]`
8. Set labels: `[-100, -100, ..., -100 (for prompt), y1, y2, ..., yn (for report)]`
9. Forward LLM: `out = llm(embeddings, attention_mask, labels)`
10. Loss: `loss = out.loss` (computed by transformers library over labeled positions only)

**Hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max pairs | 5,000 | Single epoch on MIMIC |
| Learning rate | $1 \times 10^{-4}$ | Cosine schedule with warmup |
| Warmup steps | 500 | Linear warmup, then cosine decay |
| Gradient accumulation | 4 | Effective batch size = 4 |
| Gradient clipping | 1.0 | Max norm clipping |
| Weight decay | 0.01 | L2 regularization on trainable params |
| Save interval | Every 500 steps | Checkpoint projector weights |
| Log interval | Every 25 steps | Records loss, LR, VRAM, elapsed time |
| Optimizer | AdamW | Default momentum (β1=0.9, β2=0.999) |

**Output:** `models/projector_stage1.pt` (Perceiver weights after convergence)

### 3.7.2 Stage 2: Classification Fine-Tuning (LoRA + Classification Head)

**Objective:** Fine-tune the frozen LLM (via LoRA adapters) and train a classification head to jointly optimize diagnostic accuracy and explanation quality.

**Data:** Balanced stream from multiple sources:
- MIMIC-CXR: 4,000 samples per epoch, auto-balanced to 50% NORMAL / 50% ABNORMAL
- IU-Xray: 2,500 samples (pre-loaded, split NORMAL/ABNORMAL)
- Pneumonia-Xray (Kermany): 1,841 samples (pre-loaded)
- Interleaved: For each MIMIC sample, yield [MIMIC, IU-Normal, IU-Abnormal, Pneumonia-Normal, Pneumonia-Abnormal] → 5× effective multiplier

**Epochs:** 3 (interleaved data), ~40,000 total training steps per epoch

**Loss Function:**

Weighted multi-task objective:
$$
\mathcal{L}^{\text{Stage 2}} = \alpha \cdot \mathcal{L}_{\text{cls}} + (1 - \alpha) \cdot \mathcal{L}_{\text{lm}}
$$

where $\alpha = 0.65$ (classification 65%, language modeling 35%).

**Classification Loss (with class weights):**
$$
\mathcal{L}_{\text{cls}} = -\sum_{c \in \{0, 1\}} w_c \cdot y_c \log \sigma(\text{logits}_c(E_{\text{vis}}, \mathcal{E}^{\text{rag}}))
$$

with $w_0 = 2.0$ (NORMAL), $w_1 = 1.0$ (ABNORMAL).

**Language Modeling Loss:**
$$
\mathcal{L}_{\text{lm}} = -\frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \log P_\theta(x_t | x_{<t}, E_{\text{vis}}, \mathcal{R})
$$

Same as Stage 1, but with LoRA adapters active and classification target text (diverse templates).

**Forward Process Per Sample:**

1. Stream balanced (LabeledPair): `image, report, label ∈ {0, 1}`
2. Retrieve evidence: `R = retriever.query(caption_embedding, k=5)`
3. Optional adversarial injection (20% prob): Replace last snippet with misleading text
4. Encode image: `Z = f_vis(x)`
5. Project: `E_vis = f_proj(Z)` (frozen)
6. Build chat messages with retrieved snippets
7. Apply chat template: `prompt_text = tokenizer.apply_chat_template(messages)`
8. Splice visual embeddings: `inputs_embeds, attention_mask = splice_visual(prompt_text, E_vis)`
9. Build classification target: `target = build_classification_target(label, step % num_templates)`
10. Append target tokens: `inputs_embeds = [inputs_embeds | target_embeds]`
11. Set labels: `[-100 for prompt tokens | target_ids]`
12. Extract RAG embeddings: `E_rag = [r.embedding for r in R]`
13. Forward LLM: `out = llm(inputs_embeds, attention_mask, labels)`
14. Compute `lm_loss = out.loss`
15. Forward classification head: `logits = cls_head(E_vis, E_rag)`
16. Compute `cls_loss = cross_entropy(logits, label, weight=[2.0, 1.0])`
17. Total loss: `loss = 0.65 * cls_loss + 0.35 * lm_loss`
18. Backward: `loss.backward()`
19. Clip gradients: `clip_grad_norm_(trainable_params, max_norm=1.0)`
20. Optimizer step: `optimizer.step()`
21. Update learning rate: `scheduler.step()`
22. Checkpoint every 250 steps: `save(lora_adapter, cls_head)`

**OOM Recovery (lines 476–499 in stage2_classification.py):**

If backward pass raises `RuntimeError` with "out of memory":
```python
del scaled_loss, loss, cls_loss, lm_loss, out, cls_logits, label_tensor, batch
torch.cuda.empty_cache()
optimizer.zero_grad(set_to_none=True)
continue  # Skip this sample, no update
```

Prevents training crashes on 4–6 GB VRAM; skipped samples are logged.

**Hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max pairs (MIMIC) | 4,000 | Per epoch |
| Epochs | 3 | Interleaved data |
| Learning rate (LoRA) | $2 \times 10^{-4}$ | Slightly higher than Stage 1 |
| Learning rate (input_norm) | $2 \times 10^{-5}$ | 10× lower (preserve alignment) |
| Warmup steps | 50 | Shorter warmup for fine-tuning |
| Gradient accumulation | 8 | Effective batch size = 8 |
| Gradient clipping | 1.0 | Max norm clipping |
| Weight decay | 0.01 | L2 on all trainable params |
| cls_alpha | 0.65 | Loss weighting |
| Max target tokens | 500 | Limit explanation length |
| Save interval | Every 250 steps | Checkpoint LoRA + cls_head |
| Scheduler | Cosine with warmup | `get_cosine_schedule_with_warmup` |
| Optimizer | AdamW | Default momentum |

**Outputs:**
- `models/lora_step{N}/adapter_model.safetensors` (LoRA weights, PEFT format)
- `models/lora_step{N}/adapter_config.json` (LoRA config: r=16, α=32, etc.)
- `models/cls_head.pt` (Classification head weights, PyTorch binary)

---

## 3.8 QLoRA Fine-Tuning Details

The language model is loaded with 4-bit NF4 quantization, enabling training on 6 GB VRAM. LoRA adapters reduce trainable parameters to <2% of the original model.

### 3.8.1 4-Bit NF4 Quantization (BitsAndBytes)

**Configuration (src/llm.py, lines 64–69):**

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                        # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",               # Normalized Float 4
    bnb_4bit_compute_dtype=torch.bfloat16,   # bf16 for computation (not quantized)
    bnb_4bit_use_double_quant=True,          # Quantize the quantization scale
)
```

**Quantization Mechanics:**

Original weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is represented as:
$$
\hat{W} = s \cdot Q_{\text{NF4}}(W / s)
$$

where:
- $Q_{\text{NF4}}$: Maps to 16 discrete values aligned to normal distribution (more precision for common values)
- $s$: Quantization scale, computed as $s = \text{absmax}(W) / \max(Q_{\text{NF4}})$
- **Double quantization:** The scale $s$ is itself quantized:
$$
s = s_2 \cdot Q_{\text{FP4}}(s_1)
$$
This reduces scale overhead by 50%.

**Memory Reduction:**
- Original: $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \times 4 \text{ bytes} = 4 \cdot d_{\text{out}} \cdot d_{\text{in}}$ bytes
- Quantized: $W \times 0.5 + s \times 0.5 = 0.5 \cdot d_{\text{out}} \cdot d_{\text{in}} + \text{overhead}$ bytes
- **Ratio: 8× reduction** (4 bytes → 0.5 bytes per weight)

**Compute Precision:** Despite weights being 4-bit, forward and backward passes use `bfloat16` (no computation in 4-bit int). This maintains numerical stability while reducing memory.

### 3.8.2 LoRA Adapter Configuration

**Config (adapter_config.json, applied during Stage 2):**

```json
{
  "peft_type": "LORA",
  "r": 16,                              // Rank
  "lora_alpha": 32,                     // Scaling α/r = 2.0
  "lora_dropout": 0.05,                 // 5% dropout
  "bias": "none",                       // No bias in adapters
  "target_modules": [
    "q_proj",                           // Query projection in attention
    "v_proj",                           // Value projection
    "k_proj",                           // Key projection
    "o_proj"                            // Output projection
  ],
  "task_type": "CAUSAL_LM",
  "base_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct"
}
```

**LoRA Weight Update:**

For each target module weight $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:
$$
W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A
$$

where:
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$: Input projection (randomly initialized)
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$: Output projection (zeros at init)
- $\alpha / r = 32 / 16 = 2.0$: Scaling factor

**Number of LoRA Parameters:**

LLaMA-3.2-3B has 26 transformer blocks, each with 4 attention heads (q, k, v, o).
$$
\text{Total LoRA params} = 26 \times 4 \times (r \times d_{\text{llm}} + d_{\text{llm}} \times r)
$$
$$
= 26 \times 4 \times (16 \times 3072 + 3072 \times 16) = 26 \times 4 \times 98,304 \approx 10.2M
$$

Compared to ~3.2B total LLaMA parameters, LoRA adds **0.32% trainable parameters**.

**Training Dynamics:**
- Gradients flow through $\Delta W = B A$ only (efficient backprop)
- Base weights $W_0$ frozen (4-bit quantized, no gradient storage)
- After training, adapters can be merged: $W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$ (single inference pass, no adapter overhead)

### 3.8.3 Memory Management & OOM Recovery

**VRAM Budget:** Enforced at 6 GB via `max_memory` dict:
```python
max_memory = {0: "6GB"}  # GPU 0: max 6 GB
```

**Breakdown (Stage 2 training):**
- LLaMA-3.2-3B (4-bit quantized): ~1.8 GB
- LoRA gradient buffers: ~0.5 GB
- Classification head + gradients: ~0.1 GB
- Vision encoder (frozen, inference only): ~0.1 GB
- Temporary activations (forward pass): ~3–3.5 GB
- **Total: ~5.5–6.2 GB**

**OOM Prevention:**
1. Explicit max_memory dict prevents silent CPU offload
2. Gradient accumulation (8 steps): Reduces batch size impact
3. max_target_tokens = 500: Limits sequence length
4. Empty cache periodically: `torch.cuda.empty_cache()` after every 20 steps
5. Graceful skip on OOM: If backward fails, skip sample and zero gradients

**Fallback to CPU (if CUDA fails):**
```python
try:
    model = AutoModelForCausalLM.from_pretrained(..., device_map="auto", ...)
except RuntimeError as e:
    if "CUDA" in str(e):
        print("GPU unavailable, loading on CPU (float32)")
        model = AutoModelForCausalLM.from_pretrained(..., device_map="cpu", dtype=torch.float32)
```

---

## 3.9 Inference Pipeline

Inference is the orchestration of all components into a single diagnostic output.

### 3.9.1 Forward Pass Architecture

**Input:** Chest X-ray image path or PIL Image

**Output:** DiagnosisResult containing:
- `diagnosis`: Predicted class (NORMAL / ABNORMAL / UNPARSEABLE)
- `confidence`: Classification head softmax prob (if available)
- `evidence_used`: List of retrieved snippet indices [1, 2, ...]
- `reasoning`: 4-step diagnostic explanation text
- `retrieved`: List of RetrievedSnippet objects
- `raw_output`: Full LLM-generated text (for debugging)

### 3.9.2 Inference Steps

**1. Load and Preprocess Image**
```python
image = Image.open(image_path).convert("RGB")
pixel_values = vision.preprocess(image)  # (1, 3, 224, 224), normalized
```

**2. Extract Vision Tokens**
```python
with torch.no_grad():
    vision_tokens = vision.model(pixel_values)  # (1, 49, 1280)
```

**3. Project to LLM Space**
```python
with torch.no_grad():
    visual_embeds = projector(vision_tokens)  # (1, 8, 3072, dtype=bf16)
```

**4. Generate Caption (Query for FAISS)**
```python
caption = _caption_image(visual_embeds)  # 64-token text via LLM
# Example: "Bilateral clear lungs, normal cardiac silhouette, no acute findings"
```

**5. Retrieve Evidence**
```python
retrieved = retriever.query(caption, k=5)  # List[RetrievedSnippet]
```

**6. Build Prompt with Retrieved Evidence**
```python
messages = build_chat_messages(
    system_prompt,
    retrieved_snippets,
    task_instruction=CoT_TEMPLATE
)
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
```

**7. Splice Visual Embeddings into Prompt**
```python
inputs_embeds, attention_mask = _splice_visual(
    prompt_text,
    visual_embeds,
    tokenizer,
    device
)
# Result: Prompt tokens concatenated with visual embeddings
```

**8. Generate Diagnostic Text (inputs_embeds-only)**
```python
with torch.no_grad():
    output_ids = llm.model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

**9. Parse Structured Output**
```python
parsed = parse_output(raw_text)
# Returns: {diagnosis, evidence_used, reasoning}
```

**10. (Optional) Classification Head**
```python
if cls_head is not None:
    rag_embeddings = torch.stack([r.embedding for r in retrieved])  # (1, 5, 384)
    with torch.no_grad():
        logits = cls_head(visual_embeds, rag_embeddings)  # (1, 2)
        probs = torch.softmax(logits, dim=-1)
        confidence = probs[0, 1].item()  # P(ABNORMAL)
        cls_diagnosis = "ABNORMAL" if confidence >= 0.5 else "NORMAL"
else:
    cls_diagnosis = None
```

**11. Return Unified Result**
```python
return DiagnosisResult(
    diagnosis=cls_diagnosis or parsed["diagnosis"],  # ClassHead takes precedence
    confidence=confidence,
    evidence_used=parsed["evidence_used"],
    reasoning=parsed["reasoning"],
    retrieved=retrieved,
    raw_output=raw_text,
    latency=elapsed_time,
)
```

### 3.9.3 Key Invariants

1. **inputs_embeds-only forward:** Never pass `input_ids` to LLM. Visual tokens are embedded tensors, not discrete tokens.
2. **Streaming data:** MIMIC dataset not cached locally (streamed from HuggingFace); enables inference on machines with <100 GB storage.
3. **Consistent prompts:** Prompt structure at training and inference is identical (enforced via `prompts.py`).
4. **Deterministic (greedy) decoding:** `do_sample=False, temperature=0` ensures reproducible outputs.
5. **Offline capability:** Guidelines + MIMIC mirror + IU-Xray enable deployment without APIs; optional API sources (Semantic Scholar, OpenAlex) enhance quality if available.

---

## 3.10 Summary of Methodological Innovations

| Component | Innovation | Rationale |
|-----------|-----------|-----------|
| **Perceiver Resampler** | Cross-attention instead of linear projection | Selective compression of disease-relevant patterns |
| **Multi-Source RAG** | 9 pluggable knowledge sources | Resilience, offline capability, diverse evidence types |
| **Structured CoT** | 4-step template enforced via regex | Measurable reasoning quality (GREEN metric) |
| **QLoRA + Double Quant** | 4-bit NF4, bf16 compute, 8× memory reduction | 6 GB VRAM training feasible on consumer GPUs |
| **Multi-Task Loss** | cls + lm blended (0.65 / 0.35) | Joint optimization of classification and explanation |
| **Adversarial Training** | 20% prompt injection during Stage 2 | Improve robustness to sycophancy (attempted mitigation) |
| **Temperature Scaling** | Learnable τ in classification head | Post-hoc calibration (partial remedy for ECE issues) |
| **Two-Stage Training** | Alignment → Fine-tuning | Separate concerns: projector ↔ domain reasoning |

---

**[End of Sections 1–3]**

Sections 4–8 to follow: Experimental Setup, Detailed Results, Analysis & Discussion, Limitations, Future Work, Conclusion.

