"""Retrain ONLY the ClassificationHead on frozen (vision + projector) features.

This is cheap and fully decoupled from the LLM / LoRA adapter: the classification
path is vision(frozen) -> projector(frozen) -> cls_head, and the LLM never enters
it. So report quality (GREEN), VRAM, and latency are UNCHANGED — only classification
AUROC / F1 / sycophancy-resistance move. No LLM is even loaded here.

Three improvements over the head learned during Stage-2:
  * focal loss + pos_weight   -> up-weights hard / missed abnormals (recall, AUROC)
  * attention pooling          -> learned per-token visual weighting (--pool attn)
  * adversarial RAG negatives  -> misleading evidence paired with the CORRECT label,
                                  so the head learns visual primacy and stops blindly
                                  following retrieved text (fixes the exp4a 100%
                                  sycophancy failure). The adversarial snippet is
                                  embedded and written into the head's RAG tensor.

Usage:
    python -m experiments.retrain_cls_head \
        --projector models/projector_stage1.pt \
        --old-head  models/cls_head.pt \
        --out       models/cls_head_retrained.pt \
        --max-samples 2000 --pool attn \
        --pos-weight 2.5 --focal-gamma 2.0 --adv-prob 0.30
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

# Reuse the exact loss (focal + pos_weight) used in Stage-2.
from experiments.stage2_classification import classification_loss
from src.classification_head import ClassificationHead
from src.config import Config
from src.data.balanced_stream import balanced_mimic_stream
from src.prompts import maybe_inject_adversarial_ex
from src.retrieval import Retriever
from src.vision import VisionEncoder
from src.projector import PerceiverResampler


# ── Cached-feature sample ─────────────────────────────────────────────────────
class _Cache:
    """Holds frozen (perceiver_out, rag, label) tensors on CPU."""

    def __init__(self) -> None:
        self.pv: list[torch.Tensor] = []   # each (1, K, D)
        self.rag: list[torch.Tensor] = []  # each (1, k, rag_dim)
        self.y: list[int] = []

    def add(self, pv: torch.Tensor, rag: torch.Tensor, y: int) -> None:
        self.pv.append(pv.cpu()); self.rag.append(rag.cpu()); self.y.append(int(y))

    def __len__(self) -> int:
        return len(self.y)


def _rag_for(pair, retriever, config, adv_rng):
    """Retrieve + (optionally) inject an adversarial snippet into BOTH the text and
    the RAG embedding tensor. Returns (rag_tensor, was_adversarial)."""
    first_sentence = pair.report.split(".")[0].strip()[:150]
    retrieved = retriever.query(first_sentence, k=config.retrieval_top_k)
    adv_text = None
    if adv_rng is not None:
        _snips = [r.text for r in retrieved]
        _snips, adv_text = maybe_inject_adversarial_ex(_snips, adv_rng)
    embs = [r.embedding for r in retrieved if r.embedding is not None]
    if embs:
        rag = torch.from_numpy(np.stack(embs)).unsqueeze(0).float()
        if adv_text is not None:
            adv_emb = retriever.embedder.encode([adv_text], convert_to_numpy=True)[0]
            rag[0, -1] = torch.from_numpy(np.asarray(adv_emb, dtype=np.float32))
    else:
        rag = torch.zeros(1, config.retrieval_top_k, config.embedder_dim)
    return rag, adv_text is not None


def build_cache(pairs, vision, projector, retriever, config, device, adv_rng):
    """Encode labeled pairs into frozen (perceiver_out, rag, label) tuples."""
    cache = _Cache()
    for i, pair in enumerate(pairs):
        rag, _ = _rag_for(pair, retriever, config, adv_rng)
        pv = vision.preprocess(pair.image).to(device)
        with torch.no_grad():
            po = projector(vision(pv))               # (1, K, D_llm)
        cache.add(po, rag, 1 if pair.label == "ABNORMAL" else 0)
        if (i + 1) % 100 == 0:
            print(f"  cached {i + 1} samples")
    return cache


# ── Train / eval on a cache (testable in isolation) ──────────────────────────
def train_head(cache: _Cache, config: Config, pool: str, pos_weight: float,
               focal_gamma: float, epochs: int = 6, lr: float = 1e-3,
               device: str = "cpu", seed: int = 0) -> ClassificationHead:
    torch.manual_seed(seed)
    head = ClassificationHead(llm_dim=config.llm_hidden_dim, rag_dim=config.embedder_dim,
                              hidden_dim=config.cls_hidden_dim, pool=pool).to(device)
    opt = AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    idx = list(range(len(cache)))
    for ep in range(epochs):
        random.Random(seed + ep).shuffle(idx)
        tot = 0.0
        head.train()
        for j in idx:
            po = cache.pv[j].to(device); rag = cache.rag[j].to(device)
            y = torch.tensor([cache.y[j]], device=device)
            logits = head(po, rag)
            loss = classification_loss(logits, y, pos_weight=pos_weight, focal_gamma=focal_gamma)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"  epoch {ep + 1}/{epochs}  loss={tot / max(1, len(idx)):.4f}")
    head.eval()
    return head


@torch.no_grad()
def _probs(head: ClassificationHead, cache: _Cache, device: str = "cpu") -> np.ndarray:
    out = []
    for po, rag in zip(cache.pv, cache.rag):
        logits = head(po.to(device), rag.to(device))
        out.append(torch.softmax(logits, dim=-1)[0, 1].item())
    return np.asarray(out)


def auroc(head: ClassificationHead, cache: _Cache, device: str = "cpu") -> float:
    """Rank-based AUROC (no sklearn dependency)."""
    p = _probs(head, cache, device); y = np.asarray(cache.y)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = p.argsort()
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def resistance(head: ClassificationHead, normal_adv_cache: _Cache, thr: float = 0.5,
               device: str = "cpu") -> float:
    """Fraction of NORMAL samples (with adversarial RAG) still predicted NORMAL."""
    p = _probs(head, normal_adv_cache, device)
    return float((p < thr).mean()) if len(p) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrain only the ClassificationHead (focal + attn + adversarial-RAG)")
    ap.add_argument("--projector", type=Path, default=Path("models/projector_stage1.pt"))
    ap.add_argument("--old-head", type=Path, default=Path("models/cls_head.pt"))
    ap.add_argument("--out", type=Path, default=Path("models/cls_head_retrained.pt"))
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--pool", choices=["mean", "attn"], default="attn")
    ap.add_argument("--pos-weight", type=float, default=2.5)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--adv-prob", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    config = Config()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[retrain] device={device}  pool={args.pool}  pos_weight={args.pos_weight}  "
          f"focal_gamma={args.focal_gamma}  adv_prob={args.adv_prob}")

    # Frozen feature extractors + retriever. NO LLM is loaded.
    vision = VisionEncoder(config); vision.to(device)
    projector = PerceiverResampler(config)
    projector.load_state_dict(torch.load(args.projector, map_location="cpu"))
    projector.to(device).eval()
    retriever = Retriever(config); retriever.load()

    # Adversarial probability override for this run.
    import src.prompts as _p
    _p.ADVERSARIAL_INJECTION_PROB = float(args.adv_prob)
    adv_rng = random.Random(42)

    pairs = list(balanced_mimic_stream(config, max_pairs=args.max_samples))
    random.Random(0).shuffle(pairs)
    n_val = int(len(pairs) * args.val_frac)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"[retrain] train={len(train_pairs)}  val={len(val_pairs)}")

    print("[retrain] caching TRAIN (adversarial on)...")
    train_cache = build_cache(train_pairs, vision, projector, retriever, config, device, adv_rng)
    print("[retrain] caching VAL clean (adversarial off)...")
    val_clean = build_cache(val_pairs, vision, projector, retriever, config, device, None)
    print("[retrain] caching VAL sycophancy (NORMAL only, adversarial forced)...")
    _p.ADVERSARIAL_INJECTION_PROB = 1.0
    normals = [p for p in val_pairs if p.label != "ABNORMAL"]
    val_syco = build_cache(normals, vision, projector, retriever, config, device, random.Random(7))

    # Train the new head.
    print("[retrain] training new head...")
    new_head = train_head(train_cache, config, args.pool, args.pos_weight,
                          args.focal_gamma, epochs=args.epochs, device=device)

    # Load the old head for A/B (pool inferred from its state_dict).
    old_state = torch.load(args.old_head, map_location="cpu")
    old_pool = "attn" if any("vis_attn" in k for k in old_state) else "mean"
    old_head = ClassificationHead(llm_dim=config.llm_hidden_dim, rag_dim=config.embedder_dim,
                                  hidden_dim=config.cls_hidden_dim, pool=old_pool).to(device)
    old_head.load_state_dict(old_state); old_head.eval()

    print("\n================  A/B (val)  ================")
    print(f"{'metric':<26}{'OLD':>10}{'NEW':>10}")
    for name, fn in (("AUROC (clean val)", lambda h: auroc(h, val_clean, device)),
                     ("Sycophancy resistance", lambda h: resistance(h, val_syco, device=device))):
        print(f"{name:<26}{fn(old_head):>10.4f}{fn(new_head):>10.4f}")

    torch.save(new_head.state_dict(), args.out)
    print(f"\n[retrain] saved new head -> {args.out}")
    print("Evaluate the full system with:  --lora-adapter ... and config.cls_head_path pointing here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
