"""The implementation matches the MEDDIAG paper (DRAFT JOURNAL PAPER, §3 / Table 2).

Each test names the section it pins, so a future change that drifts from the
paper fails loudly instead of silently changing the method.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.config import Config


# §3.8 Classification head ---------------------------------------------------
def test_head_is_paper_architecture_by_default():
    from src.classification_head import ClassificationHead
    h = ClassificationHead()                                   # 3072 / 384 / 512 defaults
    assert h.arch == "paper" and h.pool == "mean"
    lin1, act, lin2 = h.mlp
    assert (lin1.in_features, lin1.out_features) == (3072 + 384, 512)   # concat = 3456
    assert isinstance(act, torch.nn.GELU)
    assert (lin2.in_features, lin2.out_features) == (512, 2)
    assert not hasattr(h, "rag_proj")
    out = h(torch.randn(2, 8, 3072), torch.randn(2, 5, 384))
    assert out.shape == (2, 2)


def test_head_mean_pools_both_branches():
    from src.classification_head import ClassificationHead
    h = ClassificationHead(llm_dim=4, rag_dim=3, hidden_dim=5)
    vis, rag = torch.randn(1, 8, 4), torch.randn(1, 5, 3)
    fused = torch.cat([vis.mean(1), rag.mean(1)], dim=1)
    assert torch.allclose(h(vis, rag), h.mlp(fused) / h.temperature.clamp(min=0.1))


def test_from_state_dict_loads_both_layouts():
    from src.classification_head import ClassificationHead
    for arch, pool in (("paper", "mean"), ("projected", "attn")):
        src = ClassificationHead(llm_dim=8, rag_dim=4, hidden_dim=6, arch=arch, pool=pool)
        h = ClassificationHead.from_state_dict(src.state_dict(), 8, 4, 6)
        assert (h.arch, h.pool) == (arch, pool)


# §3.9 / Table 2 loss -----------------------------------------------------------
def test_config_defaults_are_paper_values(monkeypatch):
    for k in ("MEDDIAG_CLS_NORMAL_WEIGHT", "MEDDIAG_CLS_POS_WEIGHT", "MEDDIAG_CLS_FOCAL_GAMMA",
              "MEDDIAG_CLS_LABEL_SMOOTHING", "MEDDIAG_CLS_ARCH", "MEDDIAG_CLS_POOL",
              "MEDDIAG_CLS_ALPHA", "MEDDIAG_S2_EXTRA_SOURCES", "MEDDIAG_VISION_FINETUNE",
              "MEDDIAG_MAX_VRAM_GB"):
        monkeypatch.delenv(k, raising=False)
    c = Config()
    assert (c.cls_normal_weight, c.cls_pos_weight) == (2.0, 1.0)      # NORMAL / ABNORMAL
    assert c.cls_focal_gamma == 0.0 and c.cls_label_smoothing == 0.0  # plain class-weighted CE
    assert (c.cls_arch, c.cls_pool) == ("paper", "mean")
    assert c.cls_alpha == 0.65                                        # λ_cls (λ_LM = 0.35)
    assert c.stage2_extra_sources == frozenset()                      # Stage-2 on MIMIC only
    assert c.vision_finetune is False                                 # §3.3 frozen encoder
    assert c.max_vram_gb == 6.0                                       # project budget
    assert c.num_visual_tokens == 8 and c.llm_hidden_dim == 3072 and c.vision_hidden_dim == 1024
    assert c.retrieval_top_k == 5 and c.embedder_dim == 384


def test_class_weighted_ce_weights_normal_twice_abnormal():
    import torch.nn.functional as F
    from experiments.stage2_classification import classification_loss
    logits = torch.tensor([[0.3, -0.2]])
    for label, w in ((0, 2.0), (1, 1.0)):
        t = torch.tensor([label])
        got = classification_loss(logits, t, normal_weight=2.0, pos_weight=1.0,
                                  focal_gamma=0.0, label_smoothing=0.0)
        assert torch.allclose(got, w * F.cross_entropy(logits, t))


def test_lora_and_adversarial_hyperparameters():
    import experiments.stage2_classification as s2
    from src.prompts import ADVERSARIAL_INJECTION_PROB, ADVERSARIAL_SNIPPETS
    assert (s2.LORA_R, s2.LORA_ALPHA, s2.LORA_DROPOUT) == (16, 32, 0.05)
    assert s2.LORA_TARGET_MODULES == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert ADVERSARIAL_INJECTION_PROB == 0.20 and len(ADVERSARIAL_SNIPPETS) == 8   # §3.7, App. C


def test_pipeline_script_uses_paper_schedule_and_kb():
    from pathlib import Path
    sh = (Path(__file__).resolve().parent.parent / "run_pipeline.sh").read_text(encoding="utf-8")
    assert "WARMUP_S2=100" in sh                                     # Table 2 warm-up
    assert "--kb paper --max-mimic 20000" in sh                      # Table 1 knowledge base


# §3.10.1 labels & held-out evaluation ------------------------------------------
@pytest.mark.parametrize("text,label", [
    # the two real IU-Xray reports that keyword-matching had mislabelled
    ("radiographic findings suggestive of emphysema. calcified granuloma. normal heart size.", "ABNORMAL"),
    ("no acute cardiopulmonary process. the cardiomediastinal silhouette is within normal limits. "
     "lungs are clear without focal consolidation. no visualized pneumothorax or large pleural effusion.", "NORMAL"),
    ("Right lower lobe consolidation concerning for pneumonia.", "ABNORMAL"),
    ("Lungs are clear. No emphysema.", "NORMAL"),
    # pseudo-negation: the effusion is still present
    ("No change in the small right pleural effusion.", "ABNORMAL"),
    ("No interval change in left basilar atelectasis.", "ABNORMAL"),
    # termination word ends the negation scope
    ("No pneumothorax, but there is a large left pleural effusion.", "ABNORMAL"),
    # "no" must be a whole word: "nodule"/"normal" are not negation cues
    ("Normal heart size. Right upper lobe nodule.", "ABNORMAL"),
    # one affirmed mention outweighs a negated one
    ("No right effusion. Small left pleural effusion.", "ABNORMAL"),
])
def test_report_labeler(text, label):
    from src.data.labeler import assign_label
    assert assign_label(text) == label


def test_mimic_partition_is_disjoint():
    from datasets import Dataset
    from src.data.mimic_split import mimic_part
    ds = Dataset.from_dict({"i": list(range(20))}).to_iterable_dataset()
    train = [r["i"] for r in mimic_part(ds, "train", 15)]
    held = [r["i"] for r in mimic_part(ds, "heldout", 15)]
    assert train == list(range(15)) and held == list(range(15, 20))
    with pytest.raises(ValueError):
        mimic_part(ds, "test", 15)


# §4.7 temperature scaling -------------------------------------------------------
def test_temperature_scaling_fixes_overconfidence():
    import random
    from experiments.run_experiments import apply_temperature, fit_temperature
    from src.metrics import expected_calibration_error
    rng = random.Random(0)
    # true P(abnormal) = q, but the model reports an overconfident sharpened version
    ys, ps = [], []
    for _ in range(2000):
        q = rng.uniform(0.05, 0.95)
        ys.append(int(rng.random() < q))
        ps.append(apply_temperature([q], 1 / 3)[0])          # T = 1/3 -> overconfident
    t = fit_temperature(ys[:1000], ps[:1000])
    assert 2.4 < t < 3.6                                       # recovers T* ≈ 3
    raw = expected_calibration_error(ys[1000:], ps[1000:])
    fixed = expected_calibration_error(ys[1000:], apply_temperature(ps[1000:], t))
    assert fixed < raw / 2 and fixed < 0.1


@pytest.mark.parametrize("text,label", [
    ("No acute cardiopulmonary findings. The heart is mildly enlarged. Lung volumes are low.", None),  # contradictory -> skip
    ("The heart is mildly enlarged. Small left effusion.", "ABNORMAL"),
    ("Lungs are clear. Heart size is not enlarged.", "NORMAL"),
    ("Lungs are clear. The heart is not enlarged and there is no effusion.", "NORMAL"),
])
def test_report_labeler_cardiac_size(text, label):
    from src.data.labeler import assign_label
    assert assign_label(text) == label
