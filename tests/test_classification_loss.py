"""Unit tests for the Run-4 recall-oriented classification loss (no GPU)."""
import torch
import torch.nn.functional as F

from experiments.stage2_classification import classification_loss


def test_default_matches_label_smoothed_ce():
    # defaults (pos_weight=1.0, focal_gamma=0.0) must reproduce Run-3 behavior exactly
    logits = torch.tensor([[1.5, -0.5]])
    for label in (0, 1):
        t = torch.tensor([label])
        got = classification_loss(logits, t)
        exp = F.cross_entropy(logits, t, label_smoothing=0.1)
        assert torch.allclose(got, exp)


def test_pos_weight_scales_abnormal_only():
    logits = torch.tensor([[0.3, 0.7]])
    base_abn = classification_loss(logits, torch.tensor([1]), pos_weight=1.0)
    up_abn = classification_loss(logits, torch.tensor([1]), pos_weight=3.0)
    assert torch.allclose(up_abn, 3.0 * base_abn)          # ABNORMAL loss tripled

    base_nrm = classification_loss(logits, torch.tensor([0]), pos_weight=1.0)
    up_nrm = classification_loss(logits, torch.tensor([0]), pos_weight=3.0)
    assert torch.allclose(up_nrm, base_nrm)                # NORMAL loss unchanged


def test_focal_downweights_easy_examples():
    t = torch.tensor([1])
    easy = torch.tensor([[-3.0, 3.0]])   # confidently correct (p_true high)
    hard = torch.tensor([[-0.1, 0.1]])   # barely correct (p_true ~0.55)
    # focal shrinks the easy example's loss far more than the hard one's
    ratio_easy = classification_loss(easy, t, focal_gamma=2.0) / F.cross_entropy(easy, t)
    ratio_hard = classification_loss(hard, t, focal_gamma=2.0) / F.cross_entropy(hard, t)
    assert ratio_easy < ratio_hard
    assert ratio_easy < 0.1               # near-certain example is strongly down-weighted


def test_loss_is_finite_and_differentiable():
    logits = torch.tensor([[0.2, -0.4]], requires_grad=True)
    loss = classification_loss(logits, torch.tensor([1]), pos_weight=2.0, focal_gamma=2.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_config_defaults_preserve_run3_behavior():
    from src.config import Config
    c = Config()
    assert c.cls_pos_weight == 1.0
    assert c.cls_focal_gamma == 0.0
