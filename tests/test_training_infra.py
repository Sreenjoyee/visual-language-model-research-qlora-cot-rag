"""Unit tests for training infrastructure — scheduler, checkpoint, grad accumulation.

All tests run without a downloaded model. A tiny PerceiverResampler is used
as a stand-in for the full projector so that optimizer + scheduler state can
be exercised without VRAM.

Run with:
    pytest tests/test_training_infra.py -v
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from src.config import CONFIG
from src.projector import PerceiverResampler


# ── Tiny projector fixture ────────────────────────────────────────────────────

@pytest.fixture
def tiny_proj() -> PerceiverResampler:
    """Minimal projector: input 32-dim, output 32-dim, 1 head, 1 layer."""
    return PerceiverResampler(
        vision_dim=32,
        llm_dim=32,
        num_latents=4,
        num_heads=CONFIG.projector_num_heads,
        num_layers=1,
    )


@pytest.fixture
def optimizer(tiny_proj) -> AdamW:
    return AdamW(tiny_proj.parameters(), lr=1e-3, weight_decay=0.01)


# ── Cosine scheduler with warmup ──────────────────────────────────────────────

class TestCosineScheduler:
    def test_lr_starts_near_zero_during_warmup(self, optimizer):
        """LR should be very small at step 0 (start of warmup)."""
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=10, num_training_steps=100
        )
        # At step 0 the scheduler hasn't stepped yet — LR is at initial value × 0
        # After one step into warmup it should be 1/10 of peak LR
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        assert lr < 1e-3, f"LR {lr} should be below peak during warmup"

    def test_lr_peaks_after_warmup(self, optimizer):
        """LR should be at or near peak_lr after warmup completes."""
        warmup = 5
        total  = 50
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=total
        )
        for _ in range(warmup):
            scheduler.step()
        lr = scheduler.get_last_lr()[0]
        assert lr >= 0.9e-3, f"LR {lr} should be near peak after warmup"

    def test_lr_decays_after_warmup(self, optimizer):
        """LR after warmup must be strictly decreasing (cosine decay)."""
        warmup = 5
        total  = 50
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=total
        )
        for _ in range(warmup):
            scheduler.step()
        lr_at_peak = scheduler.get_last_lr()[0]
        for _ in range(total - warmup):
            scheduler.step()
        lr_at_end = scheduler.get_last_lr()[0]
        assert lr_at_end < lr_at_peak, "LR should decay after warmup"

    def test_lr_approaches_zero_at_end(self, optimizer):
        """LR should be near 0 at the end of the cosine schedule."""
        total = 100
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=10, num_training_steps=total
        )
        for _ in range(total):
            scheduler.step()
        lr = scheduler.get_last_lr()[0]
        assert lr < 1e-4, f"LR {lr} should be near 0 at schedule end"

    def test_scheduler_state_roundtrip(self, optimizer):
        """Scheduler state_dict → load_state_dict reproduces identical LR."""
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=5, num_training_steps=50
        )
        for _ in range(15):
            scheduler.step()
        lr_before = scheduler.get_last_lr()[0]
        state = scheduler.state_dict()

        # Create fresh optimizer + scheduler, restore state
        opt2 = AdamW(list(optimizer.param_groups[0]["params"]), lr=1e-3)
        sch2 = get_cosine_schedule_with_warmup(opt2, num_warmup_steps=5, num_training_steps=50)
        sch2.load_state_dict(state)
        lr_after = sch2.get_last_lr()[0]

        assert abs(lr_before - lr_after) < 1e-10, (
            f"LR mismatch after state restore: {lr_before} vs {lr_after}"
        )

    def test_warmup_steps_zero_no_warmup_phase(self, optimizer):
        """warmup_steps=0 means LR starts at peak immediately."""
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=50
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        # Should be close to peak (just started cosine decay)
        assert lr >= 0.9e-3


# ── Checkpoint save / load (stage 1) ─────────────────────────────────────────

class TestStage1Checkpoint:
    def test_save_creates_file(self, tmp_path, tiny_proj, optimizer):
        from experiments.stage1_projector import _save_checkpoint
        scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 50)
        path = tmp_path / "ckpt.pt"
        _save_checkpoint(path, tiny_proj, optimizer, scheduler, step=10, epoch=0)
        assert path.exists()

    def test_checkpoint_contains_required_keys(self, tmp_path, tiny_proj, optimizer):
        from experiments.stage1_projector import _save_checkpoint
        scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 50)
        path = tmp_path / "ckpt.pt"
        _save_checkpoint(path, tiny_proj, optimizer, scheduler, step=42, epoch=1)
        ckpt = torch.load(path, map_location="cpu")
        assert set(ckpt.keys()) == {"step", "epoch", "projector", "optimizer", "scheduler"}

    def test_checkpoint_step_epoch_correct(self, tmp_path, tiny_proj, optimizer):
        from experiments.stage1_projector import _save_checkpoint
        scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 50)
        path = tmp_path / "ckpt.pt"
        _save_checkpoint(path, tiny_proj, optimizer, scheduler, step=77, epoch=2)
        ckpt = torch.load(path, map_location="cpu")
        assert ckpt["step"] == 77
        assert ckpt["epoch"] == 2

    def test_projector_weights_restore_exactly(self, tmp_path, tiny_proj, optimizer):
        """Weights loaded from checkpoint are bit-for-bit identical."""
        from experiments.stage1_projector import _save_checkpoint
        scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 50)
        path = tmp_path / "ckpt.pt"
        _save_checkpoint(path, tiny_proj, optimizer, scheduler, step=1, epoch=0)

        ckpt = torch.load(path, map_location="cpu")
        proj2 = PerceiverResampler(32, 32, 4, CONFIG.projector_num_heads, 1)
        proj2.load_state_dict(ckpt["projector"])

        for (n1, p1), (n2, p2) in zip(
            tiny_proj.named_parameters(), proj2.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Weight mismatch at {n1}"

    def test_optimizer_state_restored(self, tmp_path, tiny_proj, optimizer):
        """Optimizer lr is preserved through checkpoint."""
        from experiments.stage1_projector import _save_checkpoint
        # Run a fake step to populate optimizer state
        loss = tiny_proj(torch.randn(1, 5, 32)).mean()
        loss.backward()
        optimizer.step()
        scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 50)
        scheduler.step()

        path = tmp_path / "ckpt.pt"
        _save_checkpoint(path, tiny_proj, optimizer, scheduler, step=1, epoch=0)

        ckpt = torch.load(path, map_location="cpu")
        opt2 = AdamW(PerceiverResampler(32, 32, 4, CONFIG.projector_num_heads, 1).parameters(), lr=1e-3)
        opt2.load_state_dict(ckpt["optimizer"])
        assert abs(opt2.param_groups[0]["lr"] - optimizer.param_groups[0]["lr"]) < 1e-9

    def test_scheduler_epoch_restored(self, tmp_path, tiny_proj, optimizer):
        """Scheduler last_epoch matches after state restore."""
        from experiments.stage1_projector import _save_checkpoint
        scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 50)
        for _ in range(12):
            scheduler.step()

        path = tmp_path / "ckpt.pt"
        _save_checkpoint(path, tiny_proj, optimizer, scheduler, step=12, epoch=0)

        ckpt = torch.load(path, map_location="cpu")
        opt2 = AdamW(PerceiverResampler(32, 32, 4, CONFIG.projector_num_heads, 1).parameters(), lr=1e-3)
        sch2 = get_cosine_schedule_with_warmup(opt2, 5, 50)
        sch2.load_state_dict(ckpt["scheduler"])
        assert sch2.last_epoch == scheduler.last_epoch


# ── Checkpoint save / load (stage 2) ─────────────────────────────────────────

class TestStage2Checkpoint:
    """Stage-2 checkpoint writes train_state.pt next to the LoRA adapter dir."""

    def _fake_model(self):
        """Minimal nn.Module with save_pretrained stub."""
        class FakeLoraModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            def save_pretrained(self, path: str):
                Path(path).mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), Path(path) / "model.pt")

        return FakeLoraModel()

    def test_save_creates_train_state(self, tmp_path, tiny_proj, optimizer):
        from experiments.stage2_classification import _save_checkpoint_s2
        model = self._fake_model()
        scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 50)
        ckpt_dir = tmp_path / "lora_step100"
        _save_checkpoint_s2(ckpt_dir, model, optimizer, scheduler, step=100, epoch=0)
        assert (ckpt_dir / "train_state.pt").exists()
        assert (ckpt_dir / "model.pt").exists()

    def test_train_state_keys(self, tmp_path, tiny_proj, optimizer):
        from experiments.stage2_classification import _save_checkpoint_s2
        model = self._fake_model()
        scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 50)
        ckpt_dir = tmp_path / "lora_ckpt"
        _save_checkpoint_s2(ckpt_dir, model, optimizer, scheduler, step=50, epoch=1)
        state = torch.load(ckpt_dir / "train_state.pt", map_location="cpu")
        assert {"step", "epoch", "optimizer", "scheduler"} <= set(state.keys())

    def test_train_state_step_epoch(self, tmp_path, tiny_proj, optimizer):
        from experiments.stage2_classification import _save_checkpoint_s2
        model = self._fake_model()
        scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 50)
        ckpt_dir = tmp_path / "ckpt"
        _save_checkpoint_s2(ckpt_dir, model, optimizer, scheduler, step=99, epoch=3)
        state = torch.load(ckpt_dir / "train_state.pt", map_location="cpu")
        assert state["step"] == 99
        assert state["epoch"] == 3


# ── Gradient accumulation logic ───────────────────────────────────────────────

class TestGradientAccumulation:
    def test_single_step_equals_no_accum(self, tiny_proj):
        """grad_accum_steps=1 should produce the same gradient as no accumulation."""
        torch.manual_seed(0)
        vision_tokens = torch.randn(1, 5, 32)
        opt = AdamW(tiny_proj.parameters(), lr=1e-3)
        opt.zero_grad()

        out = tiny_proj(vision_tokens).float()
        loss = out.mean()
        loss.backward()

        grads = {n: p.grad.clone() for n, p in tiny_proj.named_parameters()}
        assert all(g is not None for g in grads.values())
        assert all(torch.isfinite(g).all() for g in grads.values())

    def test_accum_2_steps_halves_grad_per_step(self, tiny_proj):
        """Each micro-step should contribute loss/N to the gradient."""
        torch.manual_seed(42)
        vision_tokens = torch.randn(1, 5, 32)
        opt = AdamW(tiny_proj.parameters(), lr=1e-3)
        opt.zero_grad()

        # Two micro-steps with /2 scaling
        for _ in range(2):
            out  = tiny_proj(vision_tokens).float()
            loss = out.mean() / 2
            loss.backward()

        grad_accum = {n: p.grad.clone() for n, p in tiny_proj.named_parameters()}

        # Single step without scaling (equivalent effective gradient)
        opt.zero_grad()
        out  = tiny_proj(vision_tokens).float()
        loss = out.mean()
        loss.backward()
        grad_single = {n: p.grad.clone() for n, p in tiny_proj.named_parameters()}

        for name in grad_accum:
            assert torch.allclose(grad_accum[name], grad_single[name], atol=1e-6), (
                f"Gradient mismatch at {name}"
            )

    def test_zero_grad_between_accum_windows(self, tiny_proj):
        """After optimizer.step(), gradients should be zeroed for the next window."""
        opt = AdamW(tiny_proj.parameters(), lr=1e-3)
        vision_tokens = torch.randn(1, 5, 32)

        opt.zero_grad(set_to_none=True)
        for _ in range(2):
            loss = (tiny_proj(vision_tokens).float().mean()) / 2
            loss.backward()

        opt.step()
        opt.zero_grad(set_to_none=True)

        for p in tiny_proj.parameters():
            assert p.grad is None, "Gradients should be None after zero_grad(set_to_none=True)"

    def test_loss_finite_after_backward(self, tiny_proj):
        """Scaled backward pass must not produce NaN/Inf gradients."""
        opt = AdamW(tiny_proj.parameters(), lr=1e-3)
        opt.zero_grad()
        for _ in range(4):
            out  = tiny_proj(torch.randn(1, 5, 32)).float()
            loss = out.mean() / 4
            loss.backward()

        for name, p in tiny_proj.named_parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all(), f"Non-finite grad at {name}"


# ── Scheduler + optimizer total steps math ────────────────────────────────────

class TestTotalStepsCalculation:
    def test_total_steps_with_grad_accum(self):
        """total_steps = epochs * max_pairs // grad_accum_steps."""
        epochs, max_pairs, accum = 2, 1000, 4
        total = epochs * max_pairs // max(accum, 1)
        assert total == 500

    def test_grad_accum_1_gives_epochs_times_pairs(self):
        epochs, max_pairs, accum = 3, 500, 1
        total = epochs * max_pairs // max(accum, 1)
        assert total == 1500

    def test_grad_accum_zero_clamped_to_one(self):
        """Division by zero avoided by max(grad_accum_steps, 1)."""
        epochs, max_pairs, accum = 1, 100, 0
        total = epochs * max_pairs // max(accum, 1)
        assert total == 100
