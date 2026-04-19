"""Shared pytest fixtures and markers for the MEDDIAG test suite.

Tests that need real models (vision encoder, LLM) are guarded by the
MEDDIAG_RUN_MODEL_TESTS env var so the fast unit-test suite stays quick.
Set MEDDIAG_RUN_MODEL_TESTS=1 before a full training run per SRS §19.3.
"""
from __future__ import annotations

import os
import pytest

from src.config import Config


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: requires downloaded models or network access"
    )


# Skip model-heavy tests unless explicitly opted in.
needs_model = pytest.mark.skipif(
    not os.getenv("MEDDIAG_RUN_MODEL_TESTS"),
    reason="Set MEDDIAG_RUN_MODEL_TESTS=1 to run model-dependent tests",
)


@pytest.fixture(scope="session")
def cfg() -> Config:
    """A test Config with FAISS dir pointed at a temp location."""
    return Config()


@pytest.fixture(scope="session")
def vision_encoder(cfg):
    """Loaded VisionEncoder — only instantiated when needed by slow tests."""
    from src.vision import VisionEncoder
    return VisionEncoder(cfg)
