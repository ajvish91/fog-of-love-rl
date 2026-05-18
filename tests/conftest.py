"""Pytest fixtures: repo root on path, headless plots, optional fast training loops."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def minimal_training_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run training in a temp cwd with short rollouts (requires debug=1 in argv)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FOL_TEST_MINIMAL", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("TQDM_DISABLE", "1")
    (tmp_path / "plots").mkdir(parents=True, exist_ok=True)
    return tmp_path
