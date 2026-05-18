"""Unit tests for `check_affinity_condition` (goal / option logic)."""

from __future__ import annotations

from fol_training.affinity_condition import check_affinity_condition


def _state_zeros(n: int = 80) -> list[float]:
    return [0.0] * n


def test_goal_zero_returns_no_match() -> None:
    s = _state_zeros()
    idx = 20
    s[idx] = 0.0
    opt, ok = check_affinity_condition(s, idx)
    assert opt == 0 and ok is False


def test_positive_goal_meets_option() -> None:
    s = _state_zeros()
    idx = 14
    s[idx] = 5.0
    s[idx - 14] = 0.0
    s[idx + 7] = 2.0
    s[idx + 14] = 0.0
    s[idx + 21] = 0.0
    opt, ok = check_affinity_condition(s, idx)
    assert ok is True
    assert opt == 1


def test_negative_goal_meets_negative_option() -> None:
    s = _state_zeros()
    idx = 18
    s[idx] = -4.0
    s[idx - 14] = 0.0
    s[idx + 7] = 0.0
    s[idx + 14] = -1.0
    s[idx + 21] = 0.0
    opt, ok = check_affinity_condition(s, idx)
    assert ok is True
    assert opt == 2
