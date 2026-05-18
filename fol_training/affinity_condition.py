"""Localized ABRL affinity gate: maps (state vector, goal index) → (option 1–3, fired?)."""

from __future__ import annotations

from typing import Any, Sequence, Tuple


def check_affinity_condition(state: Sequence[Any], affinity_idx: int) -> Tuple[int, bool]:
    """
    Evaluates the affinity condition based on the goal and options in the state array.

    Parameters:
        state: Flattened observation (indexable sequence).
        affinity_idx: Index into ``state`` for the goal channel (FoL uses 14–20).

    Returns:
        ``(option_index_1_to_3, condition_met)``; ``(0, False)`` when the goal slot is zero
        or the condition does not apply.
    """
    goal = state[affinity_idx]
    options = state[affinity_idx + 7 : affinity_idx + 22 : 7]

    if goal == 0:
        return 0, False

    goal_positive = goal > 0
    state_value = state[affinity_idx - 14]

    if (goal_positive and goal > state_value) or (not goal_positive and goal < state_value):
        for i, option in enumerate(options, start=1):
            if (goal_positive and option > 0) or (not goal_positive and option < 0):
                return i, True
        return 0, False

    return 0, False
