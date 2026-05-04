"""Small helpers for :class:`OptimizationOrchestrator` (kept import-light)."""

from __future__ import annotations

from typing import Any, Optional


def normalize_early_stop_patience(raw: Any, *, default: int = 10) -> Optional[int]:
    """Map CLI/config patience to the value used by :class:`ProgressTracker`.

    ``None`` disables early stopping on plateau (no cap on consecutive
    non-improving evaluations). Non-positive integers are treated as disabled
    so small discrete grids can be explored up to ``--optuna-trials``.

    Args:
        raw: Value from ``optimizer_args`` (typically an :class:`int`).
        default: Patience used when ``raw`` is missing or not coercible to int.

    Returns:
        Positive patience, or ``None`` when early stopping should be off.
    """
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = default
    if n <= 0:
        return None
    return n
