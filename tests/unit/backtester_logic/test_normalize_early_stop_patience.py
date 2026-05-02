"""Tests for plateau early-stop patience normalization."""

from __future__ import annotations

import pytest

from portfolio_backtester.backtester_logic.optimization_orchestrator import (
    normalize_early_stop_patience,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (10, 10),
        (1, 1),
        (0, None),
        (-1, None),
    ],
)
def test_normalize_early_stop_patience_values(raw: int, expected: int | None) -> None:
    assert normalize_early_stop_patience(raw) == expected


def test_normalize_early_stop_patience_invalid_falls_back_to_default() -> None:
    assert normalize_early_stop_patience("not_an_int", default=7) == 7


def test_normalize_early_stop_patience_none_uses_default() -> None:
    assert normalize_early_stop_patience(None, default=12) == 12
