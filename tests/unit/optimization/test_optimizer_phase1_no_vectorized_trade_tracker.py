"""Phase 1: optimizer/trading paths must not reference vectorized tracking APIs."""

from __future__ import annotations

import pathlib
from typing import Final, Iterable


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"


_PRODUCTION_PATHS_REL: Final[tuple[str, ...]] = (
    "portfolio_backtester/optimization/performance/factory.py",
    "portfolio_backtester/optimization/performance/optuna_optimizer.py",
    "portfolio_backtester/optimization/performance/genetic_optimizer.py",
    "portfolio_backtester/optimization/population_evaluator.py",
    "portfolio_backtester/trading/numba_trade_tracker.py",
)

_FORBIDDEN_SUBSTRINGS: Final[tuple[str, ...]] = (
    "VectorizedTradeTracker",
    "vectorized_tracking",
    "enable_vectorized_tracking",
    "optimize_trade_tracking",
)


def _iter_production_sources() -> Iterable[tuple[pathlib.Path, str]]:
    for rel in _PRODUCTION_PATHS_REL:
        path = _SRC / rel
        yield path, path.read_text(encoding="utf-8")


def test_optimizer_production_modules_exclude_vectorized_tracking_surface() -> None:
    """Production optimizer/trading wrappers must not reference removed vectorized-tracking API."""
    for path, source in _iter_production_sources():
        lower = source.lower()
        problematic = tuple(s for s in _FORBIDDEN_SUBSTRINGS if s.lower() in lower)
        assert not problematic, f"{path} still references: {problematic}"
