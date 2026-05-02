"""Optional execution context for strategies (panel + feature cache)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import pandas as pd

from .feature_store import FeatureStore
from .market_data_panel import MarketDataPanel

if TYPE_CHECKING:
    from .window_bounds import WindowBounds


@dataclass(frozen=True)
class StrategyDataContext:
    """Read-mostly view passed into ``generate_signals`` when enabled.

    Attributes:
        panel: Full history panel aligned to scenario/evaluation calendar.
        feature_store: Cached features for ``panel``.
        window_bounds: Optional walk-forward row bounds; ``None`` in backtests.
        current_row_ix: Row index in ``panel`` for the last bar of the expanding window.
        current_date: Rebalance/evaluation timestamp.
        universe_tickers: Tradable symbols for the scenario.
        benchmark_ticker: Benchmark symbol when configured.
    """

    panel: MarketDataPanel
    feature_store: FeatureStore
    window_bounds: Optional["WindowBounds"]
    current_row_ix: int
    current_date: pd.Timestamp
    universe_tickers: tuple[str, ...]
    benchmark_ticker: Optional[str]
