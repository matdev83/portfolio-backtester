"""Dual Momentum Strategy with Lagged Entries/Exits.

This module implements a classical dual momentum strategy with the following features:
- **Dual Momentum**: Buy stocks which exhibit upside momentum (absolute) AND outperform
  the benchmark SPX (relative momentum)
- **Lagged Entries/Exits**: Once a buy signal is generated, the strategy waits for a
  defined lag period (default 1 month) before entering. Entry only occurs if the signal
  is still active after the lag period.
- **Universe**: Designed for TOP N S&P 500 companies (via MDMP dataset)
- **Configurable Parameters**:
  - `lag_months`: How many months to wait before confirming entry (default: 1)
  - `max_holdings`: Maximum number of stocks to hold (default: 10)
  - `use_200sma_filter`: Whether to only initiate new buys when SPX > 200-day SMA
  - `position_sizer`: Position sizing strategy (subject to optimization)
  - `lookback_months`: Momentum calculation lookback period
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import logging
from portfolio_backtester.universe import get_top_weight_sp500_components

from .base_momentum_portfolio_strategy import BaseMomentumPortfolioStrategy

logger = logging.getLogger(__name__)

__all__ = ["DualMomentumLaggedPortfolioStrategy"]


class DualMomentumLaggedPortfolioStrategy(BaseMomentumPortfolioStrategy):
    """Dual Momentum strategy with lagged entries and exits.

    This strategy implements the classical dual momentum approach:
    1. Absolute Momentum: Stock must have positive momentum (price > price N months ago)
    2. Relative Momentum: Stock must outperform the benchmark (SPX)

    Additionally, entries and exits are lagged - a signal must persist for `lag_months`
    before action is taken. This reduces whipsaws and trading costs.

    Parameters (via strategy_params)
    ---------------------------------
    lookback_months : int
        Lookback period for momentum calculation (default: 12 months)
    lag_months : int
        Number of months to wait before confirming entry/exit (default: 1)
    max_holdings : int
        Maximum number of stocks to hold at any time (default: 10)
    use_200sma_filter : bool
        If True, only enter new positions when SPX close > 200-day SMA (default: True)
    min_absolute_momentum : float
        Minimum absolute momentum threshold (default: 0.0, i.e., any positive)
    """

    def __init__(self, strategy_config: Dict[str, Any]) -> None:
        super().__init__(strategy_config)

        params = self.strategy_config.get("strategy_params", {})
        if params is None:
            params = {}
            self.strategy_config["strategy_params"] = params

        # Set defaults
        params.setdefault("lookback_months", 12)
        params.setdefault("lag_months", 1)
        params.setdefault("max_holdings", 10)
        params.setdefault("use_200sma_filter", True)
        params.setdefault("min_absolute_momentum", 0.0)
        params.setdefault("sma_period", 200)  # 200-day SMA for RORO filter

        # Track pending signals (signals waiting for lag confirmation)
        # Key: ticker, Value: deque of (signal_date, signal_type) tuples
        self._pending_buy_signals: Dict[str, deque] = {}
        self._pending_sell_signals: Dict[str, deque] = {}

        # Track currently held positions
        self._current_holdings: Set[str] = set()

        # Track the last signal calculation results for lag verification
        self._previous_signal_dates: Dict[str, pd.Timestamp] = {}

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """Return optimizable parameters for this strategy."""
        return {
            "lookback_months": {
                "type": "int",
                "min": 3,
                "max": 24,
                "default": 12,
                "description": "Momentum lookback period in months",
            },
            "lag_months": {
                "type": "int",
                "min": 0,
                "max": 3,
                "default": 1,
                "description": "Lag period before entry/exit confirmation",
            },
            "max_holdings": {
                "type": "int",
                "min": 5,
                "max": 30,
                "default": 10,
                "description": "Maximum number of stocks to hold",
            },
            "use_200sma_filter": {
                "type": "categorical",
                "choices": [True, False],
                "default": True,
                "description": "Use 200-day SMA RORO filter",
            },
            "position_sizer": {
                "type": "categorical",
                "choices": [
                    "equal_weight",
                    "rolling_sharpe",
                    "rolling_sortino",
                    "rolling_downside_volatility",
                ],
                "default": "equal_weight",
                "description": "Position sizing strategy",
            },
            "min_absolute_momentum": {
                "type": "float",
                "min": -0.1,
                "max": 0.2,
                "default": 0.0,
                "description": "Minimum absolute momentum threshold",
            },
            "sma_period": {
                "type": "int",
                "min": 50,
                "max": 300,
                "default": 200,
                "description": "SMA period for RORO filter",
            },
            "trade_longs": {
                "type": "categorical",
                "choices": [True, False],
                "default": True,
                "description": "Allow long positions",
            },
            "trade_shorts": {
                "type": "categorical",
                "choices": [True, False],
                "default": False,
                "description": "Allow short positions",
            },
        }

    def _calculate_momentum_score(
        self,
        prices: pd.Series,
        current_date: pd.Timestamp,
        lookback_months: int,
    ) -> Optional[float]:
        """Calculate momentum score for a single asset.

        Returns price return over lookback period, or None if insufficient data.
        """
        lookback_date = current_date - pd.DateOffset(months=lookback_months)

        # Get price at current date (or most recent before)
        prices_up_to_current = prices[prices.index <= current_date]
        if prices_up_to_current.empty:
            return None

        current_price = prices_up_to_current.iloc[-1]

        # Get price at lookback date (or most recent before)
        prices_at_lookback = prices[prices.index <= lookback_date]
        if prices_at_lookback.empty:
            return None

        lookback_price = prices_at_lookback.iloc[-1]

        if pd.isna(lookback_price) or lookback_price == 0:
            return None

        return float((current_price / lookback_price) - 1.0)

    def _calculate_benchmark_momentum(
        self,
        benchmark_prices: pd.DataFrame,
        current_date: pd.Timestamp,
        lookback_months: int,
        price_column: str = "Close",
    ) -> Optional[float]:
        """Calculate benchmark momentum for the same period."""
        # Extract close prices from benchmark
        if isinstance(benchmark_prices.columns, pd.MultiIndex):
            try:
                bench_close = benchmark_prices.xs(price_column, level="Field", axis=1)
                if isinstance(bench_close, pd.DataFrame):
                    bench_close = bench_close.iloc[:, 0]
            except KeyError:
                # Try to get any column that might be close prices
                bench_close = benchmark_prices.iloc[:, 0]
        else:
            if price_column in benchmark_prices.columns:
                bench_close = benchmark_prices[price_column]
            else:
                bench_close = benchmark_prices.iloc[:, 0]

        return self._calculate_momentum_score(bench_close, current_date, lookback_months)

    def _is_benchmark_above_200sma(
        self,
        benchmark_prices: pd.DataFrame,
        current_date: pd.Timestamp,
        sma_period: int = 200,
        price_column: str = "Close",
    ) -> bool:
        """Check if benchmark close is above its 200-day SMA."""
        # Extract close prices from benchmark
        if isinstance(benchmark_prices.columns, pd.MultiIndex):
            try:
                bench_close = benchmark_prices.xs(price_column, level="Field", axis=1)
                if isinstance(bench_close, pd.DataFrame):
                    bench_close = bench_close.iloc[:, 0]
            except KeyError:
                bench_close = benchmark_prices.iloc[:, 0]
        else:
            if price_column in benchmark_prices.columns:
                bench_close = benchmark_prices[price_column]
            else:
                bench_close = benchmark_prices.iloc[:, 0]

        bench_close_up_to_date = bench_close[bench_close.index <= current_date]
        if len(bench_close_up_to_date) < sma_period:
            # Not enough data for SMA - allow trading
            return True

        current_price = float(bench_close_up_to_date.iloc[-1])
        sma_value = float(bench_close_up_to_date.iloc[-sma_period:].mean())

        return bool(current_price > sma_value)

    def _get_dual_momentum_candidates(
        self,
        asset_prices: pd.DataFrame,
        benchmark_prices: pd.DataFrame,
        current_date: pd.Timestamp,
        params: Dict[str, Any],
    ) -> List[tuple[str, float]]:
        """Get candidates that pass both absolute and relative momentum tests.

        Returns list of (ticker, excess_momentum) tuples, sorted by excess momentum descending.
        """
        lookback_months = int(params.get("lookback_months", 12))
        min_abs_momentum = float(params.get("min_absolute_momentum", 0.0))

        # Calculate benchmark momentum
        bench_momentum = self._calculate_benchmark_momentum(
            benchmark_prices, current_date, lookback_months
        )
        if bench_momentum is None:
            return []

        candidates: List[tuple[str, float]] = []

        for ticker in asset_prices.columns:
            asset_momentum = self._calculate_momentum_score(
                asset_prices[ticker], current_date, lookback_months
            )
            if asset_momentum is None:
                continue

            # Dual momentum conditions:
            # 1. Absolute: momentum > min_absolute_momentum (default 0, meaning positive)
            # 2. Relative: momentum > benchmark momentum
            if asset_momentum > min_abs_momentum and asset_momentum > bench_momentum:
                excess_momentum = asset_momentum - bench_momentum
                candidates.append((ticker, excess_momentum))

        # Sort by excess momentum (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _update_pending_signals(
        self,
        current_candidates: Set[str],
        current_date: pd.Timestamp,
        lag_months: int,
    ) -> tuple[Set[str], Set[str]]:
        """Update pending signals and return confirmed entries/exits.

        Returns:
            (confirmed_entries, confirmed_exits): Sets of tickers to enter/exit
        """
        lag_threshold = current_date - pd.DateOffset(months=lag_months)

        confirmed_entries: Set[str] = set()
        confirmed_exits: Set[str] = set()

        # Process potential entries: stocks that are candidates now
        for ticker in current_candidates:
            if ticker not in self._current_holdings:
                # New potential entry
                if ticker not in self._pending_buy_signals:
                    self._pending_buy_signals[ticker] = deque(maxlen=12)
                self._pending_buy_signals[ticker].append(current_date)

                # Check if signal has persisted long enough
                if self._pending_buy_signals[ticker]:
                    first_signal = min(self._pending_buy_signals[ticker])
                    if first_signal <= lag_threshold:
                        # Signal has persisted for lag_months - confirm entry
                        confirmed_entries.add(ticker)

        # Process potential exits: current holdings that are no longer candidates
        for ticker in self._current_holdings:
            if ticker not in current_candidates:
                # Potential exit
                if ticker not in self._pending_sell_signals:
                    self._pending_sell_signals[ticker] = deque(maxlen=12)
                self._pending_sell_signals[ticker].append(current_date)

                # Check if the "not a candidate" status has persisted
                if self._pending_sell_signals[ticker]:
                    first_signal = min(self._pending_sell_signals[ticker])
                    if first_signal <= lag_threshold:
                        confirmed_exits.add(ticker)
            else:
                # Still a candidate - clear any pending exit signals
                if ticker in self._pending_sell_signals:
                    del self._pending_sell_signals[ticker]

        # Clean up old pending buy signals for stocks no longer candidates
        tickers_to_remove_buy = []
        for ticker in self._pending_buy_signals:
            if ticker not in current_candidates and ticker not in self._current_holdings:
                tickers_to_remove_buy.append(ticker)
        for ticker in tickers_to_remove_buy:
            del self._pending_buy_signals[ticker]

        return confirmed_entries, confirmed_exits

    def _calculate_scores(
        self,
        asset_prices: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        """Calculate scores for ranking candidates.

        This method is required by BaseMomentumPortfolioStrategy but we override
        generate_signals for more control over the dual momentum logic.
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        lookback_months = int(params.get("lookback_months", 12))

        scores = pd.Series(dtype=float, index=asset_prices.columns)

        for ticker in asset_prices.columns:
            momentum = self._calculate_momentum_score(
                asset_prices[ticker], current_date, lookback_months
            )
            scores[ticker] = momentum if momentum is not None else np.nan

        return scores.fillna(0.0)

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Generate trading signals using dual momentum with lag.

        This overrides the base class to implement the lagged dual momentum logic.
        """
        if current_date is None:
            current_date = all_historical_data.index[-1]

        current_date = pd.Timestamp(current_date)

        # Get parameters
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        lag_months = int(params.get("lag_months", 1))
        max_holdings = int(params.get("max_holdings", 10))
        use_200sma_filter = bool(params.get("use_200sma_filter", True))
        sma_period = int(params.get("sma_period", 200))
        price_column = params.get("price_column_asset", "Close")

        # Validate data sufficiency
        is_sufficient, _ = self.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )

        # Get original assets for output shape
        original_assets = (
            all_historical_data.columns.get_level_values("Ticker").unique()
            if isinstance(all_historical_data.columns, pd.MultiIndex)
            else all_historical_data.columns
        )

        if not is_sufficient:
            return pd.DataFrame(0.0, index=[current_date], columns=original_assets)

        # Filter universe by data availability
        valid_assets = self.filter_universe_by_data_availability(all_historical_data, current_date)
        if not valid_assets:
            return pd.DataFrame(0.0, index=[current_date], columns=original_assets)

        # --- Dynamic Universe Filtering (Top 50 S&P 500) ---
        # Get the top 50 components for the current date to handle survivorship bias
        try:
            top_components = get_top_weight_sp500_components(current_date, n=50, exact=False)

            # IMPORTANT: `all_historical_data` tickers are the *local* tickers we requested
            # from the data source (e.g., "AAPL", "SPY"), not MDMP canonical IDs.
            # Converting Top-50 tickers to canonical IDs here makes the intersection empty.
            top_set = {str(t).strip().upper() for t in top_components}

            valid_assets = [str(a) for a in valid_assets if str(a).strip().upper() in top_set]

            if not valid_assets:
                logger.warning(f"No valid assets found in Top 50 for {current_date}")
                return pd.DataFrame(0.0, index=[current_date], columns=original_assets)

        except Exception as e:
            logger.error(f"Dynamic universe filtering failed: {e}")
            # Fallback: continue with original valid_assets?
            # Or fail safe? Failing safe (empty) is better than trading wrong universe.
            return pd.DataFrame(0.0, index=[current_date], columns=original_assets)
        # ---------------------------------------------------

        # Extract close prices
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            asset_prices = all_historical_data.xs(price_column, level="Field", axis=1)
            asset_prices = asset_prices.loc[:, asset_prices.columns.isin(valid_assets)]
        else:
            asset_prices = all_historical_data.loc[
                :, all_historical_data.columns.isin(valid_assets)
            ]

        asset_prices_hist_raw = asset_prices[asset_prices.index <= current_date]
        # Ensure we have a DataFrame (not Series) for asset prices
        if isinstance(asset_prices_hist_raw, pd.Series):
            asset_prices_hist = asset_prices_hist_raw.to_frame()
        else:
            asset_prices_hist = asset_prices_hist_raw

        benchmark_prices_hist = benchmark_historical_data[
            benchmark_historical_data.index <= current_date
        ]

        # Check date range
        if (start_date and current_date < start_date) or (end_date and current_date > end_date):
            return pd.DataFrame(0.0, index=[current_date], columns=original_assets)

        # Check 200-day SMA filter (RORO signal)
        if use_200sma_filter:
            is_risk_on = self._is_benchmark_above_200sma(
                benchmark_prices_hist, current_date, sma_period
            )
            if not is_risk_on:
                # Risk-off: exit all positions (immediate, no lag for exits in risk-off)
                self._current_holdings.clear()
                self._pending_buy_signals.clear()
                return pd.DataFrame(0.0, index=[current_date], columns=original_assets)

        # Get dual momentum candidates
        candidates_with_scores = self._get_dual_momentum_candidates(
            asset_prices_hist, benchmark_prices_hist, current_date, params
        )

        # Current candidate tickers (top performers that pass dual momentum)
        current_candidates = set(t for t, _ in candidates_with_scores)

        # Update pending signals and get confirmed entries/exits
        confirmed_entries, confirmed_exits = self._update_pending_signals(
            current_candidates, current_date, lag_months
        )

        # Apply exits
        for ticker in confirmed_exits:
            self._current_holdings.discard(ticker)
            if ticker in self._pending_sell_signals:
                del self._pending_sell_signals[ticker]

        # Apply entries (respecting max_holdings)
        available_slots = max_holdings - len(self._current_holdings)
        if available_slots > 0 and confirmed_entries:
            # Prioritize by excess momentum (already sorted in candidates_with_scores)
            candidates_to_enter = [t for t, _ in candidates_with_scores if t in confirmed_entries][
                :available_slots
            ]
            for ticker in candidates_to_enter:
                self._current_holdings.add(ticker)
                if ticker in self._pending_buy_signals:
                    del self._pending_buy_signals[ticker]

        # Build output weights
        output_df = pd.DataFrame(0.0, index=[current_date], columns=original_assets)

        if self._current_holdings:
            # Equal weight among holdings
            weight_per_holding = 1.0 / len(self._current_holdings)
            for ticker in self._current_holdings:
                if ticker in output_df.columns:
                    output_df.loc[current_date, ticker] = weight_per_holding

        # Log holdings periodically
        if len(self._current_holdings) > 0:
            logger.debug(
                f"[{current_date.date()}] Holdings ({len(self._current_holdings)}): "
                f"{sorted(self._current_holdings)[:5]}..."
            )

        return output_df
