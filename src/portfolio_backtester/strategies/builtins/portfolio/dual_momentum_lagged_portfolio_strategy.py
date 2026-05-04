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
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Mapping, TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import logging
from portfolio_backtester.universe import get_top_weight_sp500_components

from .base_momentum_portfolio_strategy import BaseMomentumPortfolioStrategy

if TYPE_CHECKING:
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig

logger = logging.getLogger(__name__)

__all__ = ["DualMomentumLaggedPortfolioStrategy"]


class DualMomentumLaggedPortfolioStrategy(BaseMomentumPortfolioStrategy):
    """Dual Momentum strategy with lagged entries and exits.
    ...
        panic_exposure_multiplier : float
            Exposure multiplier applied when panic-state triggers (e.g., 0.25).
    """

    def __init__(
        self, strategy_config: Union[Mapping[str, Any], "CanonicalScenarioConfig"]
    ) -> None:
        super().__init__(strategy_config)

        params = self.strategy_config.get("strategy_params", {})
        if params is None:
            params = {}
            self.strategy_config["strategy_params"] = params

        # Set defaults
        params.setdefault("lookback_months", 12)
        params.setdefault("lag_months", 1)
        params.setdefault("momentum_skip_months", 0)
        params.setdefault("max_holdings", 10)
        params.setdefault("use_200sma_filter", True)
        params.setdefault("min_absolute_momentum", 0.0)
        params.setdefault("sma_period", 200)  # 200-day SMA for RORO filter

        # Signal upgrades / robustness overlays (research-backed; opt-in via config)
        params.setdefault("ranking_method", "excess_total_return")
        params.setdefault("absolute_exit_buffer", 0.0)
        params.setdefault("relative_exit_buffer", 0.0)

        params.setdefault("vol_target_enabled", False)
        params.setdefault("target_vol_annual", 0.12)
        params.setdefault("vol_lookback_days", 63)
        params.setdefault("vol_max_gross_exposure", 1.0)
        params.setdefault("vol_target_source", "benchmark")

        params.setdefault("panic_overlay_enabled", False)
        params.setdefault("panic_drawdown_lookback_days", 126)
        params.setdefault("panic_drawdown_threshold", -0.1)
        params.setdefault("panic_vol_threshold", 0.25)
        params.setdefault("panic_exposure_multiplier", 0.25)

        # Track pending signals (signals waiting for lag confirmation)
        # Key: ticker, Value: deque of (signal_date, signal_type) tuples
        self._pending_buy_signals: Dict[str, deque] = {}
        self._pending_sell_signals: Dict[str, deque] = {}

        # Track currently held positions
        self._current_holdings: Set[str] = set()

        # Track the last signal calculation results for lag verification
        self._previous_signal_dates: Dict[str, pd.Timestamp] = {}

    def _checkpoint_target_weights_scan(self) -> dict[str, Any]:
        ckpt = super()._checkpoint_target_weights_scan()
        ckpt["__dual_pending_buy"] = {k: list(v) for k, v in self._pending_buy_signals.items()}
        ckpt["__dual_pending_sell"] = {k: list(v) for k, v in self._pending_sell_signals.items()}
        ckpt["__dual_holdings"] = set(self._current_holdings)
        ckpt["__dual_prev_sig"] = dict(self._previous_signal_dates)
        return ckpt

    def _restore_target_weights_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super()._restore_target_weights_checkpoint(checkpoint)
        if "__dual_pending_buy" in checkpoint:
            self._pending_buy_signals = {
                k: deque(v) for k, v in checkpoint["__dual_pending_buy"].items()
            }
            self._pending_sell_signals = {
                k: deque(v) for k, v in checkpoint["__dual_pending_sell"].items()
            }
            self._current_holdings = set(checkpoint["__dual_holdings"])
            self._previous_signal_dates = dict(checkpoint["__dual_prev_sig"])

    def _reset_target_weights_scan_state(self) -> None:
        super()._reset_target_weights_scan_state()
        self._pending_buy_signals = {}
        self._pending_sell_signals = {}
        self._current_holdings = set()
        self._previous_signal_dates = {}

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
            "momentum_skip_months": {
                "type": "int",
                "min": 0,
                "max": 3,
                "default": 0,
                "description": "Months skipped at the end of the momentum formation window",
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
            "ranking_method": {
                "type": "categorical",
                "choices": ["excess_total_return", "residual_momentum"],
                "default": "excess_total_return",
                "description": "How to rank dual momentum candidates",
            },
            "absolute_exit_buffer": {
                "type": "float",
                "min": 0.0,
                "max": 0.2,
                "default": 0.0,
                "description": "Absolute momentum hysteresis buffer for held names",
            },
            "relative_exit_buffer": {
                "type": "float",
                "min": 0.0,
                "max": 0.2,
                "default": 0.0,
                "description": "Relative momentum hysteresis buffer for held names",
            },
            "vol_target_enabled": {
                "type": "categorical",
                "choices": [True, False],
                "default": False,
                "description": "Enable benchmark volatility targeting overlay",
            },
            "target_vol_annual": {
                "type": "float",
                "min": 0.05,
                "max": 0.25,
                "default": 0.12,
                "description": "Target annualized volatility for exposure scaling",
            },
            "vol_lookback_days": {
                "type": "int",
                "min": 21,
                "max": 252,
                "default": 63,
                "description": "Lookback window (days) for volatility estimation",
            },
            "vol_max_gross_exposure": {
                "type": "float",
                "min": 0.25,
                "max": 2.0,
                "default": 1.0,
                "description": "Max gross exposure after scaling",
            },
            "vol_target_source": {
                "type": "categorical",
                "choices": ["benchmark", "portfolio_proxy"],
                "default": "benchmark",
                "description": "Volatility series used for scaling",
            },
            "panic_overlay_enabled": {
                "type": "categorical",
                "choices": [True, False],
                "default": False,
                "description": "Enable panic-state crash overlay",
            },
            "panic_drawdown_lookback_days": {
                "type": "int",
                "min": 63,
                "max": 504,
                "default": 126,
                "description": "Lookback window (days) for drawdown calculation",
            },
            "panic_drawdown_threshold": {
                "type": "float",
                "min": -0.5,
                "max": -0.01,
                "default": -0.1,
                "description": "Drawdown threshold to trigger panic overlay",
            },
            "panic_vol_threshold": {
                "type": "float",
                "min": 0.1,
                "max": 1.0,
                "default": 0.25,
                "description": "Volatility threshold to trigger panic overlay",
            },
            "panic_exposure_multiplier": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.25,
                "description": "Exposure multiplier applied during panic regime",
            },
        }

    @staticmethod
    def _extract_close_series(
        historical_prices: pd.DataFrame,
        price_column: str = "Close",
    ) -> pd.Series:
        """Extract a 1D close price series from an OHLC dataframe (MultiIndex or flat)."""
        if isinstance(historical_prices.columns, pd.MultiIndex):
            try:
                close_df = historical_prices.xs(price_column, level="Field", axis=1)
                if isinstance(close_df, pd.DataFrame):
                    return close_df.iloc[:, 0]
                return close_df
            except KeyError:
                first_col = historical_prices.iloc[:, 0]
                return first_col if isinstance(first_col, pd.Series) else first_col.iloc[:, 0]

        if price_column in historical_prices.columns:
            close_series = historical_prices[price_column]
            if isinstance(close_series, pd.Series):
                return close_series
        # Fallback: first column
        first = historical_prices.iloc[:, 0]
        return first if isinstance(first, pd.Series) else first.iloc[:, 0]

    @staticmethod
    def _annualized_vol_from_returns(returns: pd.Series) -> Optional[float]:
        """Compute annualized volatility from daily returns, guarding edge cases."""
        cleaned = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(cleaned) < 2:
            return None
        vol_daily = float(cleaned.std(ddof=1))
        return float(vol_daily * np.sqrt(252.0))

    def _calculate_residual_momentum(
        self,
        asset_prices: pd.Series,
        benchmark_close: pd.Series,
        current_date: pd.Timestamp,
        lookback_months: int,
        min_obs: int = 60,
    ) -> Optional[float]:
        """Compute beta-adjusted momentum (abnormal component) using daily returns.

        Uses a market-model beta estimated over the lookback window and computes
        residual daily returns as: r_asset - beta * r_benchmark.

        This intentionally retains an intercept/alpha component (i.e., not removing alpha).
        """
        lookback_start = current_date - pd.DateOffset(months=lookback_months)

        asset_px = asset_prices.loc[asset_prices.index <= current_date]
        bench_px = benchmark_close.loc[benchmark_close.index <= current_date]
        if asset_px.empty or bench_px.empty:
            return None

        asset_rets = asset_px.pct_change(fill_method=None)
        bench_rets = bench_px.pct_change(fill_method=None)

        # Avoid lookahead: use returns up to the day before current_date (if available).
        asset_rets = asset_rets.loc[asset_rets.index < current_date]
        bench_rets = bench_rets.loc[bench_rets.index < current_date]

        asset_rets = asset_rets.loc[asset_rets.index >= lookback_start]
        bench_rets = bench_rets.loc[bench_rets.index >= lookback_start]

        aligned = pd.concat([asset_rets, bench_rets], axis=1, join="inner").dropna()
        if len(aligned) < min_obs:
            return None

        y = aligned.iloc[:, 0].astype(float).to_numpy()
        x = aligned.iloc[:, 1].astype(float).to_numpy()

        x_var = float(np.var(x, ddof=1))
        if not np.isfinite(x_var) or x_var <= 0:
            return None

        beta = float(np.cov(x, y, ddof=1)[0, 1] / x_var)
        residual = y - (beta * x)

        # Robustly compound residual returns.
        residual = np.clip(residual, -0.99, 10.0)
        compounded = float(np.prod(1.0 + residual) - 1.0)
        return compounded if np.isfinite(compounded) else None

    def _exposure_scale_from_benchmark(
        self,
        asset_prices_hist: pd.DataFrame,
        benchmark_prices_hist: pd.DataFrame,
        current_date: pd.Timestamp,
        params: Dict[str, Any],
        holdings: Set[str],
    ) -> float:
        """Compute gross exposure scaling factor using vol targeting + optional panic overlay."""
        if not bool(params.get("vol_target_enabled", False)):
            return 1.0

        target_vol_annual = float(params.get("target_vol_annual", 0.12))
        vol_lookback_days = int(params.get("vol_lookback_days", 63))
        max_gross = float(params.get("vol_max_gross_exposure", 1.0))
        vol_source = str(params.get("vol_target_source", "benchmark")).lower()

        bench_close = self._extract_close_series(
            benchmark_prices_hist, price_column=str(params.get("price_column_benchmark", "Close"))
        )
        bench_close = bench_close.loc[bench_close.index <= current_date]

        vol_annual_bench: Optional[float] = None
        if len(bench_close) >= (vol_lookback_days + 2):
            bench_rets = bench_close.pct_change(fill_method=None).dropna()
            bench_rets = bench_rets.loc[bench_rets.index < current_date].tail(vol_lookback_days)
            vol_annual_bench = self._annualized_vol_from_returns(bench_rets)

        vol_annual_target: Optional[float]
        if vol_source == "portfolio_proxy":
            held = [t for t in sorted(holdings) if t in asset_prices_hist.columns]
            if len(held) < 1:
                return 1.0
            px = asset_prices_hist.loc[asset_prices_hist.index <= current_date, held]
            if len(px) < (vol_lookback_days + 2):
                return 1.0
            rets = px.pct_change(fill_method=None).dropna(how="all")
            rets = rets.loc[rets.index < current_date].tail(vol_lookback_days)
            portfolio_rets = rets.mean(axis=1, skipna=True)
            vol_annual_target = self._annualized_vol_from_returns(portfolio_rets)
        else:
            vol_annual_target = vol_annual_bench

        if vol_annual_target is None or vol_annual_target <= 0:
            return 1.0

        scale = float(np.clip(target_vol_annual / max(vol_annual_target, 1e-12), 0.0, max_gross))

        if bool(params.get("panic_overlay_enabled", False)):
            dd_lookback = int(params.get("panic_drawdown_lookback_days", 126))
            dd_threshold = float(params.get("panic_drawdown_threshold", -0.1))
            vol_threshold = float(params.get("panic_vol_threshold", 0.25))
            panic_mult = float(params.get("panic_exposure_multiplier", 0.25))

            if vol_annual_bench is None:
                return scale

            # Use close up to (but not including) current_date to avoid lookahead.
            bench_close_for_regime = bench_close.loc[bench_close.index < current_date]
            if len(bench_close_for_regime) >= 2:
                dd_window = bench_close_for_regime.tail(dd_lookback)
                if not dd_window.empty:
                    peak = float(dd_window.max())
                    last = float(dd_window.iloc[-1])
                    if peak > 0 and np.isfinite(peak) and np.isfinite(last):
                        drawdown = (last / peak) - 1.0
                        if (drawdown <= dd_threshold) and (vol_annual_bench >= vol_threshold):
                            scale *= float(np.clip(panic_mult, 0.0, 1.0))

        return float(np.clip(scale, 0.0, max_gross))

    def _calculate_momentum_score(
        self,
        prices: pd.Series,
        current_date: pd.Timestamp,
        lookback_months: int,
        skip_months: int = 0,
    ) -> Optional[float]:
        """Calculate momentum score for a single asset.

        Returns price return over lookback period, or None if insufficient data.
        """
        formation_end_date = current_date - pd.DateOffset(months=max(skip_months, 0))
        lookback_date = formation_end_date - pd.DateOffset(months=lookback_months)

        # Get price at formation end date (or most recent before)
        prices_up_to_current = prices[prices.index <= formation_end_date]
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
        skip_months: int = 0,
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

        return self._calculate_momentum_score(
            bench_close, current_date, lookback_months, skip_months=skip_months
        )

    def _extract_benchmark_close_series(
        self, benchmark_prices: pd.DataFrame, price_column: str = "Close"
    ) -> Optional[pd.Series]:
        """Return benchmark close series, or None when benchmark data is unavailable."""
        if benchmark_prices.empty or len(benchmark_prices.columns) == 0:
            return None

        if isinstance(benchmark_prices.columns, pd.MultiIndex):
            try:
                bench_close = benchmark_prices.xs(price_column, level="Field", axis=1)
                if isinstance(bench_close, pd.DataFrame):
                    if bench_close.empty or len(bench_close.columns) == 0:
                        return None
                    return bench_close.iloc[:, 0]
                return bench_close
            except KeyError:
                return benchmark_prices.iloc[:, 0]

        if price_column in benchmark_prices.columns:
            return benchmark_prices[price_column]

        return benchmark_prices.iloc[:, 0]

    def _is_benchmark_above_200sma(
        self,
        benchmark_prices: pd.DataFrame,
        current_date: pd.Timestamp,
        sma_period: int = 200,
        price_column: str = "Close",
    ) -> bool:
        """Check if benchmark close is above its 200-day SMA."""
        bench_close = self._extract_benchmark_close_series(
            benchmark_prices, price_column=price_column
        )
        if bench_close is None:
            logger.warning(
                "Benchmark data is missing; treating 200SMA filter as risk-off for %s.",
                current_date,
            )
            return False

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
        current_holdings: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Get candidates that pass both absolute and relative momentum tests.

        Returns list of (ticker, score) tuples, sorted by score descending.
        """
        lookback_months = int(params.get("lookback_months", 12))
        skip_months = int(params.get("momentum_skip_months", 0))
        min_abs_momentum = float(params.get("min_absolute_momentum", 0.0))
        ranking_method = str(params.get("ranking_method", "excess_total_return")).lower()
        abs_exit_buffer = float(params.get("absolute_exit_buffer", 0.0))
        rel_exit_buffer = float(params.get("relative_exit_buffer", 0.0))
        current_holdings = current_holdings or set()

        bench_close = self._extract_close_series(
            benchmark_prices, price_column=str(params.get("price_column_benchmark", "Close"))
        )

        # Calculate benchmark momentum
        bench_momentum = self._calculate_benchmark_momentum(
            benchmark_prices, current_date, lookback_months, skip_months=skip_months
        )
        if bench_momentum is None:
            return []

        candidates: List[Tuple[str, float]] = []

        for ticker in asset_prices.columns:
            asset_momentum = self._calculate_momentum_score(
                asset_prices[ticker], current_date, lookback_months, skip_months=skip_months
            )
            if asset_momentum is None:
                continue

            abs_threshold = min_abs_momentum
            rel_threshold = bench_momentum
            if ticker in current_holdings:
                abs_threshold = min_abs_momentum - abs_exit_buffer
                rel_threshold = bench_momentum - rel_exit_buffer

            # Dual momentum conditions:
            # 1. Absolute: momentum > threshold
            # 2. Relative: momentum > benchmark momentum threshold
            if asset_momentum > abs_threshold and asset_momentum > rel_threshold:
                score = asset_momentum - bench_momentum
                if ranking_method == "residual_momentum":
                    residual_score = self._calculate_residual_momentum(
                        asset_prices[ticker],
                        bench_close,
                        current_date,
                        lookback_months=lookback_months,
                    )
                    if residual_score is not None:
                        score = residual_score
                candidates.append((ticker, float(score)))

        # Sort by score (descending)
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

        current_date = pd.Timestamp(cast(pd.Timestamp, current_date))

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

        # --- Dynamic Universe Filtering (Top-N S&P 500 from universe config) ---
        if self.canonical_config is not None and self.canonical_config.universe_definition:
            universe_cfg = self.canonical_config.universe_definition
        else:
            universe_cfg = self.strategy_config.get("universe_config", {})

        universe_type = str(universe_cfg.get("type", "")).lower()
        method_name = str(universe_cfg.get("method_name", "")).strip()
        should_apply_dynamic_universe = (
            universe_type == "method" and method_name == "get_top_weight_sp500_components"
        )
        configured_n = universe_cfg.get("n_holdings", params.get("num_holdings", max_holdings))
        if should_apply_dynamic_universe:
            try:
                dynamic_universe_n = int(configured_n)
            except (TypeError, ValueError):
                dynamic_universe_n = 50
            if dynamic_universe_n <= 0:
                dynamic_universe_n = 50

            dynamic_universe_exact = bool(universe_cfg.get("exact", False))

            # Get top components for the current date to handle survivorship bias.
            try:
                top_components = get_top_weight_sp500_components(
                    current_date,
                    n=dynamic_universe_n,
                    exact=dynamic_universe_exact,
                )

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
                # Failing safe is better than trading the wrong dynamic universe.
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

        # Risk-off signal generator (new system: True = risk-off)
        risk_off_generator = self.get_risk_off_signal_generator()
        non_universe_data_safe = (
            non_universe_historical_data
            if non_universe_historical_data is not None
            else pd.DataFrame()
        )
        if risk_off_generator.generate_risk_off_signal(
            all_historical_data,
            benchmark_historical_data,
            non_universe_data_safe,
            current_date,
        ):
            self._current_holdings.clear()
            self._pending_buy_signals.clear()
            self._pending_sell_signals.clear()
            return pd.DataFrame(0.0, index=[current_date], columns=original_assets)

        # Get dual momentum candidates
        candidates_with_scores = self._get_dual_momentum_candidates(
            asset_prices_hist,
            benchmark_prices_hist,
            current_date,
            params,
            current_holdings=self._current_holdings,
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

        # Volatility targeting + panic overlay (scale gross exposure; leftover goes to cash)
        if self._current_holdings:
            exposure_scale = self._exposure_scale_from_benchmark(
                asset_prices_hist,
                benchmark_prices_hist,
                current_date,
                params,
                self._current_holdings,
            )
            if exposure_scale != 1.0:
                output_df = output_df.mul(exposure_scale)

        # Log holdings periodically
        if len(self._current_holdings) > 0:
            logger.debug(
                f"[{current_date.date()}] Holdings ({len(self._current_holdings)}): "
                f"{sorted(self._current_holdings)[:5]}..."
            )

        return output_df
