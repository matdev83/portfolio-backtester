from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, cast, Union, Mapping, TYPE_CHECKING

import numpy as np
import pandas as pd

from portfolio_backtester.backtester_logic.strategy_logic import _expanding_iloc_ends
from portfolio_backtester.strategies._core.base import PortfolioStrategy
from portfolio_backtester.strategies._core.target_generation import StrategyContext

if TYPE_CHECKING:
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig

logger = logging.getLogger(__name__)


class DriftRegimeConditionalFactorPortfolioStrategy(PortfolioStrategy):
    """
    Drift-Regime Conditional Factor Strategy.

    Based on the paper: "Discovery of a 13-Sharpe OOS Factor: Drift Regimes Unlock
    Hidden Cross-Sectional Predictability" by Mainak Singha.

    Key Logic:
    1. Identify Drift Regime: > 60% positive days in trailing 63-day window.
    2. Combine Signals: Short-Term Reversal (STR) and Value (Proxy).
    3. Conditional Activation: Signals are only active for stocks in the drift regime.
    """

    def __init__(
        self, strategy_config: Union[Mapping[str, Any], "CanonicalScenarioConfig"]
    ) -> None:
        super().__init__(strategy_config)

        params = self._get_params_dict()

        # Strategy specific defaults
        defaults = {
            "drift_window": 63,
            "drift_threshold": 0.60,
            "reversal_window": 10,
            "num_holdings": 10,
            "leverage": 1.0,
            "smoothing_lambda": 0.0,  # Default to no smoothing for this strategy
            "price_column_asset": "Close",
            "value_weight": 0.70,
            "selection_mode": "top_n",  # "top_n" or "all"
            "weighting_mode": "proportional",  # "proportional" or "equal"
            "price_min": None,
            "trade_longs": True,
            "trade_shorts": True,
            "min_history_days": 252 * 5,
            "risk_off_signal_config": None,
        }

        for key, value in defaults.items():
            params.setdefault(key, value)

    def _get_params_dict(self) -> Dict[str, Any]:
        """Extract strategy_params from config."""
        params_any = self.strategy_params.get("strategy_params", self.strategy_params)
        if params_any is None:
            self.strategy_params["strategy_params"] = {}
            params_any = self.strategy_params["strategy_params"]
        return cast(Dict[str, Any], params_any)

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter metadata for optimization."""
        return {
            "trade_longs": {
                "type": "bool",
                "default": True,
                "description": "Whether strategy is allowed to open long positions",
            },
            "trade_shorts": {
                "type": "bool",
                "default": True,
                "description": "Whether strategy is allowed to open short positions",
            },
            "drift_window": {"type": "int", "min": 21, "max": 126, "default": 63},
            "drift_threshold": {"type": "float", "min": 0.5, "max": 0.8, "default": 0.6},
            "reversal_window": {"type": "int", "min": 5, "max": 30, "default": 10},
            "num_holdings": {"type": "int", "min": 5, "max": 50, "default": 10},
            "leverage": {"type": "float", "min": 0.5, "max": 2.0, "default": 1.0},
            "smoothing_lambda": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            "value_weight": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.7},
            "selection_mode": {
                "type": "categorical",
                "values": ["top_n", "all"],
                "default": "top_n",
            },
            "weighting_mode": {
                "type": "categorical",
                "values": ["proportional", "equal"],
                "default": "proportional",
            },
            "price_min": {"type": "float", "min": 0.0, "max": 10.0, "default": None},
            "min_history_days": {"type": "int", "min": 0, "max": 252 * 10, "default": 252 * 5},
            "universe_config": {"type": "dict", "default": None},
            "risk_off_signal_config": {"type": "dict", "default": None},
        }

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generate trading signals based on drift-regime conditional factor.
        """
        if current_date is None:
            current_date = all_historical_data.index[-1]

        current_date = pd.Timestamp(current_date)
        params = self._get_params_dict()

        # 1. Extract price data
        price_col = params.get("price_column_asset", "Close")
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            prices_full = all_historical_data.xs(price_col, level="Field", axis=1)
        else:
            prices_full = all_historical_data

        if isinstance(prices_full, pd.Series):
            prices_full = prices_full.to_frame()

        prices_full = prices_full[prices_full.index <= current_date]
        if prices_full.empty:
            return self._empty_weights(all_historical_data, current_date)

        # Dynamic universe: require minimum history per symbol (default 5 years)
        min_history_days = int(params.get("min_history_days", 252 * 5))
        history_counts = prices_full.notna().sum(axis=0)
        eligible_cols = history_counts[history_counts >= min_history_days].index
        eligible_cols_list = list(eligible_cols)
        if len(eligible_cols_list) == 0:
            return self._empty_weights(all_historical_data, current_date)

        prices_full = prices_full.loc[:, eligible_cols_list]

        # Filter to data up to current_date and only keep what's needed for windows
        lookback_needed = max(params["drift_window"], params["reversal_window"]) + 5
        prices = prices_full.tail(lookback_needed)

        if len(prices) < 2:
            return self._empty_weights(all_historical_data, current_date)

        # 2. Calculate Drift Regime
        daily_returns = prices.pct_change(fill_method=None)
        is_positive = daily_returns > 0
        is_valid = daily_returns.notna()

        drift_window = params["drift_window"]
        positive_days = is_positive.tail(drift_window).sum()
        valid_days = is_valid.tail(drift_window).sum()

        drift_fraction = pd.Series(positive_days / valid_days).fillna(0)
        enough_history = valid_days >= drift_window
        current_drift_regime = (drift_fraction > params["drift_threshold"]) & enough_history

        # 3. Calculate Raw Factor Signals (Global)
        reversal_window = params.get("reversal_window", 10)
        if len(prices) > reversal_window:
            str_return = (prices.iloc[-1] / prices.iloc[-reversal_window - 1]) - 1
        else:
            str_return = prices.pct_change(periods=reversal_window, fill_method=None).loc[
                current_date
            ]

        str_signal_global = -str_return

        current_prices = prices.loc[current_date]
        if isinstance(current_prices, pd.DataFrame):
            current_prices = current_prices.iloc[-1]
        price_min = params.get("price_min")
        current_prices = current_prices.where(current_prices > 0)
        if price_min is not None:
            current_prices = current_prices.where(current_prices >= price_min)
        value_signal_global = 1.0 / current_prices

        # Value is percentile rank (0..1) cross-sectionally
        value_rank_global = value_signal_global.rank(pct=True)

        # Reversal is standardized (z-score) cross-sectionally
        str_mean = str_signal_global.mean(skipna=True)
        str_std = str_signal_global.std(skipna=True, ddof=0)
        if str_std and str_std > 0:
            str_z_global = (str_signal_global - str_mean) / str_std
        else:
            str_z_global = pd.Series(0.0, index=str_signal_global.index)

        # 4. Combine factors, then apply drift regime filter
        value_weight = float(params.get("value_weight", 0.70))
        value_weight = max(0.0, min(1.0, value_weight))
        combined_signal = value_weight * value_rank_global + (1.0 - value_weight) * str_z_global

        dr_mask = current_drift_regime & combined_signal.notna()
        edge_signal = combined_signal.where(dr_mask, 0.0)

        edge_valid = edge_signal.replace([np.inf, -np.inf], np.nan).dropna()
        edge_non_zero = edge_valid[edge_valid != 0.0]
        if edge_non_zero.empty:
            return self._empty_weights(all_historical_data, current_date)

        edge_mean = edge_non_zero.mean()
        edge_std = edge_non_zero.std(ddof=0)
        if edge_std and edge_std > 0:
            edge_z = (edge_non_zero - edge_mean) / edge_std
        else:
            edge_z = pd.Series(0.0, index=edge_non_zero.index)

        edge_z = edge_z.reindex(edge_signal.index).fillna(0.0)

        # 5. Portfolio Construction
        num_holdings = params.get("num_holdings", 10)
        leverage = params.get("leverage", 1.0)

        weight_indices = (
            all_historical_data.columns.get_level_values("Ticker").unique()
            if isinstance(all_historical_data.columns, pd.MultiIndex)
            else all_historical_data.columns
        )
        weights = pd.Series(0.0, index=weight_indices)

        selection_mode = params.get("selection_mode", "top_n")
        weighting_mode = params.get("weighting_mode", "proportional")
        candidates = edge_z[edge_z != 0.0]
        num_candidates = len(candidates)

        actual_n = 0
        if num_candidates > 0:
            trade_shorts = params.get("trade_shorts", True)
            trade_longs = params.get("trade_longs", True)
            if selection_mode == "all":
                long_mask = candidates > 0.0
                short_mask = candidates < 0.0
                if not trade_longs:
                    long_mask = pd.Series(False, index=candidates.index)
                if not trade_shorts:
                    short_mask = pd.Series(False, index=candidates.index)

                if trade_longs and trade_shorts:
                    long_candidates = candidates[long_mask]
                    short_candidates = candidates[short_mask]
                    if weighting_mode == "equal":
                        long_count = int(long_mask.sum())
                        short_count = int(short_mask.sum())
                        if long_count > 0:
                            weights[long_candidates.index] = (0.5 * leverage) / long_count
                        if short_count > 0:
                            weights[short_candidates.index] = (-0.5 * leverage) / short_count
                        actual_n = int(long_count + short_count)
                    else:
                        long_sum = long_candidates.sum()
                        short_sum = -short_candidates.sum()
                        if long_sum > 0:
                            weights[long_candidates.index] = (long_candidates / long_sum) * (
                                0.5 * leverage
                            )
                        if short_sum > 0:
                            weights[short_candidates.index] = (short_candidates / short_sum) * (
                                0.5 * leverage
                            )
                        actual_n = int(long_mask.sum() + short_mask.sum())
                elif trade_longs:
                    long_candidates = candidates[long_mask]
                    if weighting_mode == "equal":
                        long_count = int(long_mask.sum())
                        if long_count > 0:
                            weights[long_candidates.index] = leverage / long_count
                            actual_n = long_count
                    else:
                        long_sum = long_candidates.sum()
                        if long_sum > 0:
                            weights[long_candidates.index] = (long_candidates / long_sum) * leverage
                            actual_n = int(long_mask.sum())
                elif trade_shorts:
                    short_candidates = candidates[short_mask]
                    if weighting_mode == "equal":
                        short_count = int(short_mask.sum())
                        if short_count > 0:
                            weights[short_candidates.index] = (-leverage) / short_count
                            actual_n = short_count
                    else:
                        short_sum = -short_candidates.sum()
                        if short_sum > 0:
                            weights[short_candidates.index] = (short_candidates / short_sum) * (
                                leverage
                            )
                            actual_n = int(short_mask.sum())
            else:
                if trade_longs and trade_shorts:
                    actual_n = min(num_holdings, num_candidates // 2)
                    if actual_n > 0:
                        sorted_cand = candidates.sort_values(ascending=False)
                        top_assets = sorted_cand.head(actual_n)
                        bottom_assets = sorted_cand.tail(actual_n)

                        weights[top_assets.index] = leverage / actual_n
                        weights[bottom_assets.index] -= leverage / actual_n
                elif trade_longs:
                    actual_n = min(num_holdings, num_candidates)
                    if actual_n > 0:
                        top_assets = candidates.sort_values(ascending=False).head(actual_n)
                        weights[top_assets.index] = leverage / actual_n
                elif trade_shorts:
                    actual_n = min(num_holdings, num_candidates)
                    if actual_n > 0:
                        bottom_assets = candidates.sort_values(ascending=True).head(actual_n)
                        weights[bottom_assets.index] = -leverage / actual_n

        # Diagnostics
        if logger.isEnabledFor(logging.DEBUG):
            msg = f"Date: {current_date} | Drift: {num_candidates} | N: {actual_n}"
            logger.debug(msg)

        output = pd.DataFrame(weights).T
        output.index = pd.Index([current_date])

        return self._enforce_trade_direction_constraints(output)

    def generate_target_weights(self, context: StrategyContext) -> pd.DataFrame:
        cols = list(context.universe_tickers)
        idx = pd.DatetimeIndex(context.rebalance_dates)
        fill_value = float("nan") if context.use_sparse_nan_for_inactive_rows else 0.0
        out = pd.DataFrame(fill_value, index=idx, columns=cols, dtype=float)
        if len(idx) == 0 or len(cols) == 0:
            return out

        asset = context.asset_data
        bench = context.benchmark_data
        nu = context.non_universe_data
        expanding_ends, date_masks = _expanding_iloc_ends(asset.index, idx)
        sig = inspect.signature(self.generate_signals)
        has_nu = "non_universe_historical_data" in sig.parameters
        wfo_start = context.wfo_start_date
        wfo_end = context.wfo_end_date

        for i, d in enumerate(idx):
            if expanding_ends is not None:
                end = int(expanding_ends[i])
                ahist = asset.iloc[:end]
                bhist = bench.iloc[:end]
                nu_hist = nu.iloc[:end] if len(nu.index) > 0 else nu
            else:
                assert date_masks is not None
                mask = date_masks[d]
                ahist = asset.loc[mask]
                bhist = bench.loc[mask]
                nu_hist = nu.loc[mask] if len(nu.index) > 0 else nu
            kw: Dict[str, Any] = {}
            if has_nu:
                kw["non_universe_historical_data"] = None if nu_hist.empty else nu_hist
            row_df = self.generate_signals(
                all_historical_data=ahist,
                benchmark_historical_data=bhist,
                current_date=d,
                start_date=wfo_start,
                end_date=wfo_end,
                **kw,
            )
            if row_df is None or row_df.empty:
                continue
            row_series = row_df.iloc[0].reindex(cols)
            out.loc[d, :] = row_series.astype(float)
        return out

    def _empty_weights(self, data: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """Return zero weights for all assets."""
        tickers = self._get_tickers(data)
        return pd.DataFrame(0.0, index=[date], columns=tickers)

    def _get_tickers(self, data: pd.DataFrame) -> List[str]:
        if isinstance(data.columns, pd.MultiIndex):
            return list(data.columns.get_level_values("Ticker").unique())
        return list(data.columns)
