from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast, Union, Mapping, TYPE_CHECKING

import numpy as np
import pandas as pd
from portfolio_backtester.strategies._core.base import PortfolioStrategy

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

    def __init__(self, strategy_config: Union[Mapping[str, Any], "CanonicalScenarioConfig"]) -> None:
        super().__init__(strategy_config)
        
        params = self._get_params_dict()
        
        # Strategy specific defaults
        defaults = {
            "drift_window": 63,
            "drift_threshold": 0.60,
            "reversal_window": 10,
            "num_holdings": 10,
            "leverage": 1.0,
            "smoothing_lambda": 0.0, # Default to no smoothing for this strategy
            "price_column_asset": "Close",
            "trade_longs": True,
            "trade_shorts": True,
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
            "universe_config": {"type": "dict", "default": None},
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
            prices = all_historical_data.xs(price_col, level="Field", axis=1)
        else:
            prices = all_historical_data
            
        # Filter to data up to current_date and only keep what's needed for windows
        lookback_needed = max(params["drift_window"], params["reversal_window"]) + 5
        prices = prices[prices.index <= current_date].tail(lookback_needed)
        
        if len(prices) < 2:
            return self._empty_weights(all_historical_data, current_date)

        # 2. Calculate Drift Regime
        daily_returns = prices.pct_change(fill_method=None)
        is_positive = (daily_returns > 0)
        is_valid = daily_returns.notna()

        drift_window = params["drift_window"]
        positive_days = is_positive.tail(drift_window).sum()
        valid_days = is_valid.tail(drift_window).sum()
        
        drift_fraction = (positive_days / valid_days).fillna(0)
        current_drift_regime = (drift_fraction > params["drift_threshold"])
        
        # 3. Calculate Raw Factor Signals (Global)
        reversal_window = params.get("reversal_window", 10)
        if len(prices) > reversal_window:
            str_return = (prices.iloc[-1] / prices.iloc[-reversal_window-1]) - 1
        else:
            str_return = prices.pct_change(periods=reversal_window).loc[current_date]
        
        str_signal_global = -str_return
        
        current_prices = prices.loc[current_date]
        valid_price_mask = (current_prices >= 1.0)
        value_signal_global = (1.0 / current_prices).where(valid_price_mask)

        # 4. Standardize Factors locally WITHIN the Drift Regime
        # This is likely the secret to the early session's performance.
        dr_mask = current_drift_regime & value_signal_global.notna()
        
        if not dr_mask.any():
            return self._empty_weights(all_historical_data, current_date)

        def get_local_percentile(series: pd.Series, mask: pd.Series) -> pd.Series:
            subset = series[mask]
            if subset.empty:
                return pd.Series(0.5, index=series.index)
            ranks = subset.rank(pct=True)
            return ranks.reindex(series.index).fillna(0.0)

        value_rank_local = get_local_percentile(value_signal_global, dr_mask)
        str_rank_local = get_local_percentile(str_signal_global, dr_mask)
        
        # Combined Signal (Local)
        # Weighting: 70% Value, 30% Reversal
        combined_signal = 0.7 * value_rank_local + 0.3 * str_rank_local
        
        # 5. Portfolio Construction
        num_holdings = params.get("num_holdings", 10)
        leverage = params.get("leverage", 1.0)
        
        weight_indices = all_historical_data.columns.get_level_values("Ticker").unique() if isinstance(all_historical_data.columns, pd.MultiIndex) else all_historical_data.columns
        weights = pd.Series(0.0, index=weight_indices)

        # Candidates are drifting stocks only
        candidates = combined_signal[dr_mask]
        num_candidates = len(candidates)

        if num_candidates > 0:
            if params.get("trade_shorts", True):
                # Long/Short Split
                actual_n = min(num_holdings, num_candidates // 2)
                if actual_n > 0:
                    sorted_cand = candidates.sort_values(ascending=False)
                    top_assets = sorted_cand.head(actual_n)
                    bottom_assets = sorted_cand.tail(actual_n)
                    
                    weights[top_assets.index] = (leverage / actual_n)
                    weights[bottom_assets.index] -= (leverage / actual_n)
            else:
                # Long Only
                actual_n = min(num_holdings, num_candidates)
                if actual_n > 0:
                    top_assets = candidates.sort_values(ascending=False).head(actual_n)
                    weights[top_assets.index] = (leverage / actual_n)

        # Diagnostics
        if logger.isEnabledFor(logging.INFO):
            msg = f"Date: {current_date} | Drift: {num_candidates} | N: {actual_n if 'actual_n' in locals() else 0}"
            logger.info(msg)
            print(msg)
            
        output = pd.DataFrame(weights).T
        output.index = [current_date]
        
        return self._enforce_trade_direction_constraints(output)

    def _empty_weights(self, data: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """Return zero weights for all assets."""
        tickers = self._get_tickers(data)
        return pd.DataFrame(0.0, index=[date], columns=tickers)

    def _get_tickers(self, data: pd.DataFrame) -> List[str]:
        if isinstance(data.columns, pd.MultiIndex):
            return list(data.columns.get_level_values("Ticker").unique())
        return list(data.columns)
