from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from portfolio_backtester.strategies._core.base import PortfolioStrategy

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

    def __init__(self, strategy_config: Dict[str, Any]) -> None:
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
        # To calculate 63-day drift + 1-day return, we need about 65 rows. 
        # Using a bit more (e.g. 2x window) for safety with NaNs.
        lookback_needed = max(params["drift_window"], params["reversal_window"]) + 5
        prices = prices[prices.index <= current_date].tail(lookback_needed)
        
        if len(prices) < 2:
            return self._empty_weights(all_historical_data, current_date)

        # 2. Calculate Drift Regime
        # Handle gaps in R2K data: calculate fraction based on non-NaN returns
        daily_returns = prices.pct_change(fill_method=None)
        is_positive = (daily_returns > 0)
        is_valid = daily_returns.notna()

        # We need enough history for a meaningful drift check
        drift_window = params["drift_window"]
        # Only use the window ending at current_date
        positive_days = is_positive.tail(drift_window).sum()
        valid_days = is_valid.tail(drift_window).sum()
        
        # Current drift regime flags (Series indexed by Ticker)
        drift_fraction = (positive_days / valid_days).fillna(0)
        current_drift_regime = (drift_fraction > params["drift_threshold"])
        
        # 3. Calculate Factors across the WHOLE universe first (Global Ranking)
        # This ensures we pick the real "best" value/reversal stocks, not just best of a small subset.
        
        # Short-Term Reversal Factor (STR) - 10-day in paper
        reversal_window = params.get("reversal_window", 10)
        # Calculate return over the reversal window (ratio of current to price N-days ago)
        if len(prices) > reversal_window:
            current_prices = prices.iloc[-1]
            past_prices = prices.iloc[-reversal_window-1]
            str_return = (current_prices / past_prices) - 1
        else:
            str_return = prices.pct_change(periods=reversal_window).loc[current_date]
            
        str_signal_global = -str_return # Reversal: losers get high score
        
        # Value Factor (Paper uses Inverse Price 1/P)
        current_prices = prices.loc[current_date]
        
        # Penny Stock Filter: avoid stocks < $1 as they are often extremely noisy in R2K
        valid_price_mask = current_prices >= 1.0
        value_signal_global = (1.0 / current_prices).where(valid_price_mask)

        if logger.isEnabledFor(logging.INFO):
            num_in_regime = current_drift_regime.sum()
            total_universe = len(current_drift_regime)
            msg = f"Date: {current_date} | Universe: {total_universe} | In Drift Regime: {num_in_regime} | Valid Price: {valid_price_mask.sum()}"
            logger.info(msg)
            # Temporary direct debug print to diagnose user issue
            if total_universe > 0:
                print(msg)

        
        # 4. Combine Signals using Global Ranks
        # Paper uses Percentile Ranks (0 to 1)
        def get_global_ranks(series: pd.Series) -> pd.Series:
            if series.empty:
                return series
            return series.rank(pct=True)
            
        str_rank = get_global_ranks(str_signal_global)
        value_rank = get_global_ranks(value_signal_global)
        
        # Combined Signal: 70% Value + 30% Reversal (Global)
        combined_signal = 0.7 * value_rank.fillna(0) + 0.3 * str_rank.fillna(0)
        
        # 5. Portfolio Construction: Apply Drift Regime Filter
        # Only stocks in the drift regime are eligible candidates
        candidates = combined_signal[current_drift_regime]
        
        if candidates.empty:
             return self._empty_weights(all_historical_data, current_date)
        
        num_holdings = params.get("num_holdings", 10)
        leverage = params.get("leverage", 1.0)
        weights = pd.Series(0.0, index=prices.columns)

        # Long Side
        if params.get("trade_longs", True):
            top_assets = candidates.sort_values(ascending=False).head(num_holdings)
            if not top_assets.empty:
                weights[top_assets.index] = leverage / len(top_assets.index)

        # Short Side
        if params.get("trade_shorts", True):
            bottom_assets = candidates.sort_values(ascending=True).head(num_holdings)
            if not bottom_assets.empty:
                # If we were already long (unlikely with top/bottom decile, but for safety)
                # We subtract from weights
                weights[bottom_assets.index] -= leverage / len(bottom_assets.index)
        
        # Convert to DataFrame
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
