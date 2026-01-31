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
            "drift_window": {"type": "int", "min": 21, "max": 126, "default": 63},
            "drift_threshold": {"type": "float", "min": 0.5, "max": 0.8, "default": 0.6},
            "reversal_window": {"type": "int", "min": 5, "max": 30, "default": 10},
            "num_holdings": {"type": "int", "min": 5, "max": 50, "default": 10},
            "leverage": {"type": "float", "min": 0.5, "max": 2.0, "default": 1.0},
            "smoothing_lambda": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
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
            
        # Filter to data up to current_date
        prices = prices[prices.index <= current_date]
        
        if len(prices) < max(params["drift_window"], params["reversal_window"]):
            return self._empty_weights(all_historical_data, current_date)

        # 2. Calculate Drift Regime
        # Return > 0 check
        daily_returns = prices.pct_change(fill_method=None)
        positive_days = (daily_returns > 0).rolling(window=params["drift_window"]).sum()
        drift_fraction = positive_days / params["drift_window"]
        
        # Current drift regime flags
        current_drift_regime = drift_fraction.loc[current_date] > params["drift_threshold"]
        
        # 3. Calculate Factors
        # Short-Term Reversal Factor (STR) - 10-day in paper
        reversal_window = params.get("reversal_window", 10)
        str_return = prices.pct_change(periods=reversal_window).loc[current_date]
        str_signal = -str_return # Reversal: high returns -> low/negative signal
        
        # Value Factor (Paper uses Inverse Price 1/P)
        current_prices = prices.loc[current_date]
        value_signal = 1.0 / current_prices
        
        # 4. Combine Signals (Cross-sectional ranking as per paper)
        # We only care about stocks in the drift regime
        str_signal_dr = str_signal[current_drift_regime]
        value_signal_dr = value_signal[current_drift_regime]
        
        if str_signal_dr.empty:
             return self._empty_weights(all_historical_data, current_date)
             
        # Convert to Percentile Ranks (0 to 1) as described in the paper
        def get_ranks(series: pd.Series) -> pd.Series:
            if series.empty:
                return series
            return series.rank(pct=True)
            
        str_rank = get_ranks(str_signal_dr)
        value_rank = get_ranks(value_signal_dr)
        
        # Combined Signal: 70% Value + 30% Reversal
        combined_signal = 0.7 * value_rank + 0.3 * str_rank
        
        # 5. Portfolio Construction
        # Rank and select top N
        num_holdings = params.get("num_holdings", 10)
        top_assets = combined_signal.sort_values(ascending=False).head(num_holdings)
        
        if top_assets.empty:
            return self._empty_weights(all_historical_data, current_date)
            
        # Equal weight allocation among the top N drift-stocks
        weights = pd.Series(0.0, index=prices.columns)
        weights[top_assets.index] = 1.0 / len(top_assets.index)
        
        # Apply leverage
        weights = weights * params.get("leverage", 1.0)
        
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
