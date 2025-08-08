from typing import Any, Dict, Optional
import pandas as pd

from .base_momentum_portfolio_strategy import BaseMomentumPortfolioStrategy


class MomentumUnfilteredAtrPortfolioStrategy(BaseMomentumPortfolioStrategy):
    def generate_signals(
        self,
        all_historical_data: "pd.DataFrame",
        benchmark_historical_data: "pd.DataFrame",
        non_universe_historical_data: "Optional[pd.DataFrame]" = None,  # Optional, for compatibility
        current_date: "Optional[pd.Timestamp]" = None,
        start_date: "Optional[pd.Timestamp]" = None,
        end_date: "Optional[pd.Timestamp]" = None,
    ) -> "pd.DataFrame":
        """
        Generates trading signals for the MomentumUnfilteredAtrPortfolioStrategy.

        Parameters:
            all_historical_data: pd.DataFrame
            benchmark_historical_data: pd.DataFrame
            non_universe_historical_data: Optional[pd.DataFrame] (optional, unused)
            current_date: Optional[pd.Timestamp]
            start_date: Optional[pd.Timestamp]
            end_date: Optional[pd.Timestamp]
        """
        # Delegate to parent, ignoring non_universe_historical_data
        return pd.DataFrame(
            super().generate_signals(
                all_historical_data,
                benchmark_historical_data,
                current_date=current_date,
                start_date=start_date,
                end_date=end_date,
            )
        )

    """
    Momentum strategy variant that is unfiltered (no SMA/RoRo filters) but includes ATR-based stop loss.
    This strategy is based on the SimpleMomentumPortfolioStrategy but with:
    - No SMA filtering (sma_filter_window disabled)
    - No RoRo filtering (roro_signal disabled)
    - ATR-based stop loss enabled by default
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        # Set default parameters for unfiltered momentum with ATR stop loss
        strategy_defaults = {
            "lookback_months": 12,
            "skip_months": 0,
            "num_holdings": 10,
            "top_decile_fraction": 0.1,
            "smoothing_lambda": 0.0,  # Disable EWMA signal smoothing
            "leverage": 1.0,
            "trade_longs": True,
            "trade_shorts": False,
            "sma_filter_window": None,  # Explicitly disable SMA filtering
            "derisk_days_under_sma": 10,
            "apply_trading_lag": False,
            "price_column_asset": "Close",
            "price_column_benchmark": "Close",
        }

        # Set default ATR stop loss configuration
        stop_loss_defaults: Dict[str, str | int | float] = {
            "type": "AtrBasedStopLoss",
            "atr_length": 14,
            "atr_multiple": 2.5,
        }

        # Ensure strategy_params exists
        if "strategy_params" not in strategy_config:
            strategy_config["strategy_params"] = {}

        # Apply strategy defaults to strategy_params
        for key, value in strategy_defaults.items():
            strategy_config["strategy_params"].setdefault(key, value)

        # Set up stop loss configuration
        if "stop_loss_config" not in strategy_config:
            strategy_config["stop_loss_config"] = {}

        # Apply stop loss defaults, but allow individual parameters to be overridden
        for key, value in stop_loss_defaults.items():
            strategy_config["stop_loss_config"].setdefault(str(key), value)

        # Allow ATR parameters to be set directly in strategy_params for optimization
        # and copy them to stop_loss_config
        strategy_params = strategy_config["strategy_params"]
        stop_loss_config = strategy_config["stop_loss_config"]

        if "atr_length" in strategy_params:
            stop_loss_config["atr_length"] = strategy_params["atr_length"]
        if "atr_multiple" in strategy_params:
            stop_loss_config["atr_multiple"] = strategy_params["atr_multiple"]

        # Initialize parent class
        super().__init__(strategy_config)

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """Return the set of tunable parameters for optimization."""
        # Get base momentum parameters but exclude SMA-related ones since this is unfiltered
        base_params = {
            "lookback_months": {"type": "int", "min": 1, "max": 60, "default": 12},
            "skip_months": {"type": "int", "min": 0, "max": 12, "default": 0},
            "num_holdings": {"type": "int", "min": 1, "max": 50, "default": 10},
            "top_decile_fraction": {"type": "float", "min": 0.01, "max": 1.0, "default": 0.1},
            "smoothing_lambda": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            "leverage": {"type": "float", "min": 0.1, "max": 5.0, "default": 1.0},
            "trade_longs": {"type": "bool", "default": True},
            "trade_shorts": {"type": "bool", "default": False},
            "apply_trading_lag": {"type": "bool", "default": False},
        }
        # Add ATR-specific parameters
        atr_params = {
            "atr_length": {"type": "int", "min": 5, "max": 50, "default": 14},
            "atr_multiple": {"type": "float", "min": 0.5, "max": 5.0, "default": 2.5},
        }
        return {**base_params, **atr_params}

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods for MomentumUnfilteredAtrStrategy.
        Requires: lookback_months + skip_months + ATR requirements
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)

        # Base momentum calculation requirement
        lookback_months = params.get("lookback_months", 12)
        skip_months = params.get("skip_months", 0)
        momentum_requirement = lookback_months + skip_months

        # ATR requirement for stop loss
        atr_length = params.get("atr_length", 14)
        # Convert daily ATR periods to months (roughly) and add buffer
        # ATR typically needs daily OHLC data, so we need more historical data
        atr_requirement = max(2, atr_length // 20)  # ~20 trading days per month

        # Take the maximum of all requirements plus buffer
        total_requirement = max(momentum_requirement, atr_requirement)

        # Add 3-month buffer for reliable ATR calculations (needs more data than basic momentum)
        return int(total_requirement + 3)

    def get_roro_signal(self):
        """Override to disable RoRo filtering for unfiltered strategy."""
        return None
