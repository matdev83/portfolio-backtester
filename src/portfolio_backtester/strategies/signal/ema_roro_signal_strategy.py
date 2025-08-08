"""
EMA Crossover Strategy with RoRo Signal Integration

This strategy extends the basic EMA crossover strategy by integrating Risk-on/Risk-off (RoRo) signals.
During risk-off periods, the leverage is reduced by half to manage downside risk.

- Long signal: Fast EMA crosses above Slow EMA
- Exit signal: Fast EMA crosses below Slow EMA
- Risk management: During RoRo risk-off periods, leverage is reduced by 50%
"""

from typing import Optional, Dict, Any

import pandas as pd

from ...risk_off_signals import DummyRiskOffSignalGenerator
from .ema_crossover_signal_strategy import EmaCrossoverSignalStrategy

# No dual-path implementation needed - strategy uses parent EMA logic


class EmaRoroSignalStrategy(EmaCrossoverSignalStrategy):
    """EMA crossover strategy with RoRo signal integration for risk management."""

    # PLACEHOLDER: Set the RoRo signal class to use the dummy implementation with hardcoded dates
    # TODO: Replace DummyRiskOffSignalGenerator with actual RoRo signal implementation when available
    roro_signal_class = DummyRiskOffSignalGenerator

    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        # Store the base leverage for risk-off adjustments
        self.base_leverage = self.leverage
        self.risk_off_leverage_multiplier = strategy_config.get("risk_off_leverage_multiplier", 0.5)

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """Return dictionary of tunable parameters for this strategy."""
        base_params = EmaCrossoverSignalStrategy.tunable_parameters()
        # Add risk_off_leverage_multiplier parameter
        additional_params = {
            "risk_off_leverage_multiplier": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
            }
        }
        return {**base_params, **additional_params}

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate EMA crossover signals with RoRo risk management.

        Args:
            all_historical_data: DataFrame with historical OHLCV data for all assets
            benchmark_historical_data: DataFrame with historical OHLCV data for benchmark
            current_date: The current date for signal generation
            start_date: Optional start date for WFO window
            end_date: Optional end date for WFO window

        Returns:
            DataFrame with signals (weights) for the current date
        """
        # Handle None current_date gracefully - use the last date in the data
        if current_date is None:
            current_date = all_historical_data.index[-1]

        # Check if we should generate signals for this date
        if start_date is not None and current_date < start_date:
            return pd.DataFrame(index=[current_date], columns=all_historical_data.columns).fillna(
                0.0
            )
        if end_date is not None and current_date > end_date:
            return pd.DataFrame(index=[current_date], columns=all_historical_data.columns).fillna(
                0.0
            )

        # Get RoRo signal for risk management
        roro_signal_instance = self.get_roro_signal()
        is_risk_on = True  # Default to risk-on if no RoRo signal

        if roro_signal_instance:
            is_risk_on = roro_signal_instance.generate_signal(
                all_historical_data, benchmark_historical_data, current_date
            )

        # Adjust leverage based on RoRo signal
        if is_risk_on:
            # Risk-on: use full leverage
            current_leverage = self.base_leverage
        else:
            # Risk-off: reduce leverage by the specified multiplier (default 50%)
            current_leverage = self.base_leverage * self.risk_off_leverage_multiplier

        # Temporarily set the leverage for the parent method
        original_leverage = self.leverage
        self.leverage = current_leverage

        try:
            # Generate signals using the parent EMA strategy logic
            signals = super().generate_signals(
                all_historical_data,
                benchmark_historical_data,
                non_universe_historical_data,
                current_date,
                start_date,
                end_date,
                **kwargs,
            )
        finally:
            # Restore original leverage
            self.leverage = original_leverage

        # Enforce trade direction constraints - this will raise an exception if violated
        signals = self._enforce_trade_direction_constraints(signals)

        return signals

    def __str__(self):
        return f"EMARoRo({self.fast_ema_days},{self.slow_ema_days},mult={self.risk_off_leverage_multiplier})"
