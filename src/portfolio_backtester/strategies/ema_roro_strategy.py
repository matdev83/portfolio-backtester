"""
EMA Crossover Strategy with RoRo Signal Integration

This strategy extends the basic EMA crossover strategy by integrating Risk-on/Risk-off (RoRo) signals.
During risk-off periods, the leverage is reduced by half to manage downside risk.

- Long signal: Fast EMA crosses above Slow EMA
- Exit signal: Fast EMA crosses below Slow EMA  
- Risk management: During RoRo risk-off periods, leverage is reduced by 50%
"""

from typing import Optional, Set

import pandas as pd

from ..roro_signals import DummyRoRoSignal
from .ema_crossover_strategy import EMAStrategy

# Optional Numba optimisation
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class EMARoRoStrategy(EMAStrategy):
    """EMA crossover strategy with RoRo signal integration for risk management."""
    
    # PLACEHOLDER: Set the RoRo signal class to use the dummy implementation with hardcoded dates
    # TODO: Replace DummyRoRoSignal with actual RoRo signal implementation when available
    roro_signal_class = DummyRoRoSignal
    
    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        # Store the base leverage for risk-off adjustments
        self.base_leverage = self.leverage
        self.risk_off_leverage_multiplier = strategy_config.get('risk_off_leverage_multiplier', 0.5)
        
    @staticmethod
    def tunable_parameters() -> Set[str]:
        """Return set of tunable parameters for this strategy."""
        base_params = EMAStrategy.tunable_parameters()
        return base_params | {'risk_off_leverage_multiplier'}
    
    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
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
        # Check if we should generate signals for this date
        if start_date is not None and current_date < start_date:
            return pd.DataFrame(index=[current_date], columns=all_historical_data.columns).fillna(0.0)
        if end_date is not None and current_date > end_date:
            return pd.DataFrame(index=[current_date], columns=all_historical_data.columns).fillna(0.0)
        
        # Get RoRo signal for risk management
        roro_signal_instance = self.get_roro_signal()
        is_risk_on = True  # Default to risk-on if no RoRo signal
        
        if roro_signal_instance:
            is_risk_on = roro_signal_instance.generate_signal(
                all_historical_data, 
                benchmark_historical_data, 
                current_date
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
                current_date,
                start_date,
                end_date
            )
        finally:
            # Restore original leverage
            self.leverage = original_leverage
        
        return signals
    
    def __str__(self):
        return f"EMARoRo({self.fast_ema_days},{self.slow_ema_days},mult={self.risk_off_leverage_multiplier})"