"""
UVXY RSI Strategy

A strategy that trades UVXY (VIX volatility ETF) based on SPY RSI(2) signals.
- Universe: UVXY only
- Signal: RSI(2) on SPY daily data
- Entry: Short UVXY when SPY RSI(2) falls below threshold (default 30)
- Exit: Cover short on next trading day's close

SOLID Refactoring: This class now uses composition pattern with specialized components:
- RSICalculator: Handles RSI computation
- UvxySignalGenerator: Handles signal generation logic
- PriceDataProcessor: Handles price data extraction and processing
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from ..base.signal_strategy import SignalStrategy
from .rsi_calculator import RSICalculator
from .price_data_processor import PriceDataProcessor

# Import to ensure factory registration and get factory instance
from ...interfaces.signal_generator_interface import signal_generator_factory

# Import strategy base interface for composition instead of inheritance
from ...interfaces.strategy_base_interface import IStrategyBase, StrategyBaseFactory

# Import signal generators to trigger registration

logger = logging.getLogger(__name__)


class UvxyRsiSignalStrategy(SignalStrategy):
    """
    UVXY RSI Strategy implementation.

    This strategy:
    1. Trades only UVXY (universe should contain only UVXY)
    2. Uses SPY daily data for RSI(2) signal generation
    3. Goes short UVXY when SPY RSI(2) < threshold (configurable, default 30)
    4. Covers short position on the next trading day's close (1-day holding period)

    SOLID Refactoring: Uses IStrategyBase interface instead of direct super() calls
    to eliminate tight coupling with parent class implementation.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        # Set default parameters
        defaults = {
            "rsi_period": 2,
            "rsi_threshold": 30.0,
            "price_column_asset": "Close",
            "price_column_benchmark": "Close",
            "trade_longs": True,   # Allow longs
            "trade_shorts": True,  # Must allow shorts for UVXY strategy
        }

        # Ensure nested dict exists and apply defaults
        strategy_params = strategy_config.setdefault("strategy_params", {})
        for k, v in defaults.items():
            strategy_params.setdefault(k, v)

        # Use composition instead of inheritance - create strategy base via factory
        self._strategy_base: IStrategyBase = StrategyBaseFactory.create_strategy_base(
            strategy_config
        )

        # Still call super() for SignalStrategy compatibility, but minimize dependency
        super().__init__(strategy_config)

        # Initialize SOLID components with configuration parameters
        rsi_period = int(strategy_params.get("rsi_period", 2))
        rsi_threshold = float(strategy_params.get("rsi_threshold", 30.0))
        price_column = strategy_params.get("price_column_benchmark", "Close")

        self._rsi_calculator = RSICalculator(period=rsi_period)
        # Use factory pattern to create signal generator
        self._signal_generator = signal_generator_factory.create_generator(
            "uvxy_rsi", {"rsi_threshold": rsi_threshold, "holding_period_days": 1}
        )
        self._price_processor = PriceDataProcessor(price_column=price_column)

        # Legacy state tracking (kept for backward compatibility)
        # The timing controller now handles most state management
        self._previous_signal = 0.0
        self._entry_date_internal: Optional[pd.Timestamp] = None

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """Return the set of parameters that can be optimized."""
        return {
            "rsi_period": {
                "type": "int",
                "default": 2,
                "min": 2,
                "max": 10,
                "step": 1,
            },
            "rsi_threshold": {
                "type": "float",
                "default": 30.0,
                "min": 10.0,
                "max": 50.0,
                "step": 5.0,
            },
        }

    def get_non_universe_data_requirements(self) -> list[str]:
        """Return list of non-universe tickers needed for signal generation."""
        return ["SPY"]

    # Delegate base strategy methods to interface instead of using super()
    def get_timing_controller(self):
        """Get the timing controller via interface delegation."""
        return self._strategy_base.get_timing_controller()

    def supports_daily_signals(self) -> bool:
        """Check if strategy supports daily signals via interface delegation."""
        return bool(self._strategy_base.supports_daily_signals())

    def get_roro_signal(self):
        """Get RoRo signal - returns None by default for this signal strategy."""
        return None

    def get_stop_loss_handler(self):
        """Get stop loss handler via interface delegation."""
        return self._strategy_base.get_stop_loss_handler()

    def get_universe(self, global_config: Dict[str, Any]):
        """Get universe via interface delegation."""
        return self._strategy_base.get_universe(global_config)

    def get_universe_method_with_date(
        self, global_config: Dict[str, Any], current_date: pd.Timestamp
    ):
        """Get universe with date context via interface delegation."""
        return self._strategy_base.get_universe_method_with_date(global_config, current_date)

    def get_synthetic_data_requirements(self) -> bool:
        """Get synthetic data requirements via interface delegation."""
        return bool(self._strategy_base.get_synthetic_data_requirements())

    def get_minimum_required_periods(self) -> int:
        """Get minimum required periods via interface delegation."""
        return int(self._strategy_base.get_minimum_required_periods())

    def validate_data_sufficiency(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ):
        """Validate data sufficiency via interface delegation."""
        return self._strategy_base.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )

    def filter_universe_by_data_availability(
        self,
        all_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        min_periods_override: Optional[int] = None,
    ):
        """Filter universe by data availability via interface delegation."""
        return self._strategy_base.filter_universe_by_data_availability(
            all_historical_data, current_date, min_periods_override
        )

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame],
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate trading signals for UVXY based on SPY RSI.

        This method generates signals for ALL trading days from start_date to end_date,
        not just rebalancing dates, to ensure proper daily signal logic.
        """
        # Get universe tickers using the price processor
        universe_tickers = self._price_processor.get_universe_tickers(all_historical_data)

        # Validate that we have SPY data
        if non_universe_historical_data is None or non_universe_historical_data.empty:
            logger.warning("No SPY data available for signal generation")
            return self._price_processor.create_empty_signals_dataframe(
                universe_tickers, current_date, current_date
            )

        # Extract SPY close prices up to current date using the price processor
        spy_data = self._price_processor.extract_ticker_prices(
            non_universe_historical_data, "SPY", current_date
        )
        if spy_data is None or len(spy_data) < self._rsi_calculator.minimum_data_points:
            logger.warning("Insufficient SPY data for RSI calculation")
            return self._price_processor.create_empty_signals_dataframe(
                universe_tickers, current_date, current_date
            )

        # Calculate RSI using the RSI calculator
        spy_rsi = self._rsi_calculator.calculate(spy_data)

        # Generate signals using the signal generator interface
        if start_date is not None and end_date is not None:
            # Generate signals for the full range using interface method
            range_data: Dict[str, Any] = {"rsi_series": spy_rsi}
            signals = pd.DataFrame(
                self._signal_generator.generate_signals_for_range(
                    range_data, universe_tickers, start_date, end_date
                )
            )
        else:
            # Generate signal for current date only using interface method
            current_rsi = self._rsi_calculator.get_current_rsi(spy_data, current_date)
            date_data: Dict[str, Any] = {"current_rsi": current_rsi}
            signals = pd.DataFrame(
                self._signal_generator.generate_signal_for_date(
                    date_data, universe_tickers, current_date
                )
            )

        # Enforce trade direction constraints - this will raise an exception if violated
        signals = self._enforce_trade_direction_constraints(signals)

        return signals

    # Backward compatibility methods - delegate to signal generator interface
    @property
    def _entry_date(self) -> Optional[pd.Timestamp]:
        """Legacy property for backward compatibility."""
        # Use interface method to check if in position
        return None if not self._signal_generator.is_in_position() else pd.Timestamp.now()

    @_entry_date.setter
    def _entry_date(self, value: Optional[pd.Timestamp]) -> None:
        """Legacy property setter - updates signal generator state."""
        if value is None:
            self._signal_generator.reset_state()
            self._entry_date_internal = None
        else:
            # Can't directly set entry date, but this maintains interface compatibility
            self._entry_date_internal = value
