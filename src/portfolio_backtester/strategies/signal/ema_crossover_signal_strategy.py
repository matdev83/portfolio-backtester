"""
Simple EMA Crossover Strategy

This strategy uses exponential moving average crossovers to generate buy/sell signals.
- Long signal: Fast EMA crosses above Slow EMA
- Exit signal: Fast EMA crosses below Slow EMA

SOLID Refactoring: This strategy now uses polymorphic interfaces instead of isinstance checks:
- Uses ISignalPriceExtractor for polymorphic price extraction
- Uses IColumnHandler for polymorphic column operations
"""

from typing import Any, Dict, Optional, cast
import logging

import pandas as pd

from ..base.signal_strategy import SignalStrategy
from ...interfaces.signal_price_extractor_interface import (
    SignalPriceExtractorFactory,
    ISignalPriceExtractor,
)
from ...interfaces.column_handler_interface import (
    ColumnHandlerFactory,
    IColumnHandler,
)

# Import strategy base interface for composition instead of inheritance
from ...interfaces.strategy_base_interface import IStrategyBase, StrategyBaseFactory

logger = logging.getLogger(__name__)


class EmaCrossoverSignalStrategy(SignalStrategy):
    """Simple EMA crossover strategy implementation."""

    def __init__(self, strategy_config: dict):
        # Use composition instead of inheritance - create strategy base via factory
        self._strategy_base: IStrategyBase = StrategyBaseFactory.create_strategy_base(
            strategy_config
        )

        # Still call super() for SignalStrategy compatibility, but minimize dependency
        super().__init__(strategy_config)
        self.fast_ema_days = strategy_config.get("fast_ema_days", 20)
        self.slow_ema_days = strategy_config.get("slow_ema_days", 64)
        self.leverage = strategy_config.get("leverage", 1.0)

        # SOLID: Polymorphic components to eliminate isinstance violations
        self._price_extractor_cache: Dict[int, ISignalPriceExtractor] = {}
        self._column_handler_cache: Dict[int, IColumnHandler] = {}

    def _get_price_extractor(self, data: pd.DataFrame) -> ISignalPriceExtractor:
        """Get appropriate price extractor for the DataFrame structure."""
        data_key = id(data.columns)
        if data_key not in self._price_extractor_cache:
            self._price_extractor_cache[data_key] = SignalPriceExtractorFactory.create(data)
        return self._price_extractor_cache[data_key]

    def _get_column_handler(self, data: pd.DataFrame) -> IColumnHandler:
        """Get appropriate column handler for the DataFrame structure."""
        data_key = id(data.columns)
        if data_key not in self._column_handler_cache:
            self._column_handler_cache[data_key] = ColumnHandlerFactory.create(data)
        return self._column_handler_cache[data_key]

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """Return a dictionary defining the tunable parameters for this strategy."""
        return {
            "fast_ema_days": {
                "type": "int",
                "default": 20,
                "min": 10,
                "max": 50,
                "step": 5,
            },
            "slow_ema_days": {
                "type": "int",
                "default": 64,
                "min": 50,
                "max": 200,
                "step": 10,
            },
            "leverage": {
                "type": "float",
                "default": 1.0,
                "min": 1.0,
                "max": 1.0,
                "step": 0.1,
            },
        }

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
        Generate EMA crossover signals.

        Args:
            all_historical_data: DataFrame with historical OHLCV data for all assets
            benchmark_historical_data: DataFrame with historical OHLCV data for benchmark
            current_date: The current date for signal generation
            start_date: Optional start date for WFO window
            end_date: Optional end date for WFO window

        Returns:
            DataFrame with signals (weights) for the current date
        """
        # Check if current_date is provided
        # Handle None current_date gracefully - use the last date in the data
        if current_date is None:
            current_date = pd.Timestamp(all_historical_data.index[-1])
        # Check if we should generate signals for this date
        if start_date is not None and current_date < start_date:
            return pd.DataFrame(index=[current_date], columns=all_historical_data.columns).fillna(
                0.0
            )
        if end_date is not None and current_date > end_date:
            return pd.DataFrame(index=[current_date], columns=all_historical_data.columns).fillna(
                0.0
            )

        # Extract close prices using polymorphic approach - eliminates isinstance violations
        extractor = self._get_price_extractor(all_historical_data)
        close_prices = extractor.extract_all_close_prices(all_historical_data, "Close")

        # Get universe tickers using polymorphic approach - eliminates isinstance violations
        available_tickers = extractor.get_available_tickers(all_historical_data)

        # --------------------------------------------------------------
        # Vectorised EMA calculation for all tickers.
        # Require only *exactly* max(fast, slow) periods of history instead
        # of +10; this prevents ‘no data’ on the very first WFO window.
        # --------------------------------------------------------------

        min_periods = max(self.fast_ema_days, self.slow_ema_days)
        valid_mask = close_prices.notna().sum() >= min_periods
        valid_tickers = close_prices.columns[valid_mask]
        fast_ema = close_prices[valid_tickers].ewm(span=self.fast_ema_days).mean()
        slow_ema = close_prices[valid_tickers].ewm(span=self.slow_ema_days).mean()
        # Get EMA values at current date (or closest available date)
        if current_date in fast_ema.index and current_date in slow_ema.index:
            fast_values = fast_ema.loc[current_date]
            slow_values = slow_ema.loc[current_date]
            signal_mask = (
                (fast_values > slow_values) & (~fast_values.isna()) & (~slow_values.isna())
            )
            weights = pd.Series(0.0, index=available_tickers)
            weights.loc[signal_mask.index[signal_mask]] = 1.0
        else:
            # If current_date is not available, use the last available date before current_date
            available_dates = fast_ema.index[fast_ema.index <= current_date]
            if len(available_dates) > 0:
                last_available_date = available_dates[-1]
                fast_values = fast_ema.loc[last_available_date]
                slow_values = slow_ema.loc[last_available_date]
                signal_mask = (
                    (fast_values > slow_values) & (~fast_values.isna()) & (~slow_values.isna())
                )
                weights = pd.Series(0.0, index=available_tickers)
                weights.loc[signal_mask.index[signal_mask]] = 1.0
                import logging

                logging.getLogger(__name__).debug(
                    f"EMAStrategy: using {last_available_date} instead of {current_date}"
                )
            else:
                weights = pd.Series(0.0, index=available_tickers)
        # Equal-weight allocation among selected stocks
        if weights.sum() > 0:
            weights = weights / weights.sum()
            # Apply leverage
            weights = weights * self.leverage
        else:
            import logging

            logging.getLogger(__name__).debug(
                "EMAStrategy: no positions selected on %s", current_date
            )

        # 🚨 CRITICAL: Apply stop loss for signal strategies (fallback for non-daily evaluation)
        # Note: This is a fallback - primary stop loss is handled by daily monitor in WindowEvaluator
        try:
            # Extract current prices for the current date
            current_prices = (
                close_prices.loc[current_date]
                if current_date in close_prices.index
                else close_prices.iloc[-1]
            )
            if isinstance(current_prices, pd.DataFrame):
                current_prices = current_prices.iloc[0]  # Get first row if DataFrame

            # Ensure current_prices is a Series for type safety
            current_prices_series = (
                pd.Series(current_prices)
                if not isinstance(current_prices, pd.Series)
                else current_prices
            )

            weights_after_sl = self._apply_signal_strategy_stop_loss(
                weights, current_date, all_historical_data, current_prices_series
            )
            if not weights_after_sl.equals(weights):
                logger.info(f"Signal strategy stop loss applied on {current_date.date()}")
                weights = weights_after_sl
        except Exception as e:
            logger.error(f"Error applying signal strategy stop loss on {current_date.date()}: {e}")
            # Continue with original weights if stop loss fails

        # 🚨 CRITICAL: Apply take profit for signal strategies (fallback for non-daily evaluation)
        # Note: This is a fallback - primary take profit is handled by daily monitor in WindowEvaluator
        try:
            weights_after_tp = self._apply_signal_strategy_take_profit(
                weights, current_date, all_historical_data, current_prices_series
            )
            if not weights_after_tp.equals(weights):
                logger.info(f"Signal strategy take profit applied on {current_date.date()}")
                weights = weights_after_tp
        except Exception as e:
            logger.error(
                f"Error applying signal strategy take profit on {current_date.date()}: {e}"
            )
            # Continue with original weights if take profit fails

        # Return as DataFrame with current date as index
        result_df = pd.DataFrame([weights], index=[current_date])

        # Enforce trade direction constraints - this will raise an exception if violated
        result_df = self._enforce_trade_direction_constraints(result_df)

        return result_df

    def _apply_signal_strategy_stop_loss(
        self,
        weights: pd.Series,
        current_date: pd.Timestamp,
        all_historical_data: pd.DataFrame,
        current_prices: pd.Series,
    ) -> pd.Series:
        """Apply stop loss logic for signal strategies (fallback method)."""
        # Get stop loss handler via delegation
        sl_handler = self.get_stop_loss_handler()

        # Skip if no stop loss configured
        if sl_handler.__class__.__name__ == "NoStopLoss":
            return weights

        # Get current positions and entry prices (simplified for signal strategies)
        # Note: Signal strategies typically don't maintain position state between calls
        # This is a simplified implementation - primary stop loss is handled by daily monitor
        try:
            # For signal strategies, we'll use current prices as "entry prices"
            # This is not ideal but serves as a fallback until daily monitoring takes over
            entry_prices = current_prices.copy()

            # Calculate stop levels
            stop_levels = sl_handler.calculate_stop_levels(
                current_date=current_date,
                asset_ohlc_history=all_historical_data,
                current_weights=weights,
                entry_prices=entry_prices,
            )

            # Apply stop loss
            weights_after_sl = sl_handler.apply_stop_loss(
                current_date=current_date,
                current_asset_prices=current_prices,
                target_weights=weights,
                entry_prices=entry_prices,
                stop_levels=stop_levels,
            )

            return cast(pd.Series, weights_after_sl)

        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Signal strategy stop loss failed on {current_date.date()}: {e}"
            )
            return weights

    def _apply_signal_strategy_take_profit(
        self,
        weights: pd.Series,
        current_date: pd.Timestamp,
        all_historical_data: pd.DataFrame,
        current_prices: pd.Series,
    ) -> pd.Series:
        """Apply take profit logic for signal strategies (fallback method)."""
        # Get take profit handler via delegation
        tp_handler = self.get_take_profit_handler()

        # Skip if no take profit configured
        if tp_handler.__class__.__name__ == "NoTakeProfit":
            return weights

        # Get current positions and entry prices (simplified for signal strategies)
        # Note: Signal strategies typically don't maintain position state between calls
        # This is a simplified implementation - primary take profit is handled by daily monitor
        try:
            # For signal strategies, we'll use current prices as "entry prices"
            # This is not ideal but serves as a fallback until daily monitoring takes over
            entry_prices = current_prices.copy()

            # Calculate take profit levels
            take_profit_levels = tp_handler.calculate_take_profit_levels(
                current_date=current_date,
                asset_ohlc_history=all_historical_data,
                current_weights=weights,
                entry_prices=entry_prices,
            )

            # Apply take profit
            weights_after_tp = tp_handler.apply_take_profit(
                current_date=current_date,
                current_asset_prices=current_prices,
                target_weights=weights,
                entry_prices=entry_prices,
                take_profit_levels=take_profit_levels,
            )

            return weights_after_tp

        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Signal strategy take profit failed on {current_date.date()}: {e}"
            )
            return weights

    def __str__(self):
        return f"EMA({self.fast_ema_days},{self.slow_ema_days})"

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

    def get_non_universe_data_requirements(self):
        """Get non-universe data requirements via interface delegation."""
        return self._strategy_base.get_non_universe_data_requirements()

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
