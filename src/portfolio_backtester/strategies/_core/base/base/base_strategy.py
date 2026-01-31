from __future__ import annotations

import logging
from abc import ABC
from typing import Optional, TYPE_CHECKING, Dict, Any, List, Tuple, Mapping, Union, cast

import numpy as np
import pandas as pd

# Try to import njit for performance, fallback to no-op decorator
try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from .....risk_off_signals import IRiskOffSignalGenerator
from .....risk_management.stop_loss_handlers import BaseStopLoss, NoStopLoss
from .....risk_management.take_profit_handlers import BaseTakeProfit, NoTakeProfit

if TYPE_CHECKING:
    from .....canonical_config import CanonicalScenarioConfig
    from .....timing.timing_controller import TimingController


# Custom exception classes for trade direction validation
class TradeDirectionConfigurationError(ValueError):
    """Raised when trade direction configuration is invalid."""

    def __init__(self, strategy_class, trade_longs, trade_shorts, details):
        self.strategy_class = strategy_class
        self.trade_longs = trade_longs
        self.trade_shorts = trade_shorts
        self.details = details
        message = f"Invalid trade direction configuration in {strategy_class}: {details}"
        super().__init__(message)


class TradeDirectionViolationError(ValueError):
    """Raised when a strategy violates its trade direction constraints."""

    def __init__(
        self,
        strategy_class,
        trade_longs,
        trade_shorts,
        violation_details,
        violated_signals=None,
    ):
        self.strategy_class = strategy_class
        self.trade_longs = trade_longs
        self.trade_shorts = trade_shorts
        self.violation_details = violation_details
        self.violated_signals = violated_signals
        message = f"Trade direction violation in {strategy_class}: {violation_details}"
        super().__init__(message)


logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for trading strategies.

    TESTING NOTE: When testing strategy classes, be aware that strategy instances
    are created with a configuration dictionary that contains 'strategy_params'.
    The strategy parameters are stored in self.strategy_config['strategy_params'],
    not directly in self.strategy_config. This is important for test assertions
    that check parameter values.
    """

    # class attribute specifying which Risk-off signal generator to use
    # None means use the default NoRiskOffSignalGenerator via provider
    risk_off_signal_generator_class: type[IRiskOffSignalGenerator] | None = None
    # class attribute specifying which Stop Loss handler to use by default
    stop_loss_handler_class: type[BaseStopLoss] = NoStopLoss
    # class attribute specifying which Take Profit handler to use by default
    take_profit_handler_class: type[BaseTakeProfit] = NoTakeProfit

    def __init__(self, strategy_params: Union[Mapping[str, Any], "CanonicalScenarioConfig"]):
        from .....canonical_config import CanonicalScenarioConfig

        # 1. Store the canonical config if provided
        self.canonical_config: Optional[CanonicalScenarioConfig] = None
        if isinstance(strategy_params, CanonicalScenarioConfig):
            self.canonical_config = strategy_params
            # Extract effective strategy parameters
            effective_params = dict(strategy_params.strategy_params)
        else:
            # Traditional dict-based init
            effective_params = dict(strategy_params)

        # Ensure we have a mutable dictionary for normalization
        # Some strategies (and base class) expect to write back into strategy_params
        self.strategy_params = effective_params

        # If there's a nested strategy_params dict, unfreeze that too
        if "strategy_params" in self.strategy_params and not isinstance(
            self.strategy_params["strategy_params"], dict
        ):
            self.strategy_params["strategy_params"] = dict(self.strategy_params["strategy_params"])

        # For backward compatibility, also set strategy_config
        self.strategy_config = self.strategy_params
        self._risk_off_signal_generator_instance: IRiskOffSignalGenerator | None = None
        self._stop_loss_handler_instance: BaseStopLoss | None = None
        self._take_profit_handler_instance: BaseTakeProfit | None = None
        self.entry_prices: pd.Series | None = None
        self._timing_controller: Optional[TimingController] = None
        self.config: Dict[str, Any] = {}

        # Initialize provider interfaces
        self._universe_provider: Optional[Any] = None
        self._position_sizer_provider: Optional[Any] = None
        self._stop_loss_provider: Optional[Any] = None
        self._take_profit_provider: Optional[Any] = None
        self._risk_off_signal_provider: Optional[Any] = None

        # Initialize trade direction constraints
        self._initialize_trade_direction_constraints()

        self._initialize_timing_controller()
        self._initialize_providers()

    # ------------------------------------------------------------------ #
    # Timing Controller Integration
    # ------------------------------------------------------------------ #

    def _initialize_timing_controller(self) -> None:
        """Initialize the appropriate timing controller based on configuration."""

        try:
            # Prefer timing_config from canonical object if available
            if self.canonical_config and self.canonical_config.timing_config:
                timing_config = dict(self.canonical_config.timing_config)
            else:
                # Fallback to strategy_params (legacy/test path)
                timing_config = self.strategy_params.get("timing_config")

            # If no timing_config provided, create a simple default based on strategy type
            if timing_config is None:
                # Simple default: portfolio strategies use time_based, signal strategies use signal_based
                if "Signal" in self.__class__.__name__:
                    timing_config = {
                        "mode": "signal_based",
                        "scan_frequency": "D",
                        "min_holding_period": 1,
                    }
                else:
                    timing_config = {
                        "mode": "time_based",
                        "rebalance_frequency": self.strategy_params.get(
                            "rebalance_frequency", "ME"
                        ),
                    }
                self.strategy_params["timing_config"] = timing_config

            # Use factory to create timing controller with proper interface integration
            from .....timing.custom_timing_registry import TimingControllerFactory

            self._timing_controller = TimingControllerFactory.create_controller(timing_config)

        except Exception as e:
            # Fallback to time-based timing if initialization fails
            logger.error(f"Failed to initialize timing controller: {e}")
            logger.info("Falling back to time-based timing with monthly frequency")
            fallback_config = {"mode": "time_based", "rebalance_frequency": "M"}

            # Use factory for fallback too
            from .....timing.custom_timing_registry import TimingControllerFactory

            self._timing_controller = TimingControllerFactory.create_controller(fallback_config)

            # Update strategy config with fallback timing config
            self.strategy_params["timing_config"] = fallback_config

    def get_timing_controller(self) -> Optional["TimingController"]:
        """Get the timing controller for this strategy."""
        if self._timing_controller is None:
            self._initialize_timing_controller()
        return self._timing_controller

    def _initialize_providers(self) -> None:
        """Initialize provider interfaces - required for strategy functionality."""
        # Import here to avoid circular imports
        from .....interfaces.universe_provider_interface import UniverseProviderFactory
        from .....interfaces.position_sizer_provider_interface import (
            PositionSizerProviderFactory,
        )
        from .....interfaces.stop_loss_provider_interface import StopLossProviderFactory
        from .....interfaces.take_profit_provider_interface import (
            TakeProfitProviderFactory,
        )

        try:
            # Initialize universe provider - REQUIRED
            self._universe_provider = UniverseProviderFactory.create_config_provider(
                self.strategy_params
            )

            # Initialize position sizer provider - REQUIRED
            self._position_sizer_provider = PositionSizerProviderFactory.get_default_provider(
                self.strategy_params
            )

            # Initialize stop loss provider - REQUIRED
            self._stop_loss_provider = StopLossProviderFactory.get_default_provider(
                self.strategy_params
            )

            # Initialize take profit provider - REQUIRED
            self._take_profit_provider = TakeProfitProviderFactory.get_default_provider(
                self.strategy_params
            )

            # Initialize risk-off signal provider - REQUIRED
            from .....risk_off_signals import RiskOffSignalProviderFactory

            self._risk_off_signal_provider = RiskOffSignalProviderFactory.get_default_provider(
                self.strategy_params
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize required provider interfaces: {e}") from e

    def _initialize_trade_direction_constraints(self) -> None:
        """
        Initialize and validate trade direction constraints.

        Sets up trade_longs/trade_shorts parameters. No legacy support.
        """
        # Get strategy parameters (might be nested)
        params_dict = self.strategy_params.get("strategy_params", self.strategy_params)

        # Set trade direction parameters with defaults
        self.trade_longs = params_dict.get("trade_longs", True)
        self.trade_shorts = params_dict.get("trade_shorts", True)

        # Validate configuration
        self._validate_trade_direction_configuration()

        # Update parameters dict to include normalized values
        params_dict["trade_longs"] = self.trade_longs
        params_dict["trade_shorts"] = self.trade_shorts

        # Log the configuration
        logger.debug(
            f"Initialized trade direction constraints: trade_longs={self.trade_longs}, trade_shorts={self.trade_shorts}"
        )

    def _validate_trade_direction_configuration(self) -> None:
        """
        Validate that trade direction configuration is sensible.

        Raises:
            TradeDirectionConfigurationError: If configuration is invalid
        """
        # Must allow at least one trade direction
        if not self.trade_longs and not self.trade_shorts:
            raise TradeDirectionConfigurationError(
                strategy_class=self.__class__.__name__,
                trade_longs=self.trade_longs,
                trade_shorts=self.trade_shorts,
                details="Both trade_longs and trade_shorts are False - strategy cannot trade!",
            )

        # Validate parameter types
        if not isinstance(self.trade_longs, bool):
            raise TradeDirectionConfigurationError(
                strategy_class=self.__class__.__name__,
                trade_longs=self.trade_longs,
                trade_shorts=self.trade_shorts,
                details=f"trade_longs must be boolean, got {type(self.trade_longs).__name__}",
            )

        if not isinstance(self.trade_shorts, bool):
            raise TradeDirectionConfigurationError(
                strategy_class=self.__class__.__name__,
                trade_longs=self.trade_longs,
                trade_shorts=self.trade_shorts,
                details=f"trade_shorts must be boolean, got {type(self.trade_shorts).__name__}",
            )

    def _validate_signal_constraints(self, signals: pd.DataFrame) -> None:
        """
        Validate that generated signals comply with trade direction constraints.

        This method enforces trade_longs/trade_shorts constraints by checking
        generated signal weights and raising an exception if violations are found.
        This ensures coding errors in strategies are caught immediately.

        Args:
            signals: DataFrame of signal weights generated by strategy

        Raises:
            TradeDirectionViolationError: If signals violate trade direction constraints
        """
        if signals.empty:
            return  # No signals to validate

        # Check for positive weights (long positions) when trade_longs=False
        if not self.trade_longs:
            positive_mask = signals > 0
            if positive_mask.any().any():  # Only raise if there are actual positive values
                positive_signals = signals[positive_mask]
                violation_count = positive_mask.sum().sum()
                raise TradeDirectionViolationError(
                    strategy_class=self.__class__.__name__,
                    trade_longs=self.trade_longs,
                    trade_shorts=self.trade_shorts,
                    violation_details=f"Generated {violation_count} positive (long) signals when trade_longs=False",
                    violated_signals=positive_signals,
                )

        # Check for negative weights (short positions) when trade_shorts=False
        if not self.trade_shorts:
            negative_mask = signals < 0
            if negative_mask.any().any():  # Only raise if there are actual negative values
                negative_signals = signals[negative_mask]
                violation_count = negative_mask.sum().sum()
                raise TradeDirectionViolationError(
                    strategy_class=self.__class__.__name__,
                    trade_longs=self.trade_longs,
                    trade_shorts=self.trade_shorts,
                    violation_details=f"Generated {violation_count} negative (short) signals when trade_shorts=False",
                    violated_signals=negative_signals,
                )

    def supports_daily_signals(self) -> bool:
        """
        Determine if strategy supports daily signals based on timing controller.
        This method uses interface-based detection instead of isinstance checks.
        """
        timing_controller = self.get_timing_controller()
        if timing_controller is None:
            return False

        # Check if timing controller has signal-based characteristics
        # Signal-based timing has scan_frequency attribute, time-based doesn't
        return hasattr(timing_controller, "scan_frequency")

    # ------------------------------------------------------------------ #
    # Hooks to override in subclasses
    # ------------------------------------------------------------------ #

    def get_risk_off_signal_generator(self) -> IRiskOffSignalGenerator:
        """Get the risk-off signal generator using the risk-off signal provider."""
        if self._risk_off_signal_generator_instance is None:
            risk_off_provider = self.get_risk_off_signal_provider()
            instance = risk_off_provider.get_risk_off_signal_generator()
            if instance is None:
                from .....risk_off_signals import NoRiskOffSignalGenerator

                instance = NoRiskOffSignalGenerator()
            self._risk_off_signal_generator_instance = instance
        
        # We ensure it's not None and return the correct type
        assert self._risk_off_signal_generator_instance is not None
        return self._risk_off_signal_generator_instance

    def get_stop_loss_handler(self) -> BaseStopLoss:
        """Get the stop loss handler using the stop loss provider."""
        if self._stop_loss_handler_instance is None:
            stop_loss_provider = self.get_stop_loss_provider()
            instance = stop_loss_provider.get_stop_loss_handler()
            if instance is None:
                instance = NoStopLoss(self.strategy_params, {})
            self._stop_loss_handler_instance = instance
            
        assert self._stop_loss_handler_instance is not None
        return self._stop_loss_handler_instance

    def get_take_profit_handler(self) -> BaseTakeProfit:
        """Get the take profit handler using the take profit provider."""
        if self._take_profit_handler_instance is None:
            take_profit_provider = self.get_take_profit_provider()
            instance = take_profit_provider.get_take_profit_handler()
            if instance is None:
                instance = NoTakeProfit(self.strategy_params, {})
            self._take_profit_handler_instance = instance
            
        assert self._take_profit_handler_instance is not None
        return self._take_profit_handler_instance

    # ------------------------------------------------------------------ #
    # Provider Interface Accessors
    # ------------------------------------------------------------------ #
    def get_universe_provider(self):
        """Get the universe provider instance."""
        if self._universe_provider is None:
            raise RuntimeError(
                "Universe provider not initialized - call _initialize_providers() first"
            )
        return self._universe_provider

    def get_position_sizer_provider(self):
        """Get the position sizer provider instance."""
        if self._position_sizer_provider is None:
            raise RuntimeError(
                "Position sizer provider not initialized - call _initialize_providers() first"
            )
        return self._position_sizer_provider

    def get_stop_loss_provider(self):
        """Get the stop loss provider instance."""
        if self._stop_loss_provider is None:
            raise RuntimeError(
                "Stop loss provider not initialized - call _initialize_providers() first"
            )
        return self._stop_loss_provider

    def get_take_profit_provider(self):
        """Get the take profit provider instance."""
        if self._take_profit_provider is None:
            raise RuntimeError(
                "Take profit provider not initialized - call _initialize_providers() first"
            )
        return self._take_profit_provider

    def get_risk_off_signal_provider(self):
        """Get the risk-off signal provider instance."""
        if self._risk_off_signal_provider is None:
            raise RuntimeError(
                "Risk-off signal provider not initialized - call _initialize_providers() first"
            )
        return self._risk_off_signal_provider

    # ------------------------------------------------------------------ #
    # Universe helper
    # ------------------------------------------------------------------ #
    def get_universe(self, global_config: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Get the universe of assets for this strategy using the universe provider.

        Args:
            global_config: Global configuration dictionary

        Returns:
            List of (ticker, weight) tuples. Weight is typically 1.0 for equal consideration.

        Raises:
            ValueError: If the universe is empty (contains no symbols).
        """
        from .....interfaces.universe_provider_interface import IUniverseWeightProvider

        universe_provider = self.get_universe_provider()
        if not isinstance(universe_provider, IUniverseWeightProvider):
            raise RuntimeError("Universe provider does not support weighted universe methods")

        try:
            return universe_provider.get_universe_with_weights(global_config)
        except Exception as e:
            # Log error before fallback to satisfy tests (generic catch for provider errors)
            logger.error(f"Universe resolution error: {e}")
            logger.info("Falling back to global universe")

        # Fallback to global_config["universe"] if provided
        fallback = global_config.get("universe", [])
        if isinstance(fallback, list) and fallback:
            # Normalize to (symbol, weight) tuples with default weight 1.0
            fallback_universe: List[Tuple[str, float]] = []
            for item in fallback:
                if isinstance(item, tuple) and len(item) == 2:
                    fallback_universe.append((str(item[0]), float(item[1])))
                else:
                    fallback_universe.append((str(item), 1.0))
            return fallback_universe

        # If no fallback present, re-raise to meet tests expecting ValueError
        raise ValueError("No universe configuration found")

    def get_universe_method_with_date(
        self, global_config: Dict[str, Any], current_date: pd.Timestamp
    ) -> List[Tuple[str, float]]:
        """
        Get the universe of assets for this strategy with date context using the universe provider.

        Args:
            global_config: Global configuration dictionary
            current_date: Current date for universe resolution

        Returns:
            List of (ticker, weight) tuples

        Raises:
            ValueError: If the universe is empty (contains no symbols).
        """
        from .....interfaces.universe_provider_interface import IUniverseWeightProvider

        universe_provider = self.get_universe_provider()
        if not isinstance(universe_provider, IUniverseWeightProvider):
            raise RuntimeError("Universe provider does not support weighted universe methods")
        return universe_provider.get_universe_with_weights_and_date(global_config, current_date)

    def get_non_universe_data_requirements(self) -> List[str]:
        """
        Returns a list of tickers that are not part of the trading universe
        but are required for the strategy's calculations.
        """
        return []

    def get_synthetic_data_requirements(self) -> bool:
        """
        Returns a boolean indicating whether the strategy requires synthetic data generation.
        """
        return True

    # ------------------------------------------------------------------ #
    # Optimiser-introspection hook                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """Names and metadata of hyper-parameters this strategy understands.

        Returns a dict where keys are param names, values are dicts with 'type' (str/int/float/etc.), 'min', 'max', 'default', 'required' (bool).
        """
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
        }

    def get_roro_signal(self):
        return None

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,  # Full historical data for universe assets
        benchmark_historical_data: pd.DataFrame,  # Full historical data for benchmark
        non_universe_historical_data: pd.DataFrame,  # Full historical data for non-universe assets
        current_date: pd.Timestamp,  # The current date for signal generation
        start_date: Optional[pd.Timestamp] = None,  # Optional start date for WFO window
        end_date: Optional[pd.Timestamp] = None,  # Optional end date for WFO window
    ) -> pd.DataFrame:  # Returns a DataFrame of weights
        """
        IMPORTANT: Method signature has evolved over time!

        LEGACY INTERFACE WARNING: Some old tests may call this method with a different signature:
        - generate_signals(prices, features, benchmark) - OLD 3-argument format
        - generate_signals(all_historical_data, benchmark_historical_data, current_date) - CURRENT format

        If you encounter TypeError about numpy array vs Timestamp comparisons, it's likely
        because legacy code is passing arguments in the wrong order, causing current_date
        to receive a pandas Series instead of a Timestamp.

        The validate_data_sufficiency() method includes defensive type checking to handle
        these cases gracefully, but it's better to update calling code to use the correct signature.
        """
        # Default implementation returns empty DataFrame - should be overridden by subclasses
        return pd.DataFrame()

    def _enforce_trade_direction_constraints(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce trade direction constraints on generated signals.

        This is the main enforcement point that validates signals and interrupts
        execution if violations are found.

        Args:
            signals: Generated signal weights from strategy

        Returns:
            Validated signals (same as input if valid)

        Raises:
            TradeDirectionViolationError: If signals violate constraints
        """
        # Validate signals comply with trade direction constraints
        self._validate_signal_constraints(signals)

        # If we get here, signals are valid
        return signals

    # ------------------------------------------------------------------ #
    # Fallback helpers for legacy integration paths
    # ------------------------------------------------------------------ #

    def _apply_signal_strategy_stop_loss(
        self,
        weights: pd.Series,
        current_date: pd.Timestamp,
        all_historical_data: pd.DataFrame,
        current_prices: pd.Series,
    ) -> pd.Series:
        """
        Default stop-loss application hook used by legacy integration tests.
        Subclasses can override; base implementation delegates to configured handler
        and returns unmodified weights on errors or when no handler is configured.
        """
        try:
            handler = self.get_stop_loss_handler()
        except Exception:
            return weights

        # No-op when using NoStopLoss
        if isinstance(handler, NoStopLoss):
            return weights

        # Ensure entry prices exist
        entry_prices_series: pd.Series
        if isinstance(getattr(self, "entry_prices", None), pd.Series):
            entry_prices_series = cast(pd.Series, self.entry_prices)
        else:
            entry_prices_series = pd.Series(index=weights.index, dtype=float)

        try:
            stop_levels = handler.calculate_stop_levels(
                current_date=current_date,
                asset_ohlc_history=all_historical_data,
                current_weights=weights,
                entry_prices=entry_prices_series,
            )
            adjusted = handler.apply_stop_loss(
                current_date=current_date,
                current_asset_prices=current_prices,
                target_weights=weights,
                entry_prices=entry_prices_series,
                stop_levels=stop_levels,
            )
            return cast(pd.Series, adjusted) if isinstance(adjusted, pd.Series) else pd.Series(adjusted)
        except Exception:
            return weights

    def _apply_signal_strategy_take_profit(
        self,
        weights: pd.Series,
        current_date: pd.Timestamp,
        all_historical_data: pd.DataFrame,
        current_prices: pd.Series,
    ) -> pd.Series:
        """
        Default take-profit application hook used by legacy integration tests.
        Base implementation delegates to configured handler and returns weights
        unchanged on errors or when no handler is configured.
        """
        try:
            handler = self.get_take_profit_handler()
        except Exception:
            return weights

        if isinstance(handler, NoTakeProfit):
            return weights

        # Ensure entry prices exist
        entry_prices_series: pd.Series
        if isinstance(getattr(self, "entry_prices", None), pd.Series):
            entry_prices_series = cast(pd.Series, self.entry_prices)
        else:
            entry_prices_series = pd.Series(index=weights.index, dtype=float)

        try:
            take_profit_levels = handler.calculate_take_profit_levels(
                current_date=current_date,
                asset_ohlc_history=all_historical_data,
                current_weights=weights,
                entry_prices=entry_prices_series,
            )
            adjusted = handler.apply_take_profit(
                current_date=current_date,
                current_asset_prices=current_prices,
                target_weights=weights,
                entry_prices=entry_prices_series,
                take_profit_levels=take_profit_levels,
            )
            return cast(pd.Series, adjusted) if isinstance(adjusted, pd.Series) else pd.Series(adjusted)
        except Exception:
            return weights

    # ------------------------------------------------------------------ #
    # Trade Direction Constraint Helpers
    # ------------------------------------------------------------------ #

    def get_trade_longs(self) -> bool:
        """Get whether strategy is allowed to trade long positions."""
        return getattr(self, "trade_longs", True)

    def get_trade_shorts(self) -> bool:
        """Get whether strategy is allowed to trade short positions."""
        return getattr(self, "trade_shorts", True)

    def is_long_short_strategy(self) -> bool:
        """Check if strategy can trade both long and short positions."""
        return self.get_trade_longs() and self.get_trade_shorts()

    def is_long_only_strategy(self) -> bool:
        """Check if strategy only trades long positions."""
        return self.get_trade_longs() and not self.get_trade_shorts()

    def is_short_only_strategy(self) -> bool:
        """Check if strategy only trades short positions."""
        return not self.get_trade_longs() and self.get_trade_shorts()

    @staticmethod
    @njit
    def run_logic(
        signals,
        w_prev,
        num_holdings,
        top_decile_fraction,
        trade_shorts,
        leverage,
        smoothing_lambda,
    ):
        if num_holdings is not None and num_holdings > 0:
            nh = int(num_holdings)
        else:
            nh = max(
                int(np.ceil(top_decile_fraction * signals.shape[0])),
                1,
            )

        winners = np.argsort(signals)[-nh:]
        losers = np.argsort(signals)[:nh]

        cand = np.zeros_like(signals)
        if winners.shape[0] > 0:
            cand[winners] = 1 / winners.shape[0]
        if trade_shorts and losers.shape[0] > 0:
            cand[losers] = -1 / losers.shape[0]

        w_new = smoothing_lambda * w_prev + (1 - smoothing_lambda) * cand

        if np.abs(cand).sum() > 1e-9:
            long_lev: float = np.sum(w_new[w_new > 0])
            short_lev: float = float(-np.sum(w_new[w_new < 0]))

            if long_lev > leverage:
                w_new[w_new > 0] *= leverage / long_lev
            if short_lev > leverage:
                w_new[w_new < 0] *= leverage / short_lev

        return w_new

    # --- Helper methods that might be used by subclasses ---

    def _calculate_benchmark_sma(
        self,
        benchmark_historical_data: pd.DataFrame,
        window: int,
        price_column: str = "Close",
    ) -> pd.Series:
        """Calculates SMA for the benchmark."""
        if benchmark_historical_data.empty:
            # Ensure benchmark_historical_data.index is valid even if empty for pd.Series constructor
            index = (
                benchmark_historical_data.index
                if benchmark_historical_data.index is not None
                else pd.Index([])
            )
            return pd.Series(dtype=float, index=index)

        # Handle MultiIndex columns properly
        if isinstance(benchmark_historical_data.columns, pd.MultiIndex):
            # Check if price_column exists in the 'Field' level of the MultiIndex
            if price_column not in benchmark_historical_data.columns.get_level_values("Field"):
                index = (
                    benchmark_historical_data.index
                    if benchmark_historical_data.index is not None
                    else pd.Index([])
                )
                return pd.Series(dtype=float, index=index)
            # Extract the price series using xs
            price_series = benchmark_historical_data.xs(price_column, level="Field", axis=1)
        else:
            # Regular DataFrame with single-level columns
            if price_column not in benchmark_historical_data.columns:
                index = (
                    benchmark_historical_data.index
                    if benchmark_historical_data.index is not None
                    else pd.Index([])
                )
                return pd.Series(dtype=float, index=index)
            price_series = benchmark_historical_data[price_column]

        # Ensure we only use data up to current_date if current_date is within the index
        # This is more of a safeguard, as input data should already be sliced.
        # However, rolling calculations might inadvertently see future data if not careful with slicing *before* this call.
        # For now, assume benchmark_historical_data is correctly pre-sliced.
        result = price_series.rolling(window=window, min_periods=max(1, window // 2)).mean()

        # Ensure we return a Series (xs might return DataFrame in some edge cases)
        if isinstance(result, pd.DataFrame):
            # If DataFrame, take first column
            result = (
                result.iloc[:, 0]
                if len(result.columns) > 0
                else pd.Series(dtype=float, index=result.index)
            )

        return cast(pd.Series, result)

    def _calculate_derisk_flags(
        self,
        benchmark_prices_at_current_date: pd.Series,
        benchmark_sma_at_current_date: pd.Series,
        derisk_periods: int,
        previous_derisk_flag: bool,
        consecutive_periods_under_sma: int,
    ) -> tuple[bool, int]:
        """
        Calculates a derisk flag for the current period based on benchmark price vs. SMA.
        This is a stateful calculation for a single point in time.

        Args:
            benchmark_prices_at_current_date: Series of benchmark prices for the current_date (should be one value).
            benchmark_sma_at_current_date: Series of benchmark SMA for the current_date (should be one value).
            derisk_periods: Number of consecutive periods benchmark must be under SMA to trigger derisking.
            previous_derisk_flag: Boolean indicating if derisking was active in the previous period.
            consecutive_periods_under_sma: Count of consecutive periods benchmark was under SMA leading up to current.

        Returns:
            A tuple: (current_derisk_flag: bool, updated_consecutive_periods_under_sma: int)
        """
        current_derisk_flag = previous_derisk_flag  # Start with previous state

        if (
            benchmark_prices_at_current_date.empty
            or benchmark_sma_at_current_date.empty
            or benchmark_prices_at_current_date.iloc[0] is pd.NA
            or benchmark_sma_at_current_date.iloc[0] is pd.NA
        ):
            # Not enough data, maintain previous state or default to not derisked if no previous state
            return previous_derisk_flag, consecutive_periods_under_sma

        price = benchmark_prices_at_current_date.iloc[0].item()
        sma = benchmark_sma_at_current_date.iloc[0].item()

        if price < sma:
            consecutive_periods_under_sma += 1
        else:  # Price is >= SMA
            consecutive_periods_under_sma = 0
            current_derisk_flag = False  # If above SMA, always turn off derisk flag

        if consecutive_periods_under_sma > derisk_periods:
            current_derisk_flag = True

        return current_derisk_flag, consecutive_periods_under_sma

    def get_minimum_required_periods(self) -> int:
        """
        Calculate the minimum number of periods (months) of historical data required
        for this strategy to function properly. This should be overridden by subclasses
        to provide strategy-specific requirements.

        Returns:
            int: Minimum number of months of historical data required
        """
        # Base implementation returns a conservative default
        return 12

    def validate_data_sufficiency(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> tuple[bool, str]:
        """
        Validates that there is sufficient historical data available for the strategy
        to perform reliable calculations as of the current_date.

        Args:
            all_historical_data: DataFrame with historical data for universe assets
            benchmark_historical_data: DataFrame with historical data for benchmark
            current_date: The date for which we're checking data sufficiency

        Returns:
            tuple[bool, str]: (is_sufficient, reason_if_not)
        """
        min_periods_required = self.get_minimum_required_periods()

        # Check universe data
        if all_historical_data.empty:
            return False, "No historical data available for universe assets"

        # Check if current_date is beyond available data
        latest_available_date = all_historical_data.index.max()
        if current_date > latest_available_date:
            return (
                False,
                f"Current date {current_date} is beyond available data (latest: {latest_available_date})",
            )

        # Filter data up to current_date
        available_data = all_historical_data[all_historical_data.index <= current_date]
        if available_data.empty:
            return False, f"No historical data available up to {current_date}"

        # Calculate available periods (assuming monthly frequency)
        earliest_date = available_data.index.min()
        available_months = (current_date.year - earliest_date.year) * 12 + (
            current_date.month - earliest_date.month
        )

        if available_months < min_periods_required:
            return (
                False,
                f"Insufficient historical data: {available_months} months available, {min_periods_required} months required",
            )

        # Check benchmark data if strategy uses SMA filtering
        sma_filter_window = self.strategy_config.get("strategy_params", {}).get("sma_filter_window")
        if sma_filter_window and sma_filter_window > 0:
            if benchmark_historical_data.empty:
                return False, "No benchmark data available but SMA filtering is enabled"

            # Check if current_date is beyond benchmark data
            benchmark_latest_date = benchmark_historical_data.index.max()
            if current_date > benchmark_latest_date:
                return (
                    False,
                    f"Current date {current_date} is beyond available benchmark data (latest: {benchmark_latest_date})",
                )

            benchmark_available = benchmark_historical_data[
                benchmark_historical_data.index <= current_date
            ]
            if benchmark_available.empty:
                return False, f"No benchmark data available up to {current_date}"

            benchmark_earliest = benchmark_available.index.min()
            benchmark_months = (current_date.year - benchmark_earliest.year) * 12 + (
                current_date.month - benchmark_earliest.month
            )

            if benchmark_months < sma_filter_window:
                return (
                    False,
                    f"Insufficient benchmark data for SMA filter: {benchmark_months} months available, {sma_filter_window} months required",
                )

        return True, ""

    def filter_universe_by_data_availability(
        self,
        all_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        min_periods_override: Optional[int] = None,
    ) -> List[str]:
        """Filter the universe to assets with sufficient data as of current_date."""
        min_periods_required = min_periods_override or self.get_minimum_required_periods()

        if all_historical_data.empty:
            return []

        # Normalize current_date to match index comparisons
        current_date = pd.Timestamp(current_date)

        # Cache per (data_identity, current_date, min_periods_required)
        cache_key = (
            id(all_historical_data),
            current_date.normalize(),
            int(min_periods_required),
        )
        cache = getattr(self, "_data_availability_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_data_availability_cache", cache)
        cached = cache.get(cache_key)
        if cached is not None:
            return list(cached)

        # Extract close price matrix (tickers as columns)
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            if "Close" not in all_historical_data.columns.get_level_values("Field"):
                return []
            close_df = all_historical_data.xs("Close", level="Field", axis=1)
        else:
            close_df = all_historical_data

        if close_df.empty:
            cache[cache_key] = []
            return []

        # If current_date is beyond the available index, clamp to last index
        if current_date > close_df.index.max():
            current_date = pd.Timestamp(close_df.index.max())

        # 1) Recency check (vectorized): any non-NaN close in last 30 days
        window_start = current_date - pd.Timedelta(days=30)
        recent_window = close_df.loc[window_start:current_date]
        if recent_window.empty:
            cache[cache_key] = []
            return []
        has_recent = recent_window.notna().any(axis=0)

        # 2) First-valid date per asset (cached across WFO windows)
        bounds_key = tuple(str(c) for c in close_df.columns)
        bounds_cache = getattr(self, "_data_availability_bounds", None)
        if bounds_cache is None:
            bounds_cache = {}
            setattr(self, "_data_availability_bounds", bounds_cache)

        index_min = pd.Timestamp(close_df.index.min())
        if index_min.tz is not None:
            index_min = index_min.tz_localize(None)

        bounds_entry = bounds_cache.get(bounds_key)
        if (
            bounds_entry is None
            or not isinstance(bounds_entry, dict)
            or "first_valid" not in bounds_entry
            or "index_min" not in bounds_entry
            or index_min < pd.Timestamp(bounds_entry["index_min"])
        ):
            first_valid = close_df.apply(lambda s: s.first_valid_index())
            bounds_cache[bounds_key] = {"index_min": index_min, "first_valid": first_valid}
        else:
            first_valid = bounds_entry["first_valid"]

        # Convert to timestamps and compute months available
        first_valid_ts = pd.to_datetime(first_valid)
        # Assets with no valid date are excluded
        first_valid_ts = first_valid_ts.where(first_valid_ts.notna(), other=pd.NaT)

        # available_months = (Ydiff*12 + Mdiff)
        available_months = (current_date.year - first_valid_ts.dt.year) * 12 + (
            current_date.month - first_valid_ts.dt.month
        )

        enough_history = available_months >= int(min_periods_required)

        # Use ensure they are both Series to satisfy type checker for bitwise &
        s_recent = has_recent if isinstance(has_recent, pd.Series) else pd.Series(has_recent, index=close_df.columns)
        s_history = enough_history if isinstance(enough_history, pd.Series) else pd.Series(enough_history, index=close_df.columns)
        
        valid_mask = s_recent & s_history
        valid_assets = [str(t) for t, ok in valid_mask.items() if bool(ok)]

        total_assets = int(valid_mask.shape[0])
        if len(valid_assets) == 0:
            if logger.isEnabledFor(logging.ERROR):
                logger.error(
                    "No assets have sufficient data for %s - all %d assets excluded",
                    current_date.strftime("%Y-%m-%d"),
                    total_assets,
                )
        else:
            excluded_count = total_assets - len(valid_assets)
            exclusion_rate = excluded_count / max(total_assets, 1)
            if exclusion_rate > 0.5 and logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "High asset exclusion rate: %d/%d assets have sufficient data for %s (%.1f%% excluded)",
                    len(valid_assets),
                    total_assets,
                    current_date.strftime("%Y-%m-%d"),
                    exclusion_rate * 100.0,
                )

        cache[cache_key] = valid_assets
        return valid_assets
