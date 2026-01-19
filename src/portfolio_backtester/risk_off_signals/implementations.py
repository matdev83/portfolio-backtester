"""
Risk-off Signal Generator Implementations

Provides concrete implementations of the IRiskOffSignalGenerator interface.
Includes both production-ready and testing implementations following SOLID principles.
"""

from typing import Any, Dict, List
import logging

import pandas as pd
import numpy as np

from .interface import IRiskOffSignalGenerator

logger = logging.getLogger(__name__)


class NoRiskOffSignalGenerator(IRiskOffSignalGenerator):
    """
    Default risk-off signal generator that never signals risk-off conditions.

    This implementation follows the principle of least surprise - it never
    triggers risk-off signals, allowing strategies to operate normally.
    This is the appropriate default behavior for most strategies.

    Design Pattern: Null Object Pattern
    - Provides a default "do nothing" implementation
    - Eliminates the need for null checks in client code
    - Always returns False (risk-on conditions)
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Initialize the no-risk-off signal generator.

        Args:
            config: Configuration parameters (unused but accepted for interface compatibility)
        """
        self._config = config or {}

    def generate_risk_off_signal(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> bool:
        """
        Generate risk-off signal - always returns False (risk-on).

        This implementation never signals risk-off conditions, allowing
        strategies to operate without risk regime interference.

        Returns:
            bool: Always False (risk-on conditions)
        """
        return False  # Never signal risk-off

    def get_configuration(self) -> Dict[str, Any]:
        """Get configuration - returns empty dict as no parameters are used."""
        return dict(self._config)

    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """Validate configuration - always valid as no parameters are required."""
        return (True, "")

    def get_required_data_columns(self) -> List[str]:
        """Get required data columns - minimal requirements as signal is not data-dependent."""
        return []  # No data required for null implementation

    def get_minimum_data_periods(self) -> int:
        """Get minimum data periods - zero as no historical data is analyzed."""
        return 0  # No historical data required

    def get_signal_description(self) -> str:
        """Get description of this signal generator."""
        return "No Risk-off Signal Generator: Never signals risk-off conditions (always risk-on)"


class DummyRiskOffSignalGenerator(IRiskOffSignalGenerator):
    """
    Dummy risk-off signal generator for testing and development purposes.

    This implementation provides configurable hardcoded risk-off windows for testing.
    It should NOT be used in production - it's designed for unit testing, integration
    testing, and strategy development where predictable risk-off periods are needed.

    Configuration Parameters:
    - risk_off_windows: List of (start_date, end_date) tuples for risk-off periods
    - default_risk_state: Default risk state when not in specified windows ('on' or 'off')

    Design Pattern: Test Double (specifically a Stub)
    - Provides predictable, controlled behavior for testing
    - Configurable via constructor parameters
    - Should be replaced with real implementation in production
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Initialize dummy risk-off signal generator.

        Args:
            config: Configuration containing 'risk_off_windows' and 'default_risk_state'
        """
        self._config = config or {}

        # Parse risk-off windows from config
        self._risk_off_windows = self._parse_risk_off_windows()

        # Default risk state when not in specified windows
        self._default_risk_state = self._config.get("default_risk_state", "on")

        if self._default_risk_state not in ["on", "off"]:
            raise ValueError(
                f"Invalid default_risk_state: {self._default_risk_state}. Must be 'on' or 'off'"
            )

    def _parse_risk_off_windows(self) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
        """Parse risk-off windows from configuration."""
        windows_config = self._config.get("risk_off_windows", [])

        # If no windows specified, use some default test windows
        if not windows_config:
            return [
                (
                    pd.Timestamp("2008-09-01"),
                    pd.Timestamp("2009-03-31"),
                ),  # Financial crisis
                (pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-30")),  # COVID crash
            ]

        parsed_windows = []
        for window in windows_config:
            if isinstance(window, (list, tuple)) and len(window) == 2:
                start_date = pd.Timestamp(window[0])
                end_date = pd.Timestamp(window[1])
                parsed_windows.append((start_date, end_date))
            else:
                logger.warning(
                    f"Invalid risk-off window format: {window}. Expected (start_date, end_date) tuple."
                )

        return parsed_windows

    def generate_risk_off_signal(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> bool:
        """
        Generate risk-off signal based on configured test windows.

        Args:
            all_historical_data: Universe data (unused in dummy implementation)
            benchmark_historical_data: Benchmark data (unused in dummy implementation)
            non_universe_historical_data: Non-universe data (unused)
            current_date: Date to generate signal for

        Returns:
            bool: True if current_date falls in configured risk-off window,
                  False otherwise (unless default_risk_state is 'off')
        """
        # Check if current date falls within any configured risk-off window
        for start_date, end_date in self._risk_off_windows:
            if start_date <= current_date <= end_date:
                return True  # Risk-off signal

        # Return default state when not in risk-off windows
        return bool(self._default_risk_state == "off")

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration parameters."""
        config = dict(self._config)
        # Add parsed windows for inspection
        config["_parsed_risk_off_windows"] = [
            (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            for start, end in self._risk_off_windows
        ]
        return config

    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """Validate configuration parameters."""
        # Validate default_risk_state
        default_state = config.get("default_risk_state", "on")
        if default_state not in ["on", "off"]:
            return (
                False,
                f"Invalid default_risk_state: {default_state}. Must be 'on' or 'off'",
            )

        # Validate risk_off_windows format
        windows = config.get("risk_off_windows", [])
        if not isinstance(windows, list):
            return (
                False,
                "risk_off_windows must be a list of (start_date, end_date) tuples",
            )

        for i, window in enumerate(windows):
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                return (
                    False,
                    f"Window {i} must be a (start_date, end_date) tuple, got: {window}",
                )

            try:
                start_date = pd.Timestamp(window[0])
                end_date = pd.Timestamp(window[1])
                if start_date >= end_date:
                    return (False, f"Window {i}: start_date must be before end_date")
            except Exception as e:
                return (False, f"Window {i}: Invalid date format - {e}")

        return (True, "")

    def get_required_data_columns(self) -> List[str]:
        """Get required data columns - minimal as dummy doesn't analyze data."""
        return []  # Dummy implementation doesn't require data

    def get_minimum_data_periods(self) -> int:
        """Get minimum data periods - zero as dummy doesn't analyze data."""
        return 0  # No historical analysis in dummy implementation

    def get_signal_description(self) -> str:
        """Get description of this signal generator."""
        num_windows = len(self._risk_off_windows)
        return (
            f"Dummy Risk-off Signal Generator: Uses {num_windows} hardcoded test windows "
            f"(default: {self._default_risk_state}). FOR TESTING ONLY."
        )


def _extract_benchmark_close_series(
    benchmark_historical_data: pd.DataFrame,
    current_date: pd.Timestamp,
) -> pd.Series:
    """Extract a benchmark close-price series up to `current_date`.

    The framework may pass benchmark data as:
    - Single-column DataFrame (e.g., just prices)
    - OHLCV DataFrame (expects 'Close' when present)

    Args:
        benchmark_historical_data: Benchmark historical data.
        current_date: Current evaluation date.

    Returns:
        Close price series indexed by date (may be empty if unavailable).
    """
    if benchmark_historical_data is None or benchmark_historical_data.empty:
        return pd.Series(dtype=float)

    df = benchmark_historical_data

    if isinstance(df, pd.Series):
        ser = df
    else:
        if "Close" in df.columns:
            ser = df["Close"]
        elif "Adj Close" in df.columns:
            ser = df["Adj Close"]
        else:
            ser = df.iloc[:, 0]

    ser = ser.loc[ser.index <= current_date].dropna()
    ser.name = "Close"
    return ser


class BenchmarkSmaRiskOffSignalGenerator(IRiskOffSignalGenerator):
    """Risk-off when benchmark is below its SMA.

    Configuration:
        - sma_window_days: SMA window in trading days (default: 200).
        - min_periods: Minimum observations required (default: sma_window_days).

    Signal:
        True  => risk-off (go to cash)
        False => risk-on
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self._config = config or {}

        self._sma_window_days = int(self._config.get("sma_window_days", 200))
        self._min_periods = int(self._config.get("min_periods", self._sma_window_days))

        if self._sma_window_days <= 1:
            raise ValueError("sma_window_days must be > 1")
        if self._min_periods <= 1:
            raise ValueError("min_periods must be > 1")

    def generate_risk_off_signal(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> bool:
        prices = _extract_benchmark_close_series(benchmark_historical_data, current_date)
        if prices.empty or prices.shape[0] < self._min_periods:
            return False

        sma = prices.rolling(window=self._sma_window_days, min_periods=self._min_periods).mean()
        price_now = float(prices.iloc[-1])
        sma_now = float(sma.iloc[-1]) if not np.isnan(sma.iloc[-1]) else np.nan
        if np.isnan(sma_now):
            return False
        return bool(price_now < sma_now)

    def get_configuration(self) -> Dict[str, Any]:
        return dict(self._config)

    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, str]:
        try:
            window = int(config.get("sma_window_days", 200))
            min_periods = int(config.get("min_periods", window))
        except Exception as e:
            return (False, f"Invalid integer parameter: {e}")

        if window <= 1:
            return (False, "sma_window_days must be > 1")
        if min_periods <= 1:
            return (False, "min_periods must be > 1")
        if min_periods > window:
            return (False, "min_periods must be <= sma_window_days")
        return (True, "")

    def get_required_data_columns(self) -> List[str]:
        return []

    def get_minimum_data_periods(self) -> int:
        return self._min_periods

    def get_signal_description(self) -> str:
        return f"Benchmark SMA risk-off: Close < SMA({self._sma_window_days})"


class BenchmarkMonthlySmaRiskOffSignalGenerator(IRiskOffSignalGenerator):
    """Risk-off when benchmark is below its monthly SMA (e.g., 10-month SMA).

    This implements the classic "10-month SMA" market timing rule using monthly closes.
    It is evaluated at strategy rebalancing dates; for month-end rebalancing the signal
    uses the last available close in each calendar month (i.e., last trading day).

    Configuration:
        - sma_window_months: SMA window in months (default: 10).
        - min_periods: Minimum observations required (default: sma_window_months).

    Signal:
        True  => risk-off (go to cash)
        False => risk-on
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self._config = config or {}

        self._sma_window_months = int(self._config.get("sma_window_months", 10))
        self._min_periods = int(self._config.get("min_periods", self._sma_window_months))

        if self._sma_window_months <= 1:
            raise ValueError("sma_window_months must be > 1")
        if self._min_periods <= 1:
            raise ValueError("min_periods must be > 1")
        if self._min_periods > self._sma_window_months:
            raise ValueError("min_periods must be <= sma_window_months")

    @staticmethod
    def _monthly_last(prices: pd.Series) -> pd.Series:
        """Return one price per month using the last observed value in each month.

        The output index is the actual last observed date in each month (not a synthetic
        month-end timestamp), which aligns with business-month-end rebalancing schedules.
        """
        if prices.empty:
            return prices

        ser = prices.sort_index()
        periods = pd.PeriodIndex(pd.DatetimeIndex(ser.index), freq="M")
        return ser.groupby(periods).tail(1)

    def generate_risk_off_signal(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> bool:
        prices = _extract_benchmark_close_series(benchmark_historical_data, current_date)
        monthly_prices = self._monthly_last(prices)

        if monthly_prices.empty or monthly_prices.shape[0] < self._min_periods:
            return False

        sma = monthly_prices.rolling(
            window=self._sma_window_months,
            min_periods=self._min_periods,
        ).mean()

        price_now = float(monthly_prices.iloc[-1])
        sma_now = float(sma.iloc[-1]) if not np.isnan(sma.iloc[-1]) else np.nan
        if np.isnan(sma_now):
            return False
        return bool(price_now < sma_now)

    def get_configuration(self) -> Dict[str, Any]:
        return dict(self._config)

    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, str]:
        try:
            window = int(config.get("sma_window_months", 10))
            min_periods = int(config.get("min_periods", window))
        except Exception as e:
            return (False, f"Invalid integer parameter: {e}")

        if window <= 1:
            return (False, "sma_window_months must be > 1")
        if min_periods <= 1:
            return (False, "min_periods must be > 1")
        if min_periods > window:
            return (False, "min_periods must be <= sma_window_months")
        return (True, "")

    def get_required_data_columns(self) -> List[str]:
        return []

    def get_minimum_data_periods(self) -> int:
        return self._min_periods

    def get_signal_description(self) -> str:
        return f"Benchmark monthly SMA risk-off: Close < SMA({self._sma_window_months}m)"


class BenchmarkEmaCrossoverRiskOffSignalGenerator(IRiskOffSignalGenerator):
    """Risk-off when benchmark fast EMA is below slow EMA.

    Configuration:
        - fast_ema_days: fast EMA span (default: 50)
        - slow_ema_days: slow EMA span (default: 200)
        - min_periods: minimum data points (default: slow_ema_days)
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._fast_ema_days = int(self._config.get("fast_ema_days", 50))
        self._slow_ema_days = int(self._config.get("slow_ema_days", 200))
        self._min_periods = int(self._config.get("min_periods", self._slow_ema_days))

        if self._fast_ema_days <= 1 or self._slow_ema_days <= 1:
            raise ValueError("fast_ema_days and slow_ema_days must be > 1")
        if self._fast_ema_days >= self._slow_ema_days:
            raise ValueError("fast_ema_days must be < slow_ema_days")
        if self._min_periods <= 1:
            raise ValueError("min_periods must be > 1")

    def generate_risk_off_signal(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> bool:
        prices = _extract_benchmark_close_series(benchmark_historical_data, current_date)
        if prices.empty or prices.shape[0] < self._min_periods:
            return False

        fast = prices.ewm(
            span=self._fast_ema_days, adjust=False, min_periods=self._min_periods
        ).mean()
        slow = prices.ewm(
            span=self._slow_ema_days, adjust=False, min_periods=self._min_periods
        ).mean()

        fast_now = float(fast.iloc[-1]) if not np.isnan(fast.iloc[-1]) else np.nan
        slow_now = float(slow.iloc[-1]) if not np.isnan(slow.iloc[-1]) else np.nan
        if np.isnan(fast_now) or np.isnan(slow_now):
            return False
        return bool(fast_now < slow_now)

    def get_configuration(self) -> Dict[str, Any]:
        return dict(self._config)

    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, str]:
        try:
            fast_days = int(config.get("fast_ema_days", 50))
            slow_days = int(config.get("slow_ema_days", 200))
            min_periods = int(config.get("min_periods", slow_days))
        except Exception as e:
            return (False, f"Invalid integer parameter: {e}")

        if fast_days <= 1 or slow_days <= 1:
            return (False, "fast_ema_days and slow_ema_days must be > 1")
        if fast_days >= slow_days:
            return (False, "fast_ema_days must be < slow_ema_days")
        if min_periods <= 1:
            return (False, "min_periods must be > 1")
        return (True, "")

    def get_required_data_columns(self) -> List[str]:
        return []

    def get_minimum_data_periods(self) -> int:
        return self._min_periods

    def get_signal_description(self) -> str:
        return f"Benchmark EMA crossover risk-off: EMA({self._fast_ema_days}) < EMA({self._slow_ema_days})"


class BenchmarkDrawdownVolRiskOffSignalGenerator(IRiskOffSignalGenerator):
    """Risk-off during 'panic' conditions: deep drawdown + elevated volatility.

    Configuration:
        - drawdown_lookback_days: lookback for peak-to-trough drawdown (default: 126)
        - drawdown_threshold: drawdown trigger (negative float, default: -0.12)
        - vol_lookback_days: lookback for realized volatility (default: 63)
        - vol_threshold_annual: annualized vol trigger (default: 0.25)
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self._config = config or {}

        self._drawdown_lookback_days = int(self._config.get("drawdown_lookback_days", 126))
        self._drawdown_threshold = float(self._config.get("drawdown_threshold", -0.12))
        self._vol_lookback_days = int(self._config.get("vol_lookback_days", 63))
        self._vol_threshold_annual = float(self._config.get("vol_threshold_annual", 0.25))

        if self._drawdown_lookback_days <= 1 or self._vol_lookback_days <= 1:
            raise ValueError("drawdown_lookback_days and vol_lookback_days must be > 1")
        if self._drawdown_threshold >= 0.0:
            raise ValueError("drawdown_threshold must be negative")
        if self._vol_threshold_annual <= 0.0:
            raise ValueError("vol_threshold_annual must be > 0")

    def generate_risk_off_signal(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> bool:
        prices = _extract_benchmark_close_series(benchmark_historical_data, current_date)
        min_needed = max(self._drawdown_lookback_days, self._vol_lookback_days) + 1
        if prices.empty or prices.shape[0] < min_needed:
            return False

        window_prices = prices.tail(self._drawdown_lookback_days)
        peak = float(window_prices.max())
        last = float(window_prices.iloc[-1])
        drawdown = (last / peak) - 1.0 if peak > 0 else 0.0

        returns = prices.pct_change(fill_method=None).dropna()
        vol = float(returns.tail(self._vol_lookback_days).std() * np.sqrt(252))

        return bool(drawdown <= self._drawdown_threshold and vol >= self._vol_threshold_annual)

    def get_configuration(self) -> Dict[str, Any]:
        return dict(self._config)

    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, str]:
        try:
            dd_lb = int(config.get("drawdown_lookback_days", 126))
            dd_th = float(config.get("drawdown_threshold", -0.12))
            vol_lb = int(config.get("vol_lookback_days", 63))
            vol_th = float(config.get("vol_threshold_annual", 0.25))
        except Exception as e:
            return (False, f"Invalid parameter: {e}")

        if dd_lb <= 1 or vol_lb <= 1:
            return (False, "drawdown_lookback_days and vol_lookback_days must be > 1")
        if dd_th >= 0.0:
            return (False, "drawdown_threshold must be negative")
        if vol_th <= 0.0:
            return (False, "vol_threshold_annual must be > 0")
        return (True, "")

    def get_required_data_columns(self) -> List[str]:
        return []

    def get_minimum_data_periods(self) -> int:
        return max(self._drawdown_lookback_days, self._vol_lookback_days) + 1

    def get_signal_description(self) -> str:
        return (
            "Benchmark drawdown+vol panic risk-off: "
            f"DD({self._drawdown_lookback_days}) <= {self._drawdown_threshold:.2%} and "
            f"Vol({self._vol_lookback_days}) >= {self._vol_threshold_annual:.2%}"
        )


#         pass
#
#     def supports_non_universe_data(self) -> bool:
#         return True  # Requires VIX data
#
#     def get_required_data_columns(self) -> List[str]:
#         return ['Close']  # VIX close price
#
#     def get_minimum_data_periods(self) -> int:
#         return self._lookback_days
