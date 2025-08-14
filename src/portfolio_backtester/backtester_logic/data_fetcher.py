"""
Data fetching and preprocessing logic extracted from Backtester class.

This module implements the DataFetcher class that handles all data-related operations
including ticker collection, data fetching, normalization, and preprocessing.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Handles data fetching, normalization, and preprocessing for backtesting.

    This class encapsulates all data-related operations that were previously
    scattered throughout the Backtester class, following the Single Responsibility Principle.
    """

    def __init__(self, global_config: Dict[str, Any], data_source: Any) -> None:
        """
        Initialize DataFetcher with configuration and data source.

        Args:
            global_config: Global configuration dictionary
            data_source: Data source instance for fetching market data
        """
        self.global_config = global_config
        self.data_source = data_source
        self.logger = logger

    def collect_required_tickers(
        self, scenarios_to_run: List[Dict[str, Any]], strategy_getter
    ) -> Tuple[set, bool]:
        """
        Collect all tickers needed across scenarios.

        Args:
            scenarios_to_run: List of scenario configurations
            strategy_getter: Function to get strategy instance from config

        Returns:
            Tuple of (all_tickers_set, scenario_has_universe_flag)
        """
        all_tickers: set = set()
        all_tickers.add(self.global_config["benchmark"])
        scenario_has_universe = False

        for scenario_config in scenarios_to_run:
            # Universe handling
            if "universe" in scenario_config or "universe_config" in scenario_config:
                scenario_has_universe = True
                if "universe" in scenario_config:
                    try:
                        from ..interfaces.ticker_collector import TickerCollectorFactory

                        collector = TickerCollectorFactory.create_collector(
                            scenario_config["universe"]
                        )
                        universe_tickers = collector.collect_tickers(scenario_config["universe"])
                        all_tickers.update(universe_tickers)
                    except Exception as e:
                        logger.error(f"Failed to collect universe tickers: {e}")
                elif "universe_config" in scenario_config:
                    try:
                        from ..interfaces.ticker_collector import TickerCollectorFactory

                        collector = TickerCollectorFactory.create_collector(
                            scenario_config["universe_config"]
                        )
                        universe_tickers = collector.collect_tickers(
                            scenario_config["universe_config"]
                        )
                        all_tickers.update(universe_tickers)
                    except Exception as e:
                        logger.error(f"Failed to collect universe config tickers: {e}")

            # Strategy-specific non-universe data
            strategy = strategy_getter(
                scenario_config["strategy"], scenario_config["strategy_params"]
            )
            non_universe_tickers = strategy.get_non_universe_data_requirements()
            all_tickers.update(non_universe_tickers)

        return all_tickers, scenario_has_universe

    def fetch_daily_data(self, all_tickers: set, start_date: str) -> pd.DataFrame:
        """
        Fetch daily OHLC data and perform basic cleaning.

        Args:
            all_tickers: Set of all tickers to fetch
            start_date: Start date for data fetching

        Returns:
            DataFrame with daily OHLC data

        Raises:
            ValueError: If no data is fetched from data source
        """
        daily_data = self.data_source.get_data(
            tickers=list(all_tickers),
            start_date=start_date,
            end_date=self.global_config["end_date"],
        )
        if daily_data is None or daily_data.empty:
            logger.critical("No data fetched from data source. Aborting backtest run.")
            raise ValueError("daily_data is None after data source fetch. Cannot proceed.")
        daily_data = daily_data.dropna(how="all")
        # Ensure DataFrame return type
        return daily_data if isinstance(daily_data, pd.DataFrame) else pd.DataFrame(daily_data)

    def normalize_ohlc_format(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize OHLC format to expected shape.

        Args:
            daily_data: Raw daily data from data source

        Returns:
            DataFrame with normalized OHLC format
        """
        if (
            isinstance(daily_data.columns, pd.MultiIndex)
            and daily_data.columns.names
            and daily_data.columns.names[0] != "Ticker"
        ):
            daily_data_std_format = daily_data.stack(level=1).unstack(level=0)
        else:
            daily_data_std_format = daily_data

        if isinstance(daily_data_std_format, pd.Series):
            return daily_data_std_format.to_frame()
        return daily_data_std_format

    def extract_close_prices(self, daily_ohlc: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Close prices from daily OHLC in either MultiIndex or single-index form.

        Args:
            daily_ohlc: DataFrame with OHLC data

        Returns:
            DataFrame with close prices only

        Raises:
            ValueError: If close prices cannot be extracted
        """
        if daily_ohlc is None:
            raise ValueError("Daily OHLC data is not available.")

        # Start with DataFrame; convert Series to DataFrame to satisfy type hints
        daily_closes_df: Optional[pd.DataFrame] = None

        if isinstance(
            daily_ohlc.columns, pd.MultiIndex
        ) and "Close" in daily_ohlc.columns.get_level_values(1):
            extracted = daily_ohlc.xs("Close", level="Field", axis=1)
            daily_closes_df = (
                extracted.to_frame() if isinstance(extracted, pd.Series) else extracted
            )
        elif not isinstance(daily_ohlc.columns, pd.MultiIndex):
            # Ensure DataFrame type without constructing ambiguous mixed subclass.
            # Avoid isinstance checks that trigger mypy MRO complaints by using attribute-based guard.
            if getattr(daily_ohlc, "ndim", 2) == 1:
                # Treat as Series-like
                daily_closes_df = pd.DataFrame(
                    daily_ohlc
                )  # to_frame() equivalent without MRO confusion
            else:
                daily_closes_df = pd.DataFrame(daily_ohlc)
        else:
            # Fallback attempt with last level named 'Close'
            if "Close" in daily_ohlc.columns.get_level_values(-1):
                extracted = daily_ohlc.xs("Close", level=-1, axis=1)
                daily_closes_df = (
                    extracted.to_frame() if isinstance(extracted, pd.Series) else extracted
                )
            else:
                raise ValueError(
                    "Could not reliably extract 'Close' prices from daily OHLC due to unrecognized column structure."
                )

        if daily_closes_df is None or daily_closes_df.empty:
            raise ValueError("Daily close prices could not be extracted or are empty.")

        return daily_closes_df

    def determine_optimal_start_date(self, all_tickers: set) -> str:
        """
        Determine optimal start date based on data availability rules:

        1. For single-stock universes: use the earliest available data date for that stock
        2. For multi-stock universes: use the date where over 50% of universe members have data available

        Args:
            all_tickers: Set of all tickers including benchmark

        Returns:
            Optimal start date as string
        """
        # Remove benchmark from consideration for universe size calculation
        benchmark = self.global_config["benchmark"]
        universe_tickers = all_tickers - {benchmark}

        # Default fallback
        default_start_date = self.global_config["start_date"]

        if len(universe_tickers) == 0:
            logger.warning("No universe tickers found, using default start date")
            return str(default_start_date)

        if len(universe_tickers) == 1:
            # Rule 1: Single-symbol universe
            single_symbol = next(iter(universe_tickers))
            logger.info(
                f"Single symbol universe detected: {single_symbol}. Using earliest available data date."
            )
            try:
                # Get minimal data to find the earliest available date
                test_data = self.data_source.get_data(
                    tickers=[single_symbol],
                    start_date="1990-01-01",  # Very early date to get all available data
                    end_date=self.global_config["end_date"],
                )
                if test_data is not None and not test_data.empty:
                    first_idx = test_data.index[0]
                    ts = pd.to_datetime(first_idx)
                    logger.info(f"Using earliest available data date for {single_symbol}: {ts}")
                    return str(ts.strftime("%Y-%m-%d"))
                else:
                    logger.warning(f"No data found for {single_symbol}, using default start date")
                    return str(default_start_date)
            except Exception as e:
                logger.warning(
                    f"Could not determine earliest data date for {single_symbol}, using configured start_date: {e}"
                )
                return str(default_start_date)

        else:
            # Rule 2: Multi-stock universe - find date where >50% of tickers have data
            logger.info(
                f"Multi-ticker universe detected: {len(universe_tickers)} tickers. Finding date where >50% have data available."
            )

            try:
                # Get data for all universe tickers to determine availability
                test_data = self.data_source.get_data(
                    tickers=list(universe_tickers),
                    start_date="1990-01-01",  # Very early date to get all available data
                    end_date=self.global_config["end_date"],
                )

                if test_data is None or test_data.empty:
                    logger.warning(
                        "No data found for any universe tickers, using default start date"
                    )
                    return str(default_start_date)

                # Count data availability by date
                # Handle both MultiIndex and single-level columns
                if isinstance(test_data.columns, pd.MultiIndex):
                    # Extract close price columns for availability check
                    close_data = test_data.xs("Close", level="Field", axis=1, drop_level=False)
                    ticker_columns = close_data.columns.get_level_values("Ticker").unique()
                else:
                    close_data = test_data
                    ticker_columns = test_data.columns

                # Filter to only universe tickers that actually have columns in the data
                available_tickers = [t for t in universe_tickers if t in ticker_columns]

                if not available_tickers:
                    logger.warning(
                        "No universe tickers found in data columns, using default start date"
                    )
                    return str(default_start_date)

                logger.info(
                    f"Checking data availability for {len(available_tickers)} tickers: {available_tickers}"
                )

                # Calculate threshold (greater than 50%)
                threshold = len(available_tickers) / 2.0

                # Count non-null values for each date
                if isinstance(test_data.columns, pd.MultiIndex):
                    # For MultiIndex, select the close columns for available tickers
                    availability_data = pd.DataFrame()
                    for ticker in available_tickers:
                        if (ticker, "Close") in test_data.columns:
                            availability_data[ticker] = test_data[(ticker, "Close")]
                        else:
                            # Try other field names if Close is not available
                            ticker_cols = [col for col in test_data.columns if col[0] == ticker]
                            if ticker_cols:
                                availability_data[ticker] = test_data[ticker_cols[0]]
                else:
                    # For single-level columns
                    availability_data = test_data[available_tickers]

                # Count non-null values per date
                daily_availability_count = availability_data.notna().sum(axis=1)

                # Find first date where more than 50% of tickers have data
                qualifying_dates = daily_availability_count[daily_availability_count > threshold]

                if not qualifying_dates.empty:
                    # Use first index value robustly as Timestamp/string date
                    first_idx = qualifying_dates.index[0]
                    available_count = int(qualifying_dates.iloc[0])
                    logger.info(
                        f"Using optimal start date {first_idx} where {available_count}/{len(available_tickers)} tickers ({available_count / len(available_tickers) * 100:.1f}%) have data available"
                    )
                    # Convert to pandas Timestamp safely, then to string
                    ts = pd.to_datetime(first_idx)
                    return str(ts.strftime("%Y-%m-%d"))
                else:
                    logger.warning(
                        "Could not find date where >50% of tickers have data available, using default start date"
                    )
                    return str(default_start_date)

            except Exception as e:
                logger.warning(
                    f"Error determining optimal start date for multi-ticker universe: {e}, using default start date"
                )
                return str(default_start_date)

    def prepare_data_for_backtesting(
        self,
        scenarios_to_run: List[Dict[str, Any]],
        strategy_getter,
        scenario_has_universe: Optional[bool] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete data preparation pipeline for backtesting.

        This method orchestrates the entire data fetching and preprocessing workflow.

        Args:
            scenarios_to_run: List of scenario configurations
            strategy_getter: Function to get strategy instance from config
            scenario_has_universe: Optional flag if universe detection already done

        Returns:
            Tuple of (daily_ohlc_data, monthly_closes_data, daily_closes_data)
        """
        # Collect required tickers
        all_tickers, has_universe = self.collect_required_tickers(scenarios_to_run, strategy_getter)

        # Use provided universe flag if available
        if scenario_has_universe is not None:
            has_universe = scenario_has_universe

        # Add global universe if no scenario-specific universe found
        if not has_universe:
            logger.info("No scenario-specific universe found, using global universe")
            all_tickers.update(self.global_config.get("universe", []))
        else:
            logger.info(
                f"Using scenario-specific universe with {len(all_tickers)} tickers (including benchmark)"
            )

        # Determine optimal start date and fetch data
        start_date = self.determine_optimal_start_date(all_tickers)
        daily_data = self.fetch_daily_data(all_tickers, start_date)
        daily_ohlc = self.normalize_ohlc_format(daily_data)

        # Extract close prices and create monthly data
        daily_closes = self.extract_close_prices(daily_ohlc)
        monthly_closes = daily_closes.resample("BME").last()

        # Ensure DataFrame types
        monthly_data = (
            monthly_closes
            if isinstance(monthly_closes, pd.DataFrame)
            else pd.DataFrame(monthly_closes)
        )

        return daily_ohlc, monthly_data, daily_closes
