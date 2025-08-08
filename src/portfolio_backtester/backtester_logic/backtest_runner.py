"""
Backtest execution logic extracted from Backtester class.

This module implements the BacktestRunner class that handles core backtest execution
including scenario running and backtest mode orchestration.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ..api_stability import api_stable
from ..backtester_logic.strategy_logic import generate_signals, size_positions
from ..backtester_logic.portfolio_logic import calculate_portfolio_returns
from ..backtester_logic.data_manager import prepare_scenario_data

logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    Handles core backtest execution for backtesting.

    This class encapsulates all backtest execution operations that were previously
    part of the Backtester class, following the Single Responsibility Principle.
    """

    def __init__(
        self,
        global_config: Dict[str, Any],
        data_cache: Any,
        strategy_manager: Any,
        timeout_checker=None,
    ):
        """
        Initialize BacktestRunner with dependencies.

        Args:
            global_config: Global configuration dictionary
            data_cache: Data cache instance for performance optimization
            strategy_manager: StrategyManager instance for creating strategies
            timeout_checker: Optional timeout checker function
        """
        self.global_config = global_config
        self.data_cache = data_cache
        self.strategy_manager = strategy_manager
        self.timeout_checker = timeout_checker or (lambda: False)
        self.logger = logger

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("BacktestRunner initialized")

    def _get_default_optuna_storage_url(self) -> str:
        """Get the default Optuna storage URL."""
        from ..constants import DEFAULT_OPTUNA_STORAGE_URL
        return DEFAULT_OPTUNA_STORAGE_URL

    @api_stable(version="1.0", strict_params=True, strict_return=True)
    def run_scenario(
        self,
        scenario_config: Dict[str, Any],
        price_data_monthly_closes: pd.DataFrame,
        price_data_daily_ohlc: pd.DataFrame,
        rets_daily: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> Optional[pd.Series]:
        """
        Run a single backtest scenario.

        [API STABILITY NOTE]
        This method is protected by the @api_stable decorator to ensure its signature remains stable for critical workflows.

        Args:
            scenario_config: Configuration for the scenario to run
            price_data_monthly_closes: Monthly closing price data
            price_data_daily_ohlc: Daily OHLC price data
            rets_daily: Optional daily returns data
            verbose: Whether to log detailed information

        Returns:
            Portfolio returns as pandas Series, or None if scenario fails
        """
        if verbose:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Running scenario: {scenario_config['name']}")

        strategy = self.strategy_manager.get_strategy(
            scenario_config["strategy"], scenario_config["strategy_params"]
        )

        # Resolve universe tickers
        if "universe" in scenario_config:
            if isinstance(scenario_config["universe"], list):
                universe_tickers = scenario_config["universe"]
            else:
                # Always use strategy's universe provider - no direct universe resolution
                universe_tickers = [item[0] for item in strategy.get_universe(self.global_config)]
        elif "universe_config" in scenario_config:
            # Always use strategy's universe provider - no direct universe resolution
            universe_tickers = [item[0] for item in strategy.get_universe(self.global_config)]
        else:
            universe_tickers = [item[0] for item in strategy.get_universe(self.global_config)]

        # Persist resolved universe list back into scenario_config for downstream consistency
        scenario_config["universe"] = universe_tickers

        missing_cols = [t for t in universe_tickers if t not in price_data_monthly_closes.columns]
        if missing_cols:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"Tickers {missing_cols} not found in price data; they will be skipped for this run."
                )
            universe_tickers = [t for t in universe_tickers if t not in missing_cols]

        if not universe_tickers:
            logger.warning(
                "No universe tickers remain after filtering for missing data. Skipping scenario."
            )
            return None

        benchmark_ticker = self.global_config["benchmark"]

        price_data_monthly_closes, rets_daily = prepare_scenario_data(
            price_data_daily_ohlc, self.data_cache
        )

        # Pass a callable to lazily check timeout status during signal generation
        signals = generate_signals(
            strategy,
            scenario_config,
            price_data_daily_ohlc,
            universe_tickers,
            benchmark_ticker,
            self.timeout_checker,
        )

        sized_signals = size_positions(
            signals,
            scenario_config,
            price_data_monthly_closes,
            price_data_daily_ohlc,
            universe_tickers,
            benchmark_ticker,
            strategy,
        )

        # Calculate portfolio returns (no trade tracking for optimization)
        result = calculate_portfolio_returns(
            sized_signals,
            scenario_config,
            price_data_daily_ohlc,
            rets_daily,
            universe_tickers,
            self.global_config,
            track_trades=False,
            strategy=strategy,
        )

        # Handle both old and new return formats
        if isinstance(result, tuple):
            portfolio_rets_net, _ = result
        else:
            portfolio_rets_net = result

        if verbose:
            if logger.isEnabledFor(logging.DEBUG):
                scenario_name = scenario_config["name"]
                logger.debug(
                    f"Portfolio net returns calculated for {scenario_name}. First few net returns: {portfolio_rets_net.head().to_dict()}"
                )
                logger.debug(
                    f"Net returns index: {portfolio_rets_net.index.min()} to {portfolio_rets_net.index.max()}"
                )

        if not isinstance(portfolio_rets_net, (pd.Series, type(None))):
            raise TypeError("run_scenario must return a pd.Series or None")
        return portfolio_rets_net

    def run_backtest_mode(
        self,
        scenario_config: Dict[str, Any],
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
        study_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest mode using the new StrategyBacktester architecture.

        This method uses the pure StrategyBacktester directly for backtesting.

        Args:
            scenario_config: Scenario configuration
            monthly_data: Monthly price data
            daily_data: Daily OHLC data
            rets_full: Full period returns data
            study_name: Optional Optuna study name to load best parameters from

        Returns:
            Dictionary containing backtest results
        """
        from ..backtesting.strategy_backtester import StrategyBacktester
        from ..data_sources.base_data_source import BaseDataSource

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Running backtest using new architecture for scenario: {scenario_config['name']}"
            )

        # Load optimal parameters from Optuna study if specified
        if study_name:
            try:
                import optuna

                study = optuna.load_study(
                    study_name=study_name, storage=self._get_default_optuna_storage_url()
                )
                optimal_params = scenario_config["strategy_params"].copy()
                optimal_params.update(study.best_params)
                scenario_config["strategy_params"] = optimal_params
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"Loaded best parameters from study '{study_name}': {optimal_params}"
                    )
            except KeyError:
                self.logger.warning(
                    f"Study '{study_name}' not found. Using default parameters for scenario '{scenario_config['name']}'."
                )
            except Exception as e:
                self.logger.error(f"Error loading Optuna study: {e}. Using default parameters.")

        # Create a dummy data source for StrategyBacktester
        # In the real implementation, this should be the actual data source
        class DummyDataSource(BaseDataSource):
            def get_data(self, tickers, start_date, end_date):
                return daily_data

        dummy_data_source = DummyDataSource()
        strategy_backtester = StrategyBacktester(self.global_config, dummy_data_source)
        backtest_result = strategy_backtester.backtest_strategy(
            scenario_config, monthly_data, daily_data, rets_full
        )

        train_end_date = pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31"))

        result = {
            "returns": backtest_result.returns,
            "display_name": scenario_config["name"],
            "train_end_date": train_end_date,
            "trade_stats": backtest_result.trade_stats,
            "trade_history": backtest_result.trade_history,
            "performance_stats": backtest_result.performance_stats,
            "charts_data": backtest_result.charts_data,
        }

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Backtest completed for scenario: {scenario_config['name']}")

        return result

    def run_multiple_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
        study_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple backtest scenarios.

        Args:
            scenarios: List of scenario configurations
            monthly_data: Monthly price data
            daily_data: Daily OHLC data
            rets_full: Full period returns data
            study_name: Optional Optuna study name to load best parameters from

        Returns:
            Dictionary mapping scenario names to results
        """
        results = {}

        for scenario_config in scenarios:
            if self.timeout_checker():
                logger.warning("Timeout reached during scenario execution. Stopping.")
                break

            try:
                scenario_result = self.run_backtest_mode(
                    scenario_config, monthly_data, daily_data, rets_full, study_name
                )
                results[scenario_config["name"]] = scenario_result

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Completed scenario: {scenario_config['name']}")

            except Exception as e:
                logger.error(f"Error running scenario {scenario_config['name']}: {e}")
                # Continue with other scenarios
                continue

        return results

    def validate_scenario_data(
        self, scenario_config: Dict[str, Any], monthly_data: pd.DataFrame, daily_data: pd.DataFrame
    ) -> bool:
        """
        Validate that scenario has sufficient data for backtesting.

        Args:
            scenario_config: Scenario configuration
            monthly_data: Monthly price data
            daily_data: Daily OHLC data

        Returns:
            True if scenario has sufficient data, False otherwise
        """
        try:
            # Check if we have the required universe tickers
            universe_tickers = scenario_config.get("universe", [])
            if not universe_tickers:
                strategy = self.strategy_manager.get_strategy(
                    scenario_config["strategy"], scenario_config["strategy_params"]
                )
                universe_tickers = [item[0] for item in strategy.get_universe(self.global_config)]

            # Check data availability
            available_tickers = set(
                daily_data.columns.get_level_values(0)
                if isinstance(daily_data.columns, pd.MultiIndex)
                else daily_data.columns
            )
            missing_tickers = set(universe_tickers) - available_tickers

            if missing_tickers:
                logger.warning(
                    f"Missing tickers for scenario {scenario_config['name']}: {missing_tickers}"
                )

            # Require at least 50% of tickers to be available
            availability_ratio = (len(universe_tickers) - len(missing_tickers)) / len(
                universe_tickers
            )
            if availability_ratio < 0.5:
                logger.error(
                    f"Insufficient data for scenario {scenario_config['name']}: only {availability_ratio:.1%} of tickers available"
                )
                return False

            # Check for minimum data length
            if len(daily_data) < 252:  # Less than 1 year of data
                logger.warning(
                    f"Limited data for scenario {scenario_config['name']}: only {len(daily_data)} days available"
                )

            return True

        except Exception as e:
            logger.error(f"Error validating scenario data for {scenario_config['name']}: {e}")
            return False
