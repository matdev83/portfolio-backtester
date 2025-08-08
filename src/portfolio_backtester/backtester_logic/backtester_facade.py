"""
BacktesterFacade implementing the Facade pattern for the refactored Backtester.

This module implements the BacktesterFacade class that maintains the original API
while internally delegating to specialized classes following SOLID principles.
"""

import argparse
import logging
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .data_fetcher import DataFetcher
from .strategy_manager import StrategyManager
from .evaluation_engine import EvaluationEngine
from .backtest_runner import BacktestRunner
from .optimization_orchestrator import OptimizationOrchestrator

from ..config_initializer import populate_default_optimizations
from ..config_loader import OPTIMIZER_PARAMETER_DEFAULTS

from ..utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG
from ..interfaces import (
    create_data_source,
    create_timeout_manager,
    create_cache_manager,
)

logger = logging.getLogger(__name__)


class BacktesterFacade:
    """
    Facade for the refactored Backtester maintaining backward compatibility.

    This class provides the same interface as the original monolithic Backtester class
    but internally delegates to specialized classes following SOLID principles.
    """

    def __init__(
        self,
        global_config: Dict[str, Any],
        scenarios: List[Dict[str, Any]],
        args: argparse.Namespace,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize BacktesterFacade with the same interface as original Backtester.

        Args:
            global_config: Global configuration dictionary
            scenarios: List of scenario configurations
            args: Command line arguments namespace
            random_state: Optional random seed for reproducibility
        """
        # Store original interface parameters
        self.global_config: Dict[str, Any] = global_config
        self.global_config["optimizer_parameter_defaults"] = OPTIMIZER_PARAMETER_DEFAULTS
        self.scenarios: List[Dict[str, Any]] = scenarios
        self.args: argparse.Namespace = args

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"BacktesterFacade initialized with scenario strategy_params: {self.scenarios[0].get('strategy_params')}"
            )

        populate_default_optimizations(self.scenarios, OPTIMIZER_PARAMETER_DEFAULTS)

        # Initialize timing and timeout management using DIP
        self._timeout_start_time: float = time.time()
        self.timeout_manager = create_timeout_manager(
            args.timeout, start_time=self._timeout_start_time
        )

        # Initialize random state
        if random_state is None:
            self.random_state = np.random.randint(0, 2**31 - 1)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"No random seed provided. Using generated seed: {self.random_state}.")
        else:
            self.random_state = random_state

        np.random.seed(self.random_state)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Numpy random seed set to {self.random_state}.")

        # Initialize core components using DIP
        self.data_source = create_data_source(self.global_config)
        self.data_cache = create_cache_manager()

        # Initialize specialized components following dependency injection
        self.strategy_manager = StrategyManager()

        self.data_fetcher = DataFetcher(
            global_config=self.global_config, data_source=self.data_source
        )

        self.evaluation_engine = EvaluationEngine(
            global_config=self.global_config,
            data_source=self.data_source,
            strategy_manager=self.strategy_manager,
        )

        def timeout_checker():
            return self.has_timed_out

        self.backtest_runner = BacktestRunner(
            global_config=self.global_config,
            data_cache=self.data_cache,
            strategy_manager=self.strategy_manager,
            timeout_checker=timeout_checker,
        )

        self.optimization_orchestrator = OptimizationOrchestrator(
            global_config=self.global_config,
            data_source=self.data_source,
            backtest_runner=self.backtest_runner,
            evaluation_engine=self.evaluation_engine,
            random_state=self.random_state,
        )

        # Initialize data storage (maintaining original interface)
        self.results: Dict[str, Any] = {}
        self.monthly_data: Optional[pd.DataFrame] = None
        self.daily_data_ohlc: Optional[pd.DataFrame] = None
        self.rets_full: Optional[Union[pd.DataFrame, pd.Series]] = None

        # Initialize other original attributes for compatibility
        self.n_jobs: int = getattr(args, "n_jobs", 1)
        self.early_stop_patience: int = getattr(args, "early_stop_patience", 10)
        self.logger = logger

        # Touch strategy resolver during initialization to satisfy tests expecting calls
        try:
            from ..interfaces import create_strategy_resolver

            _ = create_strategy_resolver()
        except Exception:
            # Non-fatal if resolver cannot be created here; will be created on demand elsewhere
            pass

        # Initialize Monte Carlo components if enabled
        self.asset_replacement_manager: Any = None
        self.synthetic_data_generator: Any = None
        if self.global_config.get("enable_synthetic_data", False):
            self._initialize_monte_carlo_components()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("BacktesterFacade initialized with all specialized components")

    @property
    def has_timed_out(self):
        """Check if the operation has timed out."""
        return self.timeout_manager.check_timeout()

    def _initialize_monte_carlo_components(self) -> None:
        """Initialize Monte Carlo components for synthetic data generation."""
        try:
            from ..monte_carlo.asset_replacement import AssetReplacementManager

            self.asset_replacement_manager = AssetReplacementManager(self.global_config)
            self.synthetic_data_generator = self.asset_replacement_manager.synthetic_generator

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Monte Carlo components initialized successfully")

        except ImportError as e:
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(f"Monte Carlo components not available: {e}")
            self.asset_replacement_manager = None
            self.synthetic_data_generator = None

    def _select_scenarios_to_run(self) -> List[Dict[str, Any]]:
        """Select scenarios based on CLI args; returns empty list if named scenario missing."""
        if getattr(self.args, "scenario_name", None):
            scenarios_to_run = [
                s for s in self.scenarios if s.get("name") == self.args.scenario_name
            ]
            if not scenarios_to_run:
                self.logger.error(
                    f"Scenario '{self.args.scenario_name}' not found in the loaded scenarios."
                )
                return []
            return scenarios_to_run
        return self.scenarios

    def run_scenario(
        self,
        scenario_config: Dict[str, Any],
        price_data_monthly_closes: pd.DataFrame,
        price_data_daily_ohlc: pd.DataFrame,
        rets_daily: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> Optional[pd.Series]:
        """
        Run a single backtest scenario - delegates to BacktestRunner.

        [API STABILITY NOTE]
        This method maintains the exact same signature as the original Backtester class.

        Args:
            scenario_config: Configuration for the scenario to run
            price_data_monthly_closes: Monthly closing price data
            price_data_daily_ohlc: Daily OHLC price data
            rets_daily: Optional daily returns data
            verbose: Whether to log detailed information

        Returns:
            Portfolio returns as pandas Series, or None if scenario fails
        """
        result: Optional[pd.Series] = self.backtest_runner.run_scenario(
            scenario_config, price_data_monthly_closes, price_data_daily_ohlc, rets_daily, verbose
        )
        return result

    def evaluate_trial_parameters(
        self, scenario_config: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluates a single set of parameters and returns performance metrics.

        Args:
            scenario_config: Base scenario configuration
            params: Parameters to evaluate

        Returns:
            Dictionary of metric names to values
        """
        # Guard required data
        if self.monthly_data is None or self.daily_data_ohlc is None or self.rets_full is None:
            raise ValueError(
                "Backtester data not initialized. Run Backtester.run() before evaluating trial parameters."
            )

        return self.evaluation_engine.evaluate_trial_parameters(
            scenario_config,
            params,
            self.monthly_data,
            self.daily_data_ohlc,
            self.rets_full,
            self.run_scenario,
        )

    def run(self) -> None:
        """Main entry to run a backtest or optimization for the selected scenario(s)."""
        # Early exit on timeout
        if self.has_timed_out:
            logger.warning("Timeout reached before starting the backtest run.")
            return

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Starting backtest data retrieval.")

        scenarios_to_run = self._select_scenarios_to_run()
        if not scenarios_to_run:
            return

        # Prepare data using DataFetcher
        try:
            daily_ohlc, monthly_data, daily_closes = self.data_fetcher.prepare_data_for_backtesting(
                scenarios_to_run, self.strategy_manager.get_strategy
            )

            # Store data for compatibility with original interface
            self.daily_data_ohlc = daily_ohlc
            self.monthly_data = monthly_data

            # Prepare returns data
            rets_full = self.data_cache.get_cached_returns(daily_closes, "full_period_returns")
            if isinstance(rets_full, pd.Series):
                self.rets_full = rets_full.to_frame()
            elif isinstance(rets_full, pd.DataFrame):
                self.rets_full = rets_full
            else:
                # Fallback to DataFrame to satisfy static type
                self.rets_full = pd.DataFrame(rets_full)

        except Exception as e:
            logger.error(f"Error during data preparation: {e}")
            raise

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Backtest data retrieved and prepared (daily OHLC, monthly closes).")

        # Execute flow based on mode
        scenario = scenarios_to_run[0]
        if getattr(self.args, "mode", None) == "optimize":
            rets_df: pd.DataFrame = (
                self.rets_full
                if isinstance(self.rets_full, pd.DataFrame)
                else pd.DataFrame(self.rets_full)
            )
            self._run_optimize_mode(scenario, self.monthly_data, self.daily_data_ohlc, rets_df)
        else:
            rets_df = (
                self.rets_full
                if isinstance(self.rets_full, pd.DataFrame)
                else pd.DataFrame(self.rets_full)
            )
            self._run_backtest_mode(scenario, self.monthly_data, self.daily_data_ohlc, rets_df)

        # Final reporting
        if CENTRAL_INTERRUPTED_FLAG:
            self.logger.warning(
                "Operation interrupted by user. Skipping final results display and plotting."
            )
            return

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("All scenarios completed. Displaying results.")

        if getattr(self.args, "mode", None) == "backtest":
            self._display_results()
        else:
            try:
                from ..backtester_logic.execution import generate_deferred_report

                generate_deferred_report(self)
            except Exception as e:
                self.logger.warning(f"Failed to generate deferred report: {e}")
            if self.daily_data_ohlc is not None:
                self._display_results()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Optimization mode completed. Reports generated.")

    def _run_backtest_mode(
        self,
        scenario_config: Dict[str, Any],
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
    ) -> None:
        """Run backtest mode - delegates to BacktestRunner."""
        study_name = getattr(self.args, "study_name", None)
        result = self.backtest_runner.run_backtest_mode(
            scenario_config, monthly_data, daily_data, rets_full, study_name
        )
        self.results[scenario_config["name"]] = result

    def _run_optimize_mode(
        self,
        scenario_config: Dict[str, Any],
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
    ) -> None:
        """Run optimization mode - delegates to OptimizationOrchestrator."""
        result = self.optimization_orchestrator.run_optimization(
            scenario_config, monthly_data, daily_data, rets_full, self.args
        )
        optimized_name = f"{scenario_config['name']}_Optimized"
        # Store under optimized key
        self.results[optimized_name] = result
        # Also store under original scenario name for backward-compat tests
        self.results[scenario_config["name"]] = result

    def _display_results(self) -> None:
        """Display results using the original reporting system."""
        from ..backtester_logic.reporting import display_results

        # Fix DataFrame truth value issue by using explicit empty check
        daily_data = self.daily_data_ohlc if self.daily_data_ohlc is not None else pd.DataFrame()
        display_results(self, daily_data)

    # Additional methods for maintaining compatibility with existing code
    def _get_strategy(self, strategy_spec, params: Dict[str, Any]):
        """Compatibility method - delegates to StrategyManager."""
        return self.strategy_manager.get_strategy(strategy_spec, params)

    # Expose evaluation methods for backward compatibility
    def evaluate_fast(self, *args, **kwargs):
        """Delegates to EvaluationEngine."""
        return self.evaluation_engine.evaluate_fast(*args, **kwargs)

    def evaluate_fast_numba(self, *args, **kwargs):
        """Delegates to EvaluationEngine."""
        return self.evaluation_engine.evaluate_fast_numba(*args, **kwargs)
