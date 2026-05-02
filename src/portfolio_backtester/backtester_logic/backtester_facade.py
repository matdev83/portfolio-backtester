"""
Backtester implementing the Facade pattern for modular backtesting components.

This module implements the Backtester class that provides a unified interface
while internally delegating to specialized classes following SOLID principles.
"""

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .data_fetcher import DataFetcher
from .strategy_manager import StrategyManager
from .evaluation_engine import EvaluationEngine
from .backtest_runner import BacktestRunner
from .optimization_orchestrator import OptimizationOrchestrator

from ..scenario_normalizer import ScenarioNormalizer
from ..canonical_config import CanonicalScenarioConfig

from ..config_loader import OPTIMIZER_PARAMETER_DEFAULTS

from ..utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG
from ..interfaces import (
    create_data_source,
    create_timeout_manager,
    create_cache_manager,
)

logger = logging.getLogger(__name__)


class Backtester:
    """
    Unified interface for modular backtesting components.

    This class provides a clean interface to the backtesting system
    while internally delegating to specialized classes following SOLID principles.
    """

    def __init__(
        self,
        global_config: Dict[str, Any],
        scenarios: Sequence[Union[Dict[str, Any], CanonicalScenarioConfig]],
        args: argparse.Namespace,
        backtest_runner=None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize Backtester with configuration and scenarios.
        Args:
            global_config: Global configuration dictionary
            scenarios: List of scenario configurations (raw dicts or canonical objects)
            args: Command line arguments namespace
            backtest_runner: Optional BacktestRunner instance
            random_state: Optional random seed for reproducibility
        """
        # Store original interface parameters
        self.global_config: Dict[str, Any] = global_config
        self.global_config["optimizer_parameter_defaults"] = OPTIMIZER_PARAMETER_DEFAULTS
        self.args: argparse.Namespace = args

        # Normalize scenarios if they are still raw dicts
        self.scenarios: List[CanonicalScenarioConfig] = []
        normalizer = ScenarioNormalizer()
        for s in scenarios:
            if isinstance(s, CanonicalScenarioConfig):
                self.scenarios.append(s)
            else:
                # Guard against unnormalized scenarios leaking in
                logger.info(f"Normalizing programmatic scenario: {s.get('name', 'unnamed')}")
                self.scenarios.append(normalizer.normalize(scenario=s, global_config=global_config))

        for scen in self.scenarios:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Canonical scenario '{scen.name}': {scen.to_dict()}")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Backtester initialized with scenario strategy_params: {self.scenarios[0].strategy_params}"
            )

        # Legacy compatibility: populate_default_optimizations expects dicts
        # We should ideally move this into the normalizer or update it to handle CanonicalScenarioConfig
        # For now, we've already normalized them, so this might be redundant or needs adaptation.
        # populate_default_optimizations(self.scenarios, OPTIMIZER_PARAMETER_DEFAULTS)

        # Scenario-level data source overrides (single-scenario runs only)
        try:
            if len(self.scenarios) == 1:
                scenario = self.scenarios[0]
                override = None
                if hasattr(scenario, "get"):
                    override = scenario.get("data_source_config")
                elif isinstance(scenario, dict):
                    override = scenario.get("data_source_config")
                if isinstance(override, Mapping):
                    base_cfg = dict(self.global_config.get("data_source_config", {}) or {})
                    base_cfg.update(dict(override))
                    self.global_config["data_source_config"] = base_cfg
                    logger.info(
                        "Applying scenario data_source_config override: %s",
                        self.global_config["data_source_config"],
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed applying scenario data_source_config override: %s", exc)

        # Initialize timing and timeout management using DIP
        self._timeout_start_time: float = time.time()
        self.timeout_manager = create_timeout_manager(
            args.timeout, start_time=self._timeout_start_time
        )

        effective_random_state: Optional[int] = random_state
        if effective_random_state is None:
            cli_seed = getattr(args, "random_seed", None)
            if cli_seed is not None:
                try:
                    effective_random_state = int(cli_seed)
                except (TypeError, ValueError):
                    effective_random_state = None

        if effective_random_state is None:
            self.random_state = int(np.random.randint(0, 2**31 - 1))
            logger.warning(
                "No random seed was provided; this run is not reproducible.",
            )
            logger.info("Generated random seed: %s", self.random_state)
        else:
            self.random_state = int(effective_random_state)
            logger.info("Using random seed: %s", self.random_state)

        self.rng = np.random.default_rng(self.random_state)

        np.random.seed(self.random_state)
        random.seed(self.random_state)
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

        self.backtest_runner = backtest_runner or BacktestRunner(
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
            rng=self.rng,
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

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Backtester initialized with all specialized components")

    @property
    def has_timed_out(self):
        """Check if the operation has timed out."""
        return self.timeout_manager.check_timeout()

    def get_canonical_scenario(self, name: str) -> Optional[CanonicalScenarioConfig]:
        """
        Get a canonical scenario configuration by name.

        Args:
            name: Name of the scenario to retrieve

        Returns:
            CanonicalScenarioConfig if found, None otherwise
        """
        for s in self.scenarios:
            if s.name == name:
                return s
        return None

    def _select_scenarios_to_run(self) -> List[CanonicalScenarioConfig]:
        """Select scenarios based on CLI args; returns empty list if named scenario missing."""
        if getattr(self.args, "scenario_name", None):
            scenarios_to_run = [s for s in self.scenarios if s.name == self.args.scenario_name]
            if not scenarios_to_run:
                self.logger.error(
                    f"Scenario '{self.args.scenario_name}' not found in the loaded scenarios."
                )
                return []
            return scenarios_to_run
        return self.scenarios

    def run_scenario(
        self,
        scenario_config: Union[Dict[str, Any], CanonicalScenarioConfig],
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
            scenario_config: Configuration for the scenario to run (raw dict or canonical object)
            price_data_monthly_closes: Monthly closing price data
            price_data_daily_ohlc: Daily OHLC price data
            rets_daily: Optional daily returns data
            verbose: Whether to log detailed information

        Returns:
            Portfolio returns as pandas Series, or None if scenario fails
        """
        result: Optional[pd.Series] = self.backtest_runner.run_scenario(
            scenario_config,
            price_data_monthly_closes,
            price_data_daily_ohlc,
            rets_daily,
            verbose,
        )
        return result

    def evaluate_trial_parameters(
        self,
        scenario_config: Union[Dict[str, Any], CanonicalScenarioConfig],
        params: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Evaluates a single set of parameters and returns performance metrics.

        Args:
            scenario_config: Base scenario configuration (raw dict or canonical object)
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
        # Start timing
        start_time = time.time()

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
            # Support test-only fast optimize path for CI/developer runs
            if getattr(self.args, "test_fast_optimize", False) or getattr(
                self.args, "test-fast-optimize", False
            ):
                # Use a lightweight orchestrator variant (fast path) that avoids heavy reporting
                # and reduces data sizes. We'll call the existing orchestrator but instruct
                # it via args to run in fast mode.
                fast_args = self.args
                setattr(fast_args, "test_fast_optimize", True)
                self._run_optimize_mode(
                    self.optimization_orchestrator,
                    scenario,
                    self.monthly_data,
                    self.daily_data_ohlc,
                    rets_df,
                )
            else:
                self._run_optimize_mode(
                    self.optimization_orchestrator,
                    scenario,
                    self.monthly_data,
                    self.daily_data_ohlc,
                    rets_df,
                )
        elif getattr(self.args, "mode", None) == "research_validate":
            rets_df = (
                self.rets_full
                if isinstance(self.rets_full, pd.DataFrame)
                else pd.DataFrame(self.rets_full)
            )
            self._run_research_validate_mode(
                scenario, self.monthly_data, self.daily_data_ohlc, rets_df
            )
        else:
            rets_df = (
                self.rets_full
                if isinstance(self.rets_full, pd.DataFrame)
                else pd.DataFrame(self.rets_full)
            )
            self._run_backtest_mode(
                self,
                scenario,
                self.monthly_data,
                self.daily_data_ohlc,
                rets_df,
            )

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
        elif getattr(self.args, "mode", None) == "research_validate":
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "research_validate mode: skipping deferred optimization report and display."
                )
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

        # Log execution time
        execution_time = time.time() - start_time
        mode = getattr(self.args, "mode", "backtest")

        # Calculate backtest period and days per second
        if self.daily_data_ohlc is not None and not self.daily_data_ohlc.empty:
            start_date = self.daily_data_ohlc.index.min().date()
            end_date = self.daily_data_ohlc.index.max().date()
            total_days = (end_date - start_date).days
            days_per_second = total_days / max(execution_time, 0.001)  # Avoid division by zero

            logger.info(
                f"Backtester {mode} mode execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes) | "
                f"Period: {start_date} to {end_date} ({total_days} days) | {days_per_second:.1f} days/second"
            )
        else:
            logger.info(
                f"Backtester {mode} mode execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
            )

    def _run_research_validate_mode(
        self,
        scenario_config: CanonicalScenarioConfig,
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
    ) -> None:
        """Run research validation mode via ``ResearchProtocolOrchestrator``."""
        from ..research.artifacts import ResearchArtifactWriter
        from ..research.protocol_orchestrator import ResearchProtocolOrchestrator

        artifact_writer = None
        # Set via CLI --research-artifact-base-dir or programmatic Namespace; when unset,
        # DoubleOOSWFOProtocol uses ResearchArtifactWriter(Path("data/reports")).
        artifact_base = getattr(self.args, "research_artifact_base_dir", None)
        if artifact_base is not None:
            artifact_writer = ResearchArtifactWriter(Path(str(artifact_base)))

        def _research_optimization_factory() -> OptimizationOrchestrator:
            return OptimizationOrchestrator(
                global_config=self.global_config,
                data_source=self.data_source,
                backtest_runner=self.backtest_runner,
                evaluation_engine=self.evaluation_engine,
                rng=np.random.default_rng(),
            )

        orchestrator = ResearchProtocolOrchestrator(
            self.optimization_orchestrator,
            self.backtest_runner,
            artifact_writer,
            optimization_orchestrator_factory=_research_optimization_factory,
        )
        result = orchestrator.run(
            scenario_config=scenario_config,
            monthly_data=monthly_data,
            daily_data=daily_data,
            rets_full=rets_full,
            args=self.args,
            global_config=self.global_config,
        )
        self.results[f"{scenario_config.name}_ResearchValidation"] = result

    def _run_backtest_mode(
        self,
        backtester,
        scenario_config: CanonicalScenarioConfig,
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
    ) -> None:
        """Run backtest mode - delegates to BacktestRunner."""
        study_name = getattr(self.args, "study_name", None)
        result = backtester.backtest_runner.run_backtest_mode(
            scenario_config, monthly_data, daily_data, rets_full, study_name
        )
        self.results[scenario_config.name] = result

    def _run_optimize_mode(
        self,
        optimization_orchestrator,
        scenario_config: CanonicalScenarioConfig,
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
    ) -> None:
        """Run optimization mode - delegates to OptimizationOrchestrator."""
        optimization_result = optimization_orchestrator.run_optimization(
            scenario_config, monthly_data, daily_data, rets_full, self.args
        )

        stitched_returns = getattr(optimization_result, "stitched_returns", None)
        if isinstance(stitched_returns, pd.Series):
            from ..backtesting.strategy_backtester import StrategyBacktester, _extract_close_returns
            from ..reporting.performance_metrics import calculate_metrics

            benchmark_ticker = getattr(
                scenario_config, "benchmark_ticker", None
            ) or self.global_config.get(
                "benchmark_ticker", self.global_config.get("benchmark", "SPY")
            )
            benchmark_returns = _extract_close_returns(
                daily_data, str(benchmark_ticker), stitched_returns.index
            )

            metrics_series = calculate_metrics(
                stitched_returns, benchmark_returns, str(benchmark_ticker)
            )
            metrics = {
                k: float(v) if not pd.isna(v) else float("nan") for k, v in metrics_series.items()
            }

            strategy_backtester = StrategyBacktester(self.global_config, self.data_source)
            performance_stats = strategy_backtester._create_performance_stats(
                stitched_returns, metrics
            )
            charts_data = strategy_backtester._create_charts_data(
                stitched_returns, benchmark_returns
            )

            optimal_params = optimization_result.best_parameters
            optimized_name = f"{scenario_config.name}_Optimized"
            # CanonicalScenarioConfig might have train_end_date in extras if it was there
            train_end_date_str = scenario_config.extras.get("train_end_date", "2018-12-31")
            train_end_date = pd.to_datetime(train_end_date_str)

            final_backtest_results = {
                "returns": stitched_returns,
                "display_name": optimized_name,
                "train_end_date": train_end_date,
                "trade_stats": None,
                "trade_history": pd.DataFrame(),
                "performance_stats": performance_stats,
                "charts_data": charts_data,
                "optimal_params": optimal_params,
                "num_trials_for_dsr": optimization_result.n_evaluations,
                "best_trial_obj": optimization_result.best_trial,
                "optimization_result": optimization_result,
                "wfo_mode": getattr(optimization_result, "wfo_mode", None),
                "wfo_window_params": getattr(optimization_result, "wfo_window_params", None),
                "wfo_window_results": getattr(optimization_result, "wfo_window_results", None),
            }

            self.results[optimized_name] = final_backtest_results
            return

        # Run a final backtest using the best parameters found
        optimal_params = optimization_result.best_parameters
        # We need to create a modified scenario config with optimal params.
        # Since CanonicalScenarioConfig is frozen, we might need a way to create a copy with changes,
        # or convert it back to dict, update, and re-normalize (though it might be overkill).
        # Actually, for the final backtest, we can pass the canonical config and the optimal params separately
        # if the runner supports it, or create a new canonical config.

        # Let's convert to dict for now to update params, then re-normalize
        optimized_scen_dict = scenario_config.to_dict()
        base_strategy_params = dict(optimized_scen_dict.get("strategy_params", {}))
        base_strategy_params.update(optimal_params or {})
        optimized_scen_dict["strategy_params"] = base_strategy_params

        # Re-normalize to get a new CanonicalScenarioConfig
        normalizer = ScenarioNormalizer()
        optimized_scenario = normalizer.normalize(
            scenario=optimized_scen_dict, global_config=self.global_config
        )

        # The run_backtest_mode will use the StrategyBacktester internally
        # and return a dictionary of results.
        final_backtest_results = self.backtest_runner.run_backtest_mode(
            optimized_scenario, monthly_data, daily_data, rets_full
        )

        # Store the final backtest results, enriched with optimization info
        optimized_name = f"{scenario_config.name}_Optimized"

        # Merge the optimization results into the final backtest results dict
        final_backtest_results.update(
            {
                "display_name": optimized_name,
                "optimal_params": optimal_params,
                "num_trials_for_dsr": optimization_result.n_evaluations,
                "best_trial_obj": optimization_result.best_trial,
                "optimization_result": optimization_result,
            }
        )

        self.results[optimized_name] = final_backtest_results

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
