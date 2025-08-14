"""
Optimization workflow orchestration logic extracted from Backtester class.

This module implements the OptimizationOrchestrator class that handles all optimization-related
operations including parameter space conversion, optimization setup, and result processing.
"""

import logging
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from ..optimization.results import OptimizationData, OptimizationResult
from ..interfaces.attribute_accessor_interface import (
    IAttributeAccessor,
    create_attribute_accessor,
)

logger = logging.getLogger(__name__)


class OptimizationOrchestrator:
    """
    Handles optimization workflow orchestration for backtesting.

    This class encapsulates all optimization-related operations that were previously
    part of the Backtester class, following the Single Responsibility Principle.
    """

    def __init__(
        self,
        global_config: Dict[str, Any],
        data_source: Any,
        backtest_runner: Any,
        evaluation_engine: Any,
        rng: np.random.Generator,
        attribute_accessor: Optional[IAttributeAccessor] = None,
    ) -> None:
        """
        Initialize OptimizationOrchestrator with dependencies.

        Args:
            global_config: Global configuration dictionary
            data_source: Data source instance for fetching market data
            backtest_runner: BacktestRunner instance for running backtests
            evaluation_engine: EvaluationEngine instance for performance evaluation
            rng: NumPy random number generator for reproducibility
            attribute_accessor: Injected accessor for attribute access (DIP)
        """
        self.global_config = global_config
        self.data_source = data_source
        self.backtest_runner = backtest_runner
        self.evaluation_engine = evaluation_engine
        self.rng = rng
        self.logger = logger
        # Dependency injection for attribute access (DIP)
        self._attribute_accessor = attribute_accessor or create_attribute_accessor()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("OptimizationOrchestrator initialized")

    def _get_default_optuna_storage_url(self) -> str:
        """Get the default Optuna storage URL."""
        from ..constants import DEFAULT_OPTUNA_STORAGE_URL

        return DEFAULT_OPTUNA_STORAGE_URL

    def run_optimization(
        self,
        scenario_config: Dict[str, Any],
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
        optimizer_args: Any,
    ) -> OptimizationResult:
        # Start timing
        start_time = time.time()
        """
        Run optimization mode using the new OptimizationOrchestrator architecture.

        This method uses the new architecture with OptimizationOrchestrator,
        parameter generators, and BacktestEvaluator to perform optimization.

        Args:
            scenario_config: Scenario configuration
            monthly_data: Monthly price data
            daily_data: Daily OHLC data
            rets_full: Full period returns data
            optimizer_args: Arguments from CLI/configuration for optimization

        Returns:
            Dictionary containing optimization results
        """
        from ..optimization.factory import create_parameter_generator
        from ..optimization.orchestrator_factory import create_orchestrator
        from ..optimization.evaluator import BacktestEvaluator
        from ..backtesting.strategy_backtester import StrategyBacktester
        from ..utils import generate_enhanced_wfo_windows

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Running optimization using new architecture for scenario: {scenario_config['name']}"
            )

        # Get optimizer type from CLI args (this is the key integration point)
        optimizer_type = self._attribute_accessor.get_attribute(
            optimizer_args, "optimizer", "optuna"
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Using optimizer type from CLI: {optimizer_type}")

        try:
            # Create parameter generator using factory
            parameter_generator = create_parameter_generator(
                optimizer_type=optimizer_type, random_state=self.rng
            )
        except (ImportError, ValueError) as e:
            logger.error(f"Failed to create parameter generator '{optimizer_type}': {e}")
            raise

        try:
            # Create StrategyBacktester for pure backtesting
            strategy_backtester = StrategyBacktester(
                global_config=self.global_config, data_source=self.data_source
            )

            # Generate walk-forward windows
            # Ensure DatetimeIndex with explicit local annotation for type-checker
            # Coerce to pandas DatetimeIndex from the monthly_data index values
            monthly_idx: pd.DatetimeIndex = pd.DatetimeIndex(pd.to_datetime(monthly_data.index))
            windows = generate_enhanced_wfo_windows(
                monthly_idx, scenario_config, self.global_config, self.rng
            )

            if not windows:
                raise ValueError("Not enough data for the requested walk-forward windows.")

            # Determine optimization targets and metrics
            optimization_targets_config = scenario_config.get("optimization_targets", [])
            metrics_to_optimize = [t["name"] for t in optimization_targets_config] or [
                scenario_config.get("optimization_metric", "Calmar")
            ]
            is_multi_objective = len(metrics_to_optimize) > 1

            # Prepare optimization data
            # Hoist DataFrame → ndarray conversions once per optimization run (float32, C-contiguous)
            if isinstance(daily_data.columns, pd.MultiIndex):
                daily_close_df_maybe = daily_data.xs("Close", level="Field", axis=1)
                daily_close_df = (
                    daily_close_df_maybe
                    if isinstance(daily_close_df_maybe, pd.DataFrame)
                    else pd.DataFrame(daily_close_df_maybe)
                )
            else:
                daily_close_df = daily_data
            tickers_list = list(daily_close_df.columns)
            daily_np = np.ascontiguousarray(daily_close_df.to_numpy(dtype=np.float32))
            rets_full_df: pd.DataFrame = (
                rets_full if isinstance(rets_full, pd.DataFrame) else pd.DataFrame(rets_full)
            )
            returns_df = (
                rets_full_df.reindex(daily_close_df.index).reindex(columns=tickers_list).fillna(0.0)
            )
            returns_np = np.ascontiguousarray(returns_df.to_numpy(dtype=np.float32))
            daily_index_np = daily_close_df.index.values.astype("datetime64[ns]")

            optimization_data = OptimizationData(
                monthly=monthly_data,
                daily=daily_data,
                returns=rets_full,
                windows=windows,
            )
            # Attach prepared arrays for reuse across trials/windows (runtime cache)
            optimization_data.daily_np = daily_np  # type: ignore[attr-defined]
            optimization_data.returns_np = returns_np  # type: ignore[attr-defined]
            optimization_data.daily_index_np = daily_index_np  # type: ignore[attr-defined]
            optimization_data.tickers_list = tickers_list  # type: ignore[attr-defined]

            # Convert scenario optimization specs to parameter space format
            parameter_space = self.convert_optimization_specs_to_parameter_space(scenario_config)

            # Create optimization config from CLI args and scenario config
            optimization_config = {
                "parameter_space": parameter_space,
                "max_evaluations": self._attribute_accessor.get_attribute(
                    optimizer_args, "optuna_trials", 200
                ),
                "timeout_seconds": self._attribute_accessor.get_attribute(
                    optimizer_args, "optuna_timeout_sec", None
                ),
                "optimization_targets": optimization_targets_config,
                "metrics_to_optimize": metrics_to_optimize,
                "pruning_enabled": self._attribute_accessor.get_attribute(
                    optimizer_args, "pruning_enabled", False
                ),
                "pruning_n_startup_trials": self._attribute_accessor.get_attribute(
                    optimizer_args, "pruning_n_startup_trials", 5
                ),
                "pruning_n_warmup_steps": self._attribute_accessor.get_attribute(
                    optimizer_args, "pruning_n_warmup_trials", 0
                ),
                "pruning_interval_steps": self._attribute_accessor.get_attribute(
                    optimizer_args, "pruning_interval_steps", 1
                ),
                "study_name": self._attribute_accessor.get_attribute(
                    optimizer_args, "study_name", None
                ),
                "storage_url": self._attribute_accessor.get_attribute(
                    optimizer_args, "storage_url", None
                ),
                "early_stop_zero_trials": self._attribute_accessor.get_attribute(
                    optimizer_args, "early_stop_zero_trials", 20
                ),
                "random_seed": self.rng,
                "fresh_study": self._attribute_accessor.get_attribute(
                    optimizer_args, "fresh_study", False
                ),
                # GA-specific knobs (used when optimizer_type is population-based)
                "ga_settings": {
                    "population_size": self._attribute_accessor.get_attribute(
                        optimizer_args, "ga_population_size", 50
                    ),
                    "max_generations": self._attribute_accessor.get_attribute(
                        optimizer_args, "ga_max_generations", 10
                    ),
                    "mutation_rate": self._attribute_accessor.get_attribute(
                        optimizer_args, "ga_mutation_rate", 0.1
                    ),
                    "crossover_rate": self._attribute_accessor.get_attribute(
                        optimizer_args, "ga_crossover_rate", 0.8
                    ),
                },
                # Deduplication settings
                "use_persistent_cache": self._attribute_accessor.get_attribute(
                    optimizer_args, "use_persistent_cache", False
                ),
                "cache_dir": self._attribute_accessor.get_attribute(
                    optimizer_args, "cache_dir", None
                ),
                "cache_file": self._attribute_accessor.get_attribute(
                    optimizer_args, "cache_file", None
                ),
            }

            # Create BacktestEvaluator
            # For population-based optimizers, disable window-level parallelism to avoid
            # nested oversubscription (population parallelism is handled separately).
            if optimizer_type in [
                "genetic",
                "particle_swarm",
                "differential_evolution",
            ]:
                evaluator = BacktestEvaluator(
                    metrics_to_optimize=metrics_to_optimize,
                    is_multi_objective=is_multi_objective,
                    n_jobs=1,
                    enable_parallel_optimization=False,
                )
            else:
                evaluator = BacktestEvaluator(
                    metrics_to_optimize=metrics_to_optimize,
                    is_multi_objective=is_multi_objective,
                )

            # Create OptimizationOrchestrator using the factory
            orchestrator = create_orchestrator(
                optimizer_type=optimizer_type,
                parameter_generator=parameter_generator,
                evaluator=evaluator,
                n_jobs=self._attribute_accessor.get_attribute(optimizer_args, "n_jobs", -1),
                joblib_batch_size=self._attribute_accessor.get_attribute(
                    optimizer_args, "joblib_batch_size", None
                ),
                joblib_pre_dispatch=self._attribute_accessor.get_attribute(
                    optimizer_args, "joblib_pre_dispatch", None
                ),
                timeout=self._attribute_accessor.get_attribute(
                    optimizer_args, "optuna_timeout_sec", None
                ),
                early_stop_patience=self._attribute_accessor.get_attribute(
                    optimizer_args, "early_stop_patience", 10
                ),
            )

            optimization_result = orchestrator.optimize(
                scenario_config=scenario_config,
                optimization_config=optimization_config,
                data=optimization_data,
                backtester=strategy_backtester,  # type: ignore[arg-type]
            )

            # The rest of this function handles running a final backtest on the
            # best parameters and storing the results. For the purpose of the
            # equivalence test, we only need the OptimizationResult object itself.
            # The calling Backtester will handle the final backtest run.

            # Log execution time with additional information
            execution_time = time.time() - start_time

            # Calculate backtest period
            start_date = daily_data.index.min().date() if not daily_data.empty else None
            end_date = daily_data.index.max().date() if not daily_data.empty else None
            total_days = (end_date - start_date).days if start_date and end_date else 0

            # Get parameter combinations information
            n_trials = optimization_result.n_evaluations
            combinations_per_minute = (
                (n_trials / (execution_time / 60)) if execution_time > 0 else 0
            )

            logger.info(
                f"Optimization ({optimizer_type}) execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes) | "
                f"Period: {start_date} to {end_date} ({total_days} days) | "
                f"Parameter combinations tested: {n_trials} ({combinations_per_minute:.1f} combinations/minute)"
            )

            return optimization_result

        except (ValueError, TypeError) as e:
            # Log execution time even on failure
            execution_time = time.time() - start_time
            logger.error(f"Error during optimization setup: {e}")

            # Calculate backtest period if possible
            start_date = None
            end_date = None
            total_days = 0
            if "daily_data" in locals() and not daily_data.empty:
                start_date = daily_data.index.min().date()
                end_date = daily_data.index.max().date()
                total_days = (end_date - start_date).days

            logger.info(
                f"Failed optimization execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
                + (
                    f" | Period: {start_date} to {end_date} ({total_days} days)"
                    if start_date and end_date
                    else ""
                )
            )
            raise
        except Exception as e:
            # Log execution time even on failure
            execution_time = time.time() - start_time
            logger.error(f"An unexpected error occurred during optimization: {e}")

            # Calculate backtest period if possible
            start_date = None
            end_date = None
            total_days = 0
            if "daily_data" in locals() and not daily_data.empty:
                start_date = daily_data.index.min().date()
                end_date = daily_data.index.max().date()
                total_days = (end_date - start_date).days

            logger.info(
                f"Failed optimization execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
                + (
                    f" | Period: {start_date} to {end_date} ({total_days} days)"
                    if start_date and end_date
                    else ""
                )
            )
            raise

    def convert_optimization_specs_to_parameter_space(
        self, scenario_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert scenario optimization specs to parameter space format.

        This method supports both legacy and modern optimization parameter formats:
        1. Legacy format: 'optimize' section with min_value/max_value/step
        2. Modern format: 'strategy.params.param_name.optimization' with range/step/exclude

        Args:
            scenario_config: Scenario configuration containing optimization parameters

        Returns:
            Dictionary defining the parameter space for optimization
        """
        parameter_space = {}

        # Handle legacy format: top-level 'optimize' section
        legacy_optimization_specs = scenario_config.get("optimize", [])
        for spec in legacy_optimization_specs:
            param_name = spec["parameter"]

            # Get parameter type from spec or defaults
            param_type = spec.get("type")
            if not param_type:
                if "min_value" in spec and "max_value" in spec:
                    # Infer type from min/max value
                    if isinstance(spec["min_value"], int) and isinstance(spec["max_value"], int):
                        param_type = "int"
                    else:
                        param_type = "float"
                else:
                    param_type = (
                        self.global_config.get("optimizer_parameter_defaults", {})
                        .get(param_name, {})
                        .get("type", "float")
                    )

            # Convert to parameter space format
            if param_type == "int":
                parameter_space[param_name] = {
                    "type": "int",
                    "low": spec["min_value"],
                    "high": spec["max_value"],
                    "step": spec.get("step", 1),
                }
            elif param_type == "float":
                parameter_space[param_name] = {
                    "type": "float",
                    "low": spec["min_value"],
                    "high": spec.get("max_value"),
                    "step": spec.get("step", None),
                }
            elif param_type == "categorical":
                choices = spec.get("choices") or spec.get("values")
                if not choices:
                    logger.error(
                        f"Categorical parameter '{param_name}' is missing 'choices' or 'values' in spec: {spec}"
                    )
                    raise KeyError(
                        f"Categorical parameter '{param_name}' must have 'choices' or 'values' defined."
                    )
                parameter_space[param_name] = {
                    "type": "categorical",
                    "choices": choices,
                }
            elif param_type == "multi-categorical":
                choices = spec.get("choices") or spec.get("values")
                if not choices:
                    logger.error(
                        f"Multi-categorical parameter '{param_name}' is missing 'choices' or 'values' in spec: {spec}"
                    )
                    raise KeyError(
                        f"Multi-categorical parameter '{param_name}' must have 'choices' or 'values' defined."
                    )
                parameter_space[param_name] = {
                    "type": "multi-categorical",
                    "values": choices,
                }

        # Handle modern format: strategy.params.param_name.optimization
        strategy_config_obj: Any = scenario_config.get("strategy", {})
        strategy_config: Dict[str, Any] = (
            strategy_config_obj if isinstance(strategy_config_obj, dict) else {}
        )
        if isinstance(strategy_config, dict):
            # Ensure proper types for nested config to satisfy static type checkers
            # Coerce params safely to a dict
            raw_params_any: Any = strategy_config.get("params")
            params_config: Dict[str, Any] = (
                raw_params_any if isinstance(raw_params_any, dict) else {}
            )
            if isinstance(params_config, dict):
                for param_name, param_config in params_config.items():
                    if isinstance(param_config, dict) and "optimization" in param_config:
                        # Guard against None to satisfy type checker and runtime
                        opt_raw: Any = param_config.get("optimization")
                        if not isinstance(opt_raw, dict):
                            continue
                        opt_config: Dict[str, Any] = opt_raw

                        if "range" in opt_config:
                            # Handle range-based optimization
                            range_values = opt_config["range"]
                            if not isinstance(range_values, list) or len(range_values) < 2:
                                logger.error(
                                    f"Invalid range for parameter '{param_name}': {range_values}"
                                )
                                continue

                            # Determine parameter type from range values
                            if all(isinstance(v, int) for v in range_values):
                                param_type = "int"
                            else:
                                param_type = "float"

                            # Create parameter space entry
                            parameter_space[param_name] = {
                                "type": param_type,
                                "low": range_values[0],
                                "high": range_values[1],
                                "step": opt_config.get("step", None),
                            }

                        elif "values" in opt_config:
                            # Handle values-based optimization (categorical)
                            values = opt_config["values"]
                            if not isinstance(values, list):
                                logger.error(
                                    f"Invalid values for parameter '{param_name}': {values}"
                                )
                                continue

                            # Determine if multi-categorical or categorical
                            if opt_config.get("multi_select", False):
                                parameter_space[param_name] = {
                                    "type": "multi-categorical",
                                    "values": values,
                                }
                            else:
                                parameter_space[param_name] = {
                                    "type": "categorical",
                                    "choices": values,
                                }

                        # Handle exclude list (for both range and values)
                        exclude = opt_config.get("exclude", [])
                        if exclude:
                            parameter_space[param_name]["exclude"] = exclude

        return parameter_space

    def validate_optimization_config(self, scenario_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate optimization configuration."""
        errors = []
        if not scenario_config.get("optimize"):
            errors.append("Missing 'optimize' section in scenario configuration.")
            return False, errors

        strategy_spec = scenario_config.get("strategy")
        if isinstance(strategy_spec, dict):
            if not strategy_spec.get("params"):
                errors.append("Missing 'params' in strategy specification.")
        
        return len(errors) == 0, errors
