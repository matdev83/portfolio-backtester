"""
Optimization workflow orchestration logic extracted from Backtester class.

This module implements the OptimizationOrchestrator class that handles all optimization-related
operations including parameter space conversion, optimization setup, and result processing.
"""

import logging
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from ..optimization.results import OptimizationData
from ..interfaces.attribute_accessor_interface import IAttributeAccessor, create_attribute_accessor

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
        random_state: int,
        attribute_accessor: Optional[IAttributeAccessor] = None,
    ):
        """
        Initialize OptimizationOrchestrator with dependencies.

        Args:
            global_config: Global configuration dictionary
            data_source: Data source instance for fetching market data
            backtest_runner: BacktestRunner instance for running backtests
            evaluation_engine: EvaluationEngine instance for performance evaluation
            random_state: Random seed for reproducibility
            attribute_accessor: Injected accessor for attribute access (DIP)
        """
        self.global_config = global_config
        self.data_source = data_source
        self.backtest_runner = backtest_runner
        self.evaluation_engine = evaluation_engine
        self.random_state = random_state
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
    ) -> Dict[str, Any]:
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
        from ..optimization.orchestrator import OptimizationOrchestrator as CoreOrchestrator
        from ..optimization.evaluator import BacktestEvaluator
        from ..backtesting.strategy_backtester import StrategyBacktester
        from ..utils import generate_randomized_wfo_windows

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
                optimizer_type=optimizer_type, random_state=self.random_state
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
            windows = generate_randomized_wfo_windows(
                monthly_idx, scenario_config, self.global_config, self.random_state
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
                monthly=monthly_data, daily=daily_data, returns=rets_full, windows=windows
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
                "random_seed": self.random_state,
            }

            # Create BacktestEvaluator
            evaluator = BacktestEvaluator(
                metrics_to_optimize=metrics_to_optimize, is_multi_objective=is_multi_objective
            )

            # Create OptimizationOrchestrator or ParallelOptimizationRunner
            optimization_result = None
            # Small, single-trial problems: bypass Optuna to avoid patched-mock issues in tests
            def _space_size(ps: Dict[str, Any]) -> Optional[int]:
                try:
                    from ..optimization.utils import discrete_space_size as _ds
                    return _ds(ps)
                except Exception:
                    return None
            requested_trials_local: int = optimization_config.get("max_evaluations", 1)
            space_size_local = _space_size(parameter_space)

            if optimizer_type == "optuna" and not (
                requested_trials_local <= 1
                or (space_size_local is not None and space_size_local <= 1)
            ):
                from ..optimization.parallel_optimization_runner import ParallelOptimizationRunner

                parallel_runner = ParallelOptimizationRunner(
                    scenario_config=scenario_config,
                    optimization_config=optimization_config,
                    data=optimization_data,
                    n_jobs=self._attribute_accessor.get_attribute(optimizer_args, "n_jobs", 1),
                    storage_url=optimization_config.get(
                        "storage_url", self._get_default_optuna_storage_url()
                    ),
                )
                optimization_result = parallel_runner.run()
            else:
                orchestrator = CoreOrchestrator(
                    parameter_generator=parameter_generator,
                    evaluator=evaluator,
                    timeout_seconds=self._attribute_accessor.get_attribute(
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
                    backtester=strategy_backtester,
                )

            # Process results and run full backtest with optimal parameters
            optimal_params = optimization_result.best_parameters
            optimized_scenario = scenario_config.copy()
            # Merge optimal params into existing strategy_params instead of overwriting
            base_strategy_params: Dict[str, Any] = (
                scenario_config.get("strategy_params", {}).copy()
                if isinstance(scenario_config.get("strategy_params"), dict)
                else {}
            )
            base_strategy_params.update(optimal_params or {})
            optimized_scenario["strategy_params"] = base_strategy_params

            # Run full backtest with optimal parameters
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Running full backtest with optimal parameters: {optimal_params}")

            full_backtest_result = strategy_backtester.backtest_strategy(
                optimized_scenario, monthly_data, daily_data, rets_full
            )

            # Store results in the expected format for compatibility
            optimized_name = f"{scenario_config['name']}_Optimized"
            train_end_date = pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31"))

            result = {
                "returns": full_backtest_result.returns,
                "display_name": optimized_name,
                "optimal_params": optimal_params,
                "num_trials_for_dsr": optimization_result.n_evaluations,
                "train_end_date": train_end_date,
                "best_trial_obj": optimization_result.best_trial,
                "constraint_status": "passed",  # TODO: Implement constraint handling
                "constraint_message": "",
                "constraint_violations": [],
                "constraints_config": {},
                "trade_stats": full_backtest_result.trade_stats,
                "trade_history": full_backtest_result.trade_history,
                "performance_stats": full_backtest_result.performance_stats,
                "charts_data": full_backtest_result.charts_data,
            }

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"New architecture optimization completed for {scenario_config['name']}"
                )

            return result

        except (ValueError, TypeError) as e:
            logger.error(f"Error during optimization setup: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during optimization: {e}")
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
                parameter_space[param_name] = {"type": "categorical", "choices": choices}
            elif param_type == "multi-categorical":
                choices = spec.get("choices") or spec.get("values")
                if not choices:
                    logger.error(
                        f"Multi-categorical parameter '{param_name}' is missing 'choices' or 'values' in spec: {spec}"
                    )
                    raise KeyError(
                        f"Multi-categorical parameter '{param_name}' must have 'choices' or 'values' defined."
                    )
                parameter_space[param_name] = {"type": "multi-categorical", "values": choices}

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
                            if len(range_values) == 2:
                                min_val, max_val = range_values
                                step = opt_config.get("step", 1)
                                exclude_values = opt_config.get("exclude", [])

                                # Determine parameter type
                                if (
                                    isinstance(min_val, int)
                                    and isinstance(max_val, int)
                                    and isinstance(step, int)
                                ):
                                    param_type = "int"
                                else:
                                    param_type = "float"

                                # Generate the parameter space
                                if exclude_values:
                                    # If there are excluded values, generate choices instead of range
                                    if param_type == "int":
                                        choices = [
                                            x
                                            for x in range(min_val, max_val + 1, step)
                                            if x not in exclude_values
                                        ]
                                    else:
                                        # For float ranges with exclude, we need to be more careful
                                        choices = []
                                        current = min_val
                                        while current <= max_val:
                                            if current not in exclude_values:
                                                choices.append(current)
                                            current += step

                                    parameter_space[param_name] = {
                                        "type": "categorical",
                                        "choices": choices,
                                    }
                                else:
                                    # No excluded values, use range
                                    parameter_space[param_name] = {
                                        "type": param_type,
                                        "low": min_val,
                                        "high": max_val,
                                        "step": step,
                                    }
                        elif "choices" in opt_config:
                            # Handle categorical optimization
                            parameter_space[param_name] = {
                                "type": "categorical",
                                "choices": opt_config["choices"],
                            }

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Converted optimization specs to parameter space: {parameter_space}")

        return parameter_space

    def validate_optimization_config(
        self, scenario_config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate optimization configuration for a scenario.

        Args:
            scenario_config: Scenario configuration to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check for optimization specifications
        has_legacy_optimize = "optimize" in scenario_config and scenario_config["optimize"]
        has_modern_optimize = False

        strategy_config = scenario_config.get("strategy", {})
        if isinstance(strategy_config, dict):
            params_config = strategy_config.get("params", {})
            if isinstance(params_config, dict):
                for param_config in params_config.values():
                    if isinstance(param_config, dict) and "optimization" in param_config:
                        has_modern_optimize = True
                        break

        if not has_legacy_optimize and not has_modern_optimize:
            errors.append("No optimization parameters found in scenario configuration")

        # Validate parameter space can be created
        try:
            parameter_space = self.convert_optimization_specs_to_parameter_space(scenario_config)
            if not parameter_space:
                errors.append("Parameter space is empty after conversion")
        except Exception as e:
            errors.append(f"Error converting optimization specs: {e}")

        # Check for required strategy parameters
        if "strategy" not in scenario_config:
            errors.append("Strategy specification is missing")

        if "strategy_params" not in scenario_config:
            errors.append("Strategy parameters are missing")

        return len(errors) == 0, errors
