"""
Evaluation engine logic extracted from Backtester class.

This module implements the EvaluationEngine class that handles all performance
evaluation operations including fast evaluation, walk-forward analysis, and parameter testing.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Union, cast, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..canonical_config import CanonicalScenarioConfig

from ..optimization.results import OptimizationData
from ..interfaces.attribute_accessor_interface import (
    IAttributeAccessor,
    IModuleAttributeAccessor,
    create_attribute_accessor,
    create_module_attribute_accessor,
)

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """
    Handles performance evaluation for backtesting and optimization.

    This class encapsulates all evaluation-related operations that were previously
    part of the Backtester class, following the Single Responsibility Principle.
    """

    def __init__(
        self,
        global_config: Dict[str, Any],
        data_source: Any,
        strategy_manager: Any,
        attribute_accessor: Optional[IAttributeAccessor] = None,
        module_accessor: Optional[IModuleAttributeAccessor] = None,
    ) -> None:
        """
        Initialize EvaluationEngine with configuration and dependencies.

        Args:
            global_config: Global configuration dictionary
            data_source: Data source instance for fetching market data
            strategy_manager: StrategyManager instance for creating strategies
            attribute_accessor: Injected accessor for attribute access (DIP)
            module_accessor: Injected accessor for module attribute access (DIP)
        """
        self.global_config = global_config
        self.data_source = data_source
        self.strategy_manager = strategy_manager
        self.logger = logger
        # Dependency injection for attribute access (DIP)
        self._attribute_accessor = attribute_accessor or create_attribute_accessor()
        self._module_accessor = module_accessor or create_module_attribute_accessor()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("EvaluationEngine initialized")

    def evaluate_walk_forward_fast(
        self,
        trial: Any,
        scenario_config: Union[Dict[str, Any], CanonicalScenarioConfig],
        windows: list,
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
        metrics_to_optimize: list,
        is_multi_objective: bool,
    ) -> float | tuple[float, ...]:
        """
        Evaluate walk-forward optimization using fast path.

        Args:
            trial: Optimization trial object
            scenario_config: Scenario configuration (raw dict or canonical object)
            windows: List of walk-forward windows
            monthly_data: Monthly price data
            daily_data: Daily OHLC data
            rets_full: Full period returns
            metrics_to_optimize: List of metrics to optimize
            is_multi_objective: Whether this is multi-objective optimization

        Returns:
            Objective value (float for single objective, tuple for multi-objective)
        """
        # NOTE:
        # This "fast" path previously reconstructed DataFrames from raw NumPy arrays,
        # dropping the index/column labels needed for universe resolution. That made
        # WFO optimization degenerate (empty universes → NaN metrics). We keep the
        # structure but evaluate via the new-architecture backtester using the
        # original labeled DataFrames.

        from ..backtesting.strategy_backtester import StrategyBacktester
        from ..optimization.evaluator import BacktestEvaluator
        from ..canonical_config import CanonicalScenarioConfig
        from ..scenario_normalizer import ScenarioNormalizer

        if not isinstance(scenario_config, CanonicalScenarioConfig):
            logger.warning(
                "ACCIDENTAL BYPASS: Raw scenario dictionary passed to EvaluationEngine.evaluate_walk_forward_fast. "
                "All scenarios should be canonicalized at the boundary. "
                "Scenario: %s",
                scenario_config.get("name", "unnamed"),
            )
            normalizer = ScenarioNormalizer()
            canonical_config = normalizer.normalize(
                scenario=scenario_config, global_config=self.global_config
            )
        else:
            canonical_config = scenario_config

        # Create new architecture components
        strategy_backtester = StrategyBacktester(self.global_config, self.data_source)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=metrics_to_optimize,
            is_multi_objective=is_multi_objective,
        )

        # Create optimization data
        optimization_data = OptimizationData(
            monthly=monthly_data, daily=daily_data, returns=rets_full, windows=windows
        )

        # Extract parameters from trial
        parameters = dict(canonical_config.strategy_params)
        if hasattr(trial, "params") and trial.params is not None:
            parameters.update(trial.params)
        elif hasattr(trial, "user_attrs") and "parameters" in trial.user_attrs:
            parameters.update(trial.user_attrs["parameters"])

        # Evaluate using new architecture
        evaluation_result = evaluator.evaluate_parameters(
            parameters, canonical_config, optimization_data, strategy_backtester
        )

        obj = evaluation_result.objective_value
        # Normalize to match annotation float | tuple[float, ...]
        if isinstance(obj, list):
            return tuple(obj)
        return cast(float | tuple[float, ...], obj)

    def evaluate_fast(
        self,
        trial: Any,
        scenario_config: Union[Dict[str, Any], CanonicalScenarioConfig],
        windows: list,
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
        metrics_to_optimize: list,
        is_multi_objective: bool,
    ) -> tuple[float | tuple[float, ...], pd.Series]:
        """
        Evaluate trial parameters via the canonical StrategyBacktester / BacktestEvaluator path.

        Args:
            trial: Optimization trial object
            scenario_config: Scenario configuration (raw dict or canonical object)
            windows: List of walk-forward windows
            monthly_data: Monthly price data
            daily_data: Daily OHLC data
            rets_full: Full period returns
            metrics_to_optimize: List of metrics to optimize
            is_multi_objective: Whether this is multi-objective optimization

        Returns:
            Tuple of (objective_value, full_pnl_returns)
        """
        from ..backtesting.strategy_backtester import StrategyBacktester
        from ..canonical_config import CanonicalScenarioConfig
        from ..optimization.evaluator import BacktestEvaluator
        from ..scenario_normalizer import ScenarioNormalizer

        if not isinstance(scenario_config, CanonicalScenarioConfig):
            logger.warning(
                "ACCIDENTAL BYPASS: Raw scenario dictionary passed to EvaluationEngine.evaluate_fast. "
                "All scenarios should be canonicalized at the boundary. "
                "Scenario: %s",
                scenario_config.get("name", "unnamed"),
            )
            normalizer = ScenarioNormalizer()
            canonical_config = normalizer.normalize(
                scenario=scenario_config, global_config=self.global_config
            )
        else:
            canonical_config = scenario_config

        strategy_backtester = StrategyBacktester(self.global_config, self.data_source)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=metrics_to_optimize,
            is_multi_objective=is_multi_objective,
        )

        optimization_data = OptimizationData(
            monthly=monthly_data,
            daily=daily_data,
            returns=rets_full,
            windows=windows,
        )

        parameters = dict(canonical_config.strategy_params)
        if hasattr(trial, "params") and trial.params is not None:
            parameters.update(trial.params)
        elif hasattr(trial, "user_attrs") and "parameters" in trial.user_attrs:
            parameters.update(trial.user_attrs["parameters"])

        evaluation_result = evaluator.evaluate_parameters(
            parameters, canonical_config, optimization_data, strategy_backtester
        )

        objective_value_raw = evaluation_result.objective_value
        if isinstance(objective_value_raw, list):
            norm_obj: float | tuple[float, ...] = tuple(objective_value_raw)
        else:
            norm_obj = cast(float | tuple[float, ...], objective_value_raw)

        full_pnl_returns = pd.Series(dtype=float)
        if trial and hasattr(trial, "user_attrs") and "full_pnl_returns" in trial.user_attrs:
            pnl_dict = trial.user_attrs["full_pnl_returns"]
            if isinstance(pnl_dict, dict):
                full_pnl_returns = pd.Series(pnl_dict)
                full_pnl_returns.index = pd.to_datetime(full_pnl_returns.index)
        return norm_obj, full_pnl_returns

    def evaluate_trial_parameters(
        self,
        scenario_config: Union[Dict[str, Any], CanonicalScenarioConfig],
        params: Dict[str, Any],
        monthly_data: pd.DataFrame,
        daily_data_ohlc: pd.DataFrame,
        rets_full: Union[pd.DataFrame, pd.Series],
        run_scenario_func,
    ) -> Dict[str, float]:
        """
        Evaluates a single set of parameters and returns performance metrics.

        Args:
            scenario_config: Base scenario configuration (raw dict or canonical object)
            params: Parameters to evaluate
            monthly_data: Monthly price data
            daily_data_ohlc: Daily OHLC data
            rets_full: Full period returns data
            run_scenario_func: Function to run scenario evaluation

        Returns:
            Dictionary of metric names to values
        """
        from ..canonical_config import CanonicalScenarioConfig
        from ..scenario_normalizer import ScenarioNormalizer

        if not isinstance(scenario_config, CanonicalScenarioConfig):
            logger.warning(
                "ACCIDENTAL BYPASS: Raw scenario dictionary passed to EvaluationEngine.evaluate_trial_parameters. "
                "All scenarios should be canonicalized at the boundary. "
                "Scenario: %s",
                scenario_config.get("name", "unnamed"),
            )
            normalizer = ScenarioNormalizer()
            canonical_config = normalizer.normalize(
                scenario=scenario_config, global_config=self.global_config
            )
        else:
            canonical_config = scenario_config

        # Create a modified canonical config with updated parameters
        params_dict = dict(canonical_config.strategy_params)
        params_dict.update(params)

        scen_dict = canonical_config.to_dict()
        scen_dict["strategy_params"] = params_dict

        # Re-normalize
        normalizer = ScenarioNormalizer()
        temp_scenario_config = normalizer.normalize(
            scenario=scen_dict, global_config=self.global_config
        )

        returns_pkg = run_scenario_func(
            temp_scenario_config,
            monthly_data,
            daily_data_ohlc,
            rets_full if isinstance(rets_full, pd.DataFrame) else rets_full.to_frame(),
            verbose=False,
        )
        exposure_weights: pd.Series | pd.DataFrame | None = None
        if isinstance(returns_pkg, tuple):
            returns = returns_pkg[0]
            if len(returns_pkg) > 1:
                exposure_weights = returns_pkg[1]
        else:
            returns = returns_pkg

        metrics_list = self._attribute_accessor.get_attribute(self, "metrics_to_optimize", None)
        if returns is None or returns.empty:
            if metrics_list is None:
                return {}
            return {metric: 0.0 for metric in metrics_list}

        from ..reporting.performance_metrics import calculate_metrics
        from ..reporting.risk_free import build_optional_risk_free_series

        benchmark_series = daily_data_ohlc[self.global_config["benchmark"]]
        benchmark_rets = benchmark_series.pct_change(fill_method=None).fillna(0)
        rf_series = build_optional_risk_free_series(
            daily_data_ohlc, self.global_config, returns.index, temp_scenario_config
        )
        aligned_exposure = (
            exposure_weights.reindex(returns.index) if exposure_weights is not None else None
        )
        metrics = calculate_metrics(
            returns,
            benchmark_rets,
            self.global_config["benchmark"],
            risk_free_rets=rf_series,
            exposure=aligned_exposure,
        )
        if metrics_list is None:
            # Ensure precise type dict[str, float]
            return {k: float(v) for k, v in metrics.items()}
        return {metric: float(metrics.get(metric, 0.0)) for metric in metrics_list}
