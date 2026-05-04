"""Walk-forward re-optimization helpers for :class:`OptimizationOrchestrator`."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, List

import pandas as pd

from ..canonical_config import CanonicalScenarioConfig
from ..optimization.market_data_panel import MarketDataPanel
from ..optimization.results import OptimizationData, OptimizationResult
from ..optimization.window_bounds import build_window_bounds
from .optimization_orchestrator_support import normalize_early_stop_patience

if TYPE_CHECKING:
    from .optimization_orchestrator import OptimizationOrchestrator

logger = logging.getLogger(__name__)


def run_reoptimized_wfo(
    orch: OptimizationOrchestrator,
    scenario_config: CanonicalScenarioConfig,
    monthly_data: pd.DataFrame,
    daily_data: pd.DataFrame,
    rets_full: pd.DataFrame,
    windows: List[Any],
    metrics_to_optimize: List[str],
    is_multi_objective: bool,
    optimization_config: Dict[str, Any],
    optimizer_type: str,
    optimizer_args: Any,
) -> OptimizationResult:
    """Run true walk-forward: re-optimize each window, stitch OOS returns."""
    from ..backtesting.strategy_backtester import StrategyBacktester
    from ..optimization.evaluator import BacktestEvaluator, _lookup_metric
    from ..optimization.factory import create_parameter_generator
    from ..optimization.orchestrator_factory import create_orchestrator
    from ..optimization.wfo_window import WFOWindow
    from ..reporting.performance_metrics import calculate_metrics
    from ..reporting.risk_free import build_optional_risk_free_series
    from ..scenario_normalizer import ScenarioNormalizer

    if not windows:
        raise ValueError("No WFO windows available for re-optimization.")

    strategy_backtester = StrategyBacktester(
        global_config=orch.global_config, data_source=orch.data_source
    )

    window_results = []
    window_params: List[Dict[str, Any]] = []
    stitched_returns_list: List[pd.Series] = []
    total_evaluations = 0

    for idx, window in enumerate(windows, start=1):
        train_window = WFOWindow(
            train_start=window.train_start,
            train_end=window.train_end,
            test_start=window.train_start,
            test_end=window.train_end,
            evaluation_frequency=getattr(window, "evaluation_frequency", "M"),
            strategy_name=getattr(window, "strategy_name", None),
        )

        inner_panel = MarketDataPanel.from_daily_ohlc_and_returns(daily_data, rets_full)
        optimization_data = OptimizationData(
            monthly=monthly_data,
            daily=daily_data,
            returns=rets_full,
            windows=[train_window],
            market_data=inner_panel,
            daily_np=inner_panel.daily_np,
            returns_np=inner_panel.returns_np,
            daily_index_np=inner_panel.row_index_naive_datetime64(),
            tickers_list=list(inner_panel.tickers),
            window_bounds=[build_window_bounds(inner_panel.daily_index_naive, train_window)],
        )

        parameter_generator = create_parameter_generator(
            optimizer_type=optimizer_type, random_state=orch.rng
        )
        evaluator_n_jobs = orch._attribute_accessor.get_attribute(optimizer_args, "n_jobs", 1)
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
                n_jobs=int(evaluator_n_jobs),
            )

        window_opt_config = dict(optimization_config)
        window_suffix = f"wfo_{idx}_{window.train_end:%Y%m%d}"
        if window_opt_config.get("study_name"):
            window_opt_config["study_name"] = f"{window_opt_config['study_name']}_{window_suffix}"
        else:
            window_opt_config["study_name"] = f"{scenario_config.name}_{window_suffix}"

        orchestrator = create_orchestrator(
            optimizer_type=optimizer_type,
            parameter_generator=parameter_generator,
            evaluator=evaluator,
            n_jobs=orch._attribute_accessor.get_attribute(optimizer_args, "n_jobs", -1),
            joblib_batch_size=orch._attribute_accessor.get_attribute(
                optimizer_args, "joblib_batch_size", None
            ),
            joblib_pre_dispatch=orch._attribute_accessor.get_attribute(
                optimizer_args, "joblib_pre_dispatch", None
            ),
            timeout=orch._attribute_accessor.get_attribute(
                optimizer_args, "optuna_timeout_sec", None
            ),
            early_stop_patience=normalize_early_stop_patience(
                orch._attribute_accessor.get_attribute(optimizer_args, "early_stop_patience", 10)
            ),
            enable_adaptive_batch_sizing=orch._attribute_accessor.get_attribute(
                optimizer_args, "enable_adaptive_batch_sizing", True
            ),
            enable_hybrid_parallelism=orch._attribute_accessor.get_attribute(
                optimizer_args, "enable_hybrid_parallelism", True
            ),
            enable_incremental_evaluation=orch._attribute_accessor.get_attribute(
                optimizer_args, "enable_incremental_evaluation", True
            ),
            enable_gpu_acceleration=orch._attribute_accessor.get_attribute(
                optimizer_args, "enable_gpu_acceleration", True
            ),
        )

        optimization_result = orchestrator.optimize(
            scenario_config=scenario_config,
            optimization_config=window_opt_config,
            data=optimization_data,
            backtester=strategy_backtester,  # type: ignore[arg-type]
        )
        total_evaluations += optimization_result.n_evaluations

        best_params = optimization_result.best_parameters or {}
        window_params.append(best_params)

        params_dict = dict(scenario_config.strategy_params)
        params_dict.update(best_params)
        scen_dict = scenario_config.to_dict()
        scen_dict["strategy_params"] = params_dict

        normalizer = ScenarioNormalizer()
        eval_config = normalizer.normalize(scenario=scen_dict, global_config=orch.global_config)

        test_result = strategy_backtester.evaluate_window(
            eval_config, window, monthly_data, daily_data, rets_full
        )
        window_results.append(test_result)
        if (
            isinstance(test_result.window_returns, pd.Series)
            and not test_result.window_returns.empty
        ):
            stitched_returns_list.append(test_result.window_returns)

    if stitched_returns_list:
        stitched_returns = pd.concat(stitched_returns_list).sort_index()
        stitched_returns = stitched_returns[~stitched_returns.index.duplicated(keep="first")]
    else:
        stitched_returns = pd.Series(dtype=float)

    benchmark_ticker = orch.global_config.get("benchmark", "SPY")
    if isinstance(scenario_config, CanonicalScenarioConfig):
        benchmark_ticker = scenario_config.benchmark_ticker or benchmark_ticker
    elif isinstance(scenario_config, Mapping):
        benchmark_ticker = scenario_config.get("benchmark_ticker", benchmark_ticker)
    benchmark_returns = pd.Series(0.0, index=stitched_returns.index)
    if not stitched_returns.empty and benchmark_ticker:
        try:
            if isinstance(daily_data.columns, pd.MultiIndex):
                if "Close" in daily_data.columns.get_level_values("Field"):
                    daily_close = daily_data.xs("Close", level="Field", axis=1)
                else:
                    daily_close = daily_data
            else:
                daily_close = daily_data

            if benchmark_ticker in daily_close.columns:
                benchmark_returns = (
                    daily_close[benchmark_ticker]
                    .pct_change(fill_method=None)
                    .reindex(stitched_returns.index)
                    .fillna(0.0)
                )
        except (KeyError, TypeError, ValueError, AttributeError, IndexError):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Benchmark extraction failed for stitched returns; using zeros.",
                    exc_info=True,
                )
            benchmark_returns = pd.Series(0.0, index=stitched_returns.index)

    rf_opt = build_optional_risk_free_series(
        daily_data, orch.global_config, stitched_returns.index, scenario_config
    )
    metrics_series = calculate_metrics(
        stitched_returns, benchmark_returns, benchmark_ticker, risk_free_rets=rf_opt
    )
    metrics = {k: float(v) if not pd.isna(v) else float("nan") for k, v in metrics_series.items()}

    objective_values = [_lookup_metric(metrics, metric, -1e9) for metric in metrics_to_optimize]
    best_value: float | list[float]
    if is_multi_objective:
        best_value = objective_values
    else:
        best_value = float(objective_values[0]) if objective_values else -1e9

    consensus_params = orch._calculate_consensus_params(window_params)

    return OptimizationResult(
        best_parameters=consensus_params,
        best_value=best_value,
        n_evaluations=total_evaluations,
        optimization_history=[],
        best_trial=None,
        wfo_mode="reoptimize",
        wfo_window_params=window_params,
        wfo_window_results=window_results,
        stitched_returns=stitched_returns,
    )
