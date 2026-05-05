"""Benchmark ``BacktestEvaluator.evaluate_parameters`` (optimizer trial path).

Uses the same scenario/data/WFO shortening knobs as ``scripts/profile_evaluate_parameters.py``
but records wall times instead of cProfile.

Cold vs warm:
  - ``cold``: time the first ``evaluate_parameters`` call after data load.
  - ``warm``: one untimed ``evaluate_parameters`` warmup, then time a second call.
  - ``both``: timed first call (cold) and timed second call (warm).

Example::

    .venv/Scripts/python.exe scripts/benchmark_optimizer_objective.py \\
        --scenario-filename config/scenarios/builtins/signal/dummy_signal_strategy/default.yaml \\
        --trial-start-date 2018-01-01 --trial-end-date 2019-06-30 \\
        --profile-train-months 6 --profile-test-months 3 \\
        --timing-mode both --mdmp-cache-only
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from portfolio_backtester.backtester_logic.data_fetcher import DataFetcher
from portfolio_backtester.backtester_logic.strategy_manager import StrategyManager
from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
from portfolio_backtester.canonical_config import CanonicalScenarioConfig
from portfolio_backtester.config_loader import (
    OPTIMIZER_PARAMETER_DEFAULTS,
    load_config,
    load_scenario_from_file,
)
from portfolio_backtester.interfaces import create_cache_manager, create_data_source
from portfolio_backtester.optimization.evaluator import BacktestEvaluator
from portfolio_backtester.optimization.results import OptimizationData
from portfolio_backtester.scenario_normalizer import ScenarioNormalizer

logger = logging.getLogger(__name__)

_PEP_MODULE: Any | None = None

TimingModeName = Literal["cold", "warm", "both"]


def _get_profile_evaluate_parameters_module() -> Any:
    global _PEP_MODULE
    if _PEP_MODULE is None:
        pep_path = Path(__file__).resolve().parent / "profile_evaluate_parameters.py"
        spec = importlib.util.spec_from_file_location(
            "_portfolio_bt_profile_eval_params_helpers",
            pep_path,
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PEP_MODULE = mod
    return _PEP_MODULE


@dataclass(frozen=True)
class OptimizerObjectiveBenchmarkResult:
    scenario_file: str
    timing_mode: str
    train_window_months: int
    test_window_months: int
    trial_start_date: Optional[str]
    trial_end_date: Optional[str]
    data_fetch_s: float
    evaluate_parameters_cold_s: Optional[float]
    evaluate_parameters_warm_s: Optional[float]
    n_windows: int
    metrics: list[str]
    trial_params: dict[str, Any]
    objective_value_cold: Optional[float]
    objective_value_warm: Optional[float]


def optimizer_objective_benchmark_result_to_dict(
    rec: OptimizerObjectiveBenchmarkResult,
) -> dict[str, Any]:
    sections: dict[str, float] = {"data_fetch_s": rec.data_fetch_s}
    if rec.evaluate_parameters_cold_s is not None:
        sections["evaluate_parameters_cold_s"] = rec.evaluate_parameters_cold_s
    if rec.evaluate_parameters_warm_s is not None:
        sections["evaluate_parameters_warm_s"] = rec.evaluate_parameters_warm_s
    out: dict[str, Any] = {
        "scenario_file": rec.scenario_file,
        "timing_mode": rec.timing_mode,
        "train_window_months": rec.train_window_months,
        "test_window_months": rec.test_window_months,
        "trial_start_date": rec.trial_start_date,
        "trial_end_date": rec.trial_end_date,
        "n_windows": rec.n_windows,
        "metrics": list(rec.metrics),
        "trial_params": dict(rec.trial_params),
        "objective_value_cold": rec.objective_value_cold,
        "objective_value_warm": rec.objective_value_warm,
        "sections": sections,
    }
    return out


def execute_timed_evaluations(
    evaluator: BacktestEvaluator,
    *,
    parameters: dict[str, Any],
    scenario_config: CanonicalScenarioConfig,
    data: OptimizationData,
    backtester: StrategyBacktester,
    timing_mode: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    if timing_mode not in ("cold", "warm", "both"):
        raise ValueError(f"timing_mode must be cold|warm|both, got {timing_mode!r}")

    timings: dict[str, float] = {}
    meta: dict[str, Any] = {}

    if timing_mode == "cold":
        t0 = time.perf_counter()
        r1 = evaluator.evaluate_parameters(
            parameters=parameters,
            scenario_config=scenario_config,
            data=data,
            backtester=backtester,
        )
        timings["evaluate_parameters_cold_s"] = time.perf_counter() - t0
        meta["objective_value_cold"] = r1.objective_value
        return timings, meta

    if timing_mode == "warm":
        evaluator.evaluate_parameters(
            parameters=parameters,
            scenario_config=scenario_config,
            data=data,
            backtester=backtester,
        )
        t1 = time.perf_counter()
        r2 = evaluator.evaluate_parameters(
            parameters=parameters,
            scenario_config=scenario_config,
            data=data,
            backtester=backtester,
        )
        timings["evaluate_parameters_warm_s"] = time.perf_counter() - t1
        meta["objective_value_warm"] = r2.objective_value
        return timings, meta

    t0 = time.perf_counter()
    r_cold = evaluator.evaluate_parameters(
        parameters=parameters,
        scenario_config=scenario_config,
        data=data,
        backtester=backtester,
    )
    timings["evaluate_parameters_cold_s"] = time.perf_counter() - t0
    meta["objective_value_cold"] = r_cold.objective_value

    t1 = time.perf_counter()
    r_warm = evaluator.evaluate_parameters(
        parameters=parameters,
        scenario_config=scenario_config,
        data=data,
        backtester=backtester,
    )
    timings["evaluate_parameters_warm_s"] = time.perf_counter() - t1
    meta["objective_value_warm"] = r_warm.objective_value
    return timings, meta


def run_optimizer_objective_benchmark(
    *,
    scenario_path: Path,
    trial_start_date: Optional[str] = None,
    trial_end_date: Optional[str] = None,
    profile_train_months: int = 12,
    profile_test_months: int = 6,
    timing_mode: TimingModeName = "both",
    mdmp_cache_only: bool = False,
    feature_flag_pairs: Sequence[tuple[str, Any]] | None = None,
    rng_seed: int = 42,
) -> OptimizerObjectiveBenchmarkResult:
    pep = _get_profile_evaluate_parameters_module()
    load_config()
    from portfolio_backtester.config_loader import GLOBAL_CONFIG

    gc = copy.deepcopy(GLOBAL_CONFIG)
    gc["optimizer_parameter_defaults"] = OPTIMIZER_PARAMETER_DEFAULTS
    pep.apply_feature_flag_pairs_to_global_config(gc, list(feature_flag_pairs or []))
    if mdmp_cache_only:
        dsc = gc.setdefault("data_source_config", {})
        dsc["cache_only"] = True
        logger.info("MDMP cache-only enabled.")

    if not scenario_path.is_file():
        raise FileNotFoundError(str(scenario_path.resolve()))

    raw = load_scenario_from_file(scenario_path)
    trial = copy.deepcopy(raw)
    if trial_start_date:
        trial["start_date"] = trial_start_date
    if trial_end_date:
        trial["end_date"] = trial_end_date
    trial["train_window_months"] = profile_train_months
    trial["test_window_months"] = profile_test_months

    normalizer = ScenarioNormalizer()
    canonical = normalizer.normalize(scenario=trial, global_config=gc)

    if canonical.start_date is not None:
        gc["start_date"] = str(pd.Timestamp(canonical.start_date).date())
    if canonical.end_date is not None:
        gc["end_date"] = str(pd.Timestamp(canonical.end_date).date())

    data_source = create_data_source(gc)
    data_cache = create_cache_manager()
    fetcher = DataFetcher(global_config=gc, data_source=data_source)
    strategy_manager = StrategyManager()

    logger.info("Fetching data for scenario %r …", canonical.name)
    t_fetch0 = time.perf_counter()
    daily_ohlc, monthly_data, daily_closes = fetcher.prepare_data_for_backtesting(
        [canonical], strategy_manager.get_strategy
    )
    data_fetch_s = time.perf_counter() - t_fetch0
    logger.info("Data fetch done in %.2fs", data_fetch_s)

    rets_full = data_cache.get_cached_returns(daily_closes, "full_period_returns")
    if isinstance(rets_full, pd.Series):
        rets_df = rets_full.to_frame()
    elif isinstance(rets_full, pd.DataFrame):
        rets_df = rets_full
    else:
        rets_df = pd.DataFrame(rets_full)

    rng = np.random.default_rng(rng_seed)
    opt_data = pep._build_optimization_data(
        monthly_data=monthly_data,
        daily_data=daily_ohlc,
        rets_full=rets_df,
        canonical=canonical,
        global_config=gc,
        rng=rng,
    )

    metrics, multi = pep._metrics_from_canonical(canonical)
    evaluator = BacktestEvaluator(
        metrics_to_optimize=metrics,
        is_multi_objective=multi,
        n_jobs=1,
        enable_parallel_optimization=False,
    )
    backtester = StrategyBacktester(global_config=gc, data_source=None, data_cache=data_cache)
    params = pep._mid_trial_params(list(canonical.optimize) if canonical.optimize else None)

    eval_times, eval_meta = execute_timed_evaluations(
        evaluator,
        parameters=params,
        scenario_config=canonical,
        data=opt_data,
        backtester=backtester,
        timing_mode=timing_mode,
    )

    cold_s = eval_times.get("evaluate_parameters_cold_s")
    warm_s = eval_times.get("evaluate_parameters_warm_s")

    return OptimizerObjectiveBenchmarkResult(
        scenario_file=str(scenario_path),
        timing_mode=timing_mode,
        train_window_months=profile_train_months,
        test_window_months=profile_test_months,
        trial_start_date=str(trial.get("start_date")) if trial.get("start_date") else None,
        trial_end_date=str(trial.get("end_date")) if trial.get("end_date") else None,
        data_fetch_s=data_fetch_s,
        evaluate_parameters_cold_s=cold_s,
        evaluate_parameters_warm_s=warm_s,
        n_windows=len(opt_data.windows),
        metrics=list(metrics),
        trial_params=dict(params),
        objective_value_cold=eval_meta.get("objective_value_cold"),
        objective_value_warm=eval_meta.get("objective_value_warm"),
    )


def main(argv: Optional[list[str]] = None) -> int:
    pep = _get_profile_evaluate_parameters_module()
    parser = argparse.ArgumentParser(
        description="Wall-clock benchmark for BacktestEvaluator.evaluate_parameters.",
    )
    parser.add_argument(
        "--scenario-filename",
        type=Path,
        required=True,
        help="Scenario YAML path (same as backtester --scenario-filename).",
    )
    parser.add_argument(
        "--trial-start-date",
        type=str,
        default=None,
        help="Override scenario start_date for a shorter benchmark window.",
    )
    parser.add_argument(
        "--trial-end-date",
        type=str,
        default=None,
        help="Override scenario end_date.",
    )
    parser.add_argument(
        "--profile-train-months",
        type=int,
        default=12,
        help="Train window months (matches profile_evaluate_parameters).",
    )
    parser.add_argument(
        "--profile-test-months",
        type=int,
        default=6,
        help="Test window months.",
    )
    parser.add_argument(
        "--timing-mode",
        choices=("cold", "warm", "both"),
        default="both",
        help="cold=first eval only; warm=warmup+second eval; both=timed first+second.",
    )
    parser.add_argument(
        "--mdmp-cache-only",
        action="store_true",
        help="Set data_source_config.cache_only (same as CLI --mdmp-cache-only).",
    )
    parser.add_argument(
        "--feature-flag",
        action="append",
        default=[],
        type=pep.parse_feature_flag_assignment,
        metavar="KEY=VALUE",
        help="Global feature flag (repeatable), same as profile_evaluate_parameters.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON benchmark payload to this path.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    result = run_optimizer_objective_benchmark(
        scenario_path=args.scenario_filename,
        trial_start_date=args.trial_start_date,
        trial_end_date=args.trial_end_date,
        profile_train_months=args.profile_train_months,
        profile_test_months=args.profile_test_months,
        timing_mode=args.timing_mode,
        mdmp_cache_only=bool(args.mdmp_cache_only),
        feature_flag_pairs=args.feature_flag or [],
    )
    payload = optimizer_objective_benchmark_result_to_dict(result)
    text = json.dumps(payload, indent=2 if args.output else None)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
