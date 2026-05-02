"""Profile one ``BacktestEvaluator.evaluate_parameters`` call (optimizer trial path).

Loads a scenario YAML, optionally shortens the date range and WFO train/test months so
data fetch and one evaluation stay small, then runs ``cProfile`` around
``evaluate_parameters`` and writes ``pstats`` summaries (cumulative and tottime).

Example::

    .venv/Scripts/python.exe scripts/profile_evaluate_parameters.py \\
        --scenario-filename config/scenarios/builtins/signal/mmm_qs_swing_nasdaq_signal_strategy/optimize_sl_tp_sortino.yaml \\
        --trial-start-date 2013-01-01 --trial-end-date 2015-06-30 \\
        --profile-train-months 12 --profile-test-months 6 \\
        --profile-out scripts/profile_evaluate_parameters_mmm_sample.txt \\
        --feature-flag signal_cache=true --feature-flag fast_optimizer_metrics=false \\
        --mdmp-cache-only
"""

from __future__ import annotations

import argparse
import cProfile
import copy
import io
import logging
import pstats
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Tuple

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
from portfolio_backtester.optimization.market_data_panel import MarketDataPanel
from portfolio_backtester.optimization.results import OptimizationData
from portfolio_backtester.optimization.window_bounds import build_window_bounds
from portfolio_backtester.optimization.wfo_window import WFOWindow
from portfolio_backtester.scenario_normalizer import ScenarioNormalizer
from portfolio_backtester.utils import generate_enhanced_wfo_windows

logger = logging.getLogger(__name__)


def parse_feature_flag_assignment(raw: str) -> Tuple[str, Any]:
    """Parse ``key=value`` CLI feature flag; bools from true/false/1/0/yes/no (case-insensitive)."""
    s = raw.strip()
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"Feature flag must be key=value, got {raw!r}")
    key, val = s.split("=", 1)
    key = key.strip()
    val_s = val.strip()
    low = val_s.lower()
    if low in ("true", "1", "yes"):
        return key, True
    if low in ("false", "0", "no"):
        return key, False
    return key, val_s


def apply_feature_flag_pairs_to_global_config(
    gc: dict[str, Any], pairs: Sequence[Tuple[str, Any]]
) -> None:
    """Merge ``(key, value)`` pairs into ``gc['feature_flags']``."""
    ff = gc.setdefault("feature_flags", {})
    for k, v in pairs:
        ff[k] = v


def format_profile_header_metadata(gc: dict[str, Any]) -> str:
    """Format profiling header lines for feature flags and optional signal cache stats."""
    lines = [f"feature_flags={dict(gc.get('feature_flags') or {})}"]
    cache = gc.get("_signal_matrix_cache")
    if cache is not None and hasattr(cache, "stats"):
        lines.append(f"signal_matrix_cache_stats={cache.stats()}")
    return "\n".join(lines) + "\n"


def _mid_trial_params(
    optimize_specs: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Pick a single interior point in each optimized dimension for profiling."""
    if not optimize_specs:
        return {}
    out: dict[str, Any] = {}
    for spec in optimize_specs:
        name = spec["parameter"]
        ptype = spec.get("type")
        if not ptype:
            mv, xv = spec.get("min_value"), spec.get("max_value")
            if isinstance(mv, int) and isinstance(xv, int):
                ptype = "int"
            else:
                ptype = "float"
        if ptype == "categorical":
            choices = spec.get("choices")
            if isinstance(choices, Sequence) and choices:
                out[name] = choices[len(choices) // 2]
            continue

        if "min_value" not in spec:
            # Unknown shape: skip rather than blocking profile smoke runs.
            continue

        low = spec["min_value"]
        high = spec.get("max_value", low)
        if ptype == "int":
            out[name] = int((int(low) + int(high)) // 2)
        elif ptype == "float":
            out[name] = float((float(low) + float(high)) / 2.0)
        else:
            # Unknown: skip for profiling
            continue
    return out


def _metrics_from_canonical(canonical: CanonicalScenarioConfig) -> tuple[list[str], bool]:
    targets = canonical.extras.get("optimization_targets", [])
    if targets:
        names = [t["name"] for t in targets]
        return names, len(names) > 1
    return [canonical.optimization_metric or "Calmar"], False


def _build_optimization_data(
    *,
    monthly_data: pd.DataFrame,
    daily_data: pd.DataFrame,
    rets_full: pd.DataFrame,
    canonical: CanonicalScenarioConfig,
    global_config: dict[str, Any],
    rng: np.random.Generator,
) -> OptimizationData:
    monthly_idx = pd.DatetimeIndex(pd.to_datetime(monthly_data.index))
    windows = generate_enhanced_wfo_windows(monthly_idx, canonical, global_config, rng)
    if not windows:
        dmin = pd.to_datetime(daily_data.index.min())
        dmax = pd.to_datetime(daily_data.index.max())
        logger.warning(
            "WFO generator returned no windows; using single span [%s, %s] from daily data.",
            dmin,
            dmax,
        )
        windows = [
            WFOWindow(
                train_start=dmin,
                train_end=dmax,
                test_start=dmin,
                test_end=dmax,
            )
        ]

    panel = MarketDataPanel.from_daily_ohlc_and_returns(daily_data, rets_full)
    wb = [build_window_bounds(panel.daily_index_naive, w) for w in windows]
    optimization_data = OptimizationData(
        monthly=monthly_data,
        daily=daily_data,
        returns=rets_full,
        windows=windows,
        market_data=panel,
        daily_np=panel.daily_np,
        returns_np=panel.returns_np,
        daily_index_np=panel.row_index_naive_datetime64(),
        tickers_list=list(panel.tickers),
        window_bounds=wb,
    )
    return optimization_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="cProfile one BacktestEvaluator.evaluate_parameters call."
    )
    parser.add_argument(
        "--scenario-filename",
        type=Path,
        required=True,
        help="Path to scenario YAML (same as backtester --scenario-filename).",
    )
    parser.add_argument(
        "--trial-start-date",
        type=str,
        default=None,
        help="Override scenario start_date for a shorter profiling window.",
    )
    parser.add_argument(
        "--trial-end-date",
        type=str,
        default=None,
        help="Override scenario end_date for a shorter profiling window.",
    )
    parser.add_argument(
        "--profile-train-months",
        type=int,
        default=12,
        help="Train window months for profiling (short ranges need smaller windows).",
    )
    parser.add_argument(
        "--profile-test-months",
        type=int,
        default=6,
        help="Test window months for profiling.",
    )
    parser.add_argument(
        "--profile-out",
        type=Path,
        default=Path("scripts/profile_evaluate_parameters_last_run.txt"),
        help="Text file for pstats output.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=40,
        help="Number of lines per sort key in the report.",
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
        type=parse_feature_flag_assignment,
        metavar="KEY=VALUE",
        help="Set global feature flag (repeatable), e.g. --feature-flag signal_cache=true.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    load_config()
    from portfolio_backtester.config_loader import GLOBAL_CONFIG

    gc = copy.deepcopy(GLOBAL_CONFIG)
    gc["optimizer_parameter_defaults"] = OPTIMIZER_PARAMETER_DEFAULTS
    apply_feature_flag_pairs_to_global_config(gc, args.feature_flag or [])
    if args.mdmp_cache_only:
        dsc = gc.setdefault("data_source_config", {})
        dsc["cache_only"] = True
        logger.info("MDMP cache-only enabled.")

    scenario_path = args.scenario_filename
    if not scenario_path.is_file():
        raise SystemExit(f"Scenario file not found: {scenario_path.resolve()}")

    raw = load_scenario_from_file(scenario_path)
    trial = copy.deepcopy(raw)
    if args.trial_start_date:
        trial["start_date"] = args.trial_start_date
    if args.trial_end_date:
        trial["end_date"] = args.trial_end_date
    trial["train_window_months"] = args.profile_train_months
    trial["test_window_months"] = args.profile_test_months

    normalizer = ScenarioNormalizer()
    canonical = normalizer.normalize(scenario=trial, global_config=gc)

    # DataFetcher.fetch_daily_data uses global_config end_date (not scenario alone).
    if canonical.start_date is not None:
        gc["start_date"] = str(pd.Timestamp(canonical.start_date).date())
    if canonical.end_date is not None:
        gc["end_date"] = str(pd.Timestamp(canonical.end_date).date())

    data_source = create_data_source(gc)
    data_cache = create_cache_manager()
    fetcher = DataFetcher(global_config=gc, data_source=data_source)
    strategy_manager = StrategyManager()

    logger.info("Fetching data for scenario %r …", canonical.name)
    t0 = time.perf_counter()
    daily_ohlc, monthly_data, daily_closes = fetcher.prepare_data_for_backtesting(
        [canonical], strategy_manager.get_strategy
    )
    logger.info("Data fetch done in %.2fs", time.perf_counter() - t0)

    rets_full = data_cache.get_cached_returns(daily_closes, "full_period_returns")
    if isinstance(rets_full, pd.Series):
        rets_df = rets_full.to_frame()
    elif isinstance(rets_full, pd.DataFrame):
        rets_df = rets_full
    else:
        rets_df = pd.DataFrame(rets_full)

    rng = np.random.default_rng(42)
    opt_data = _build_optimization_data(
        monthly_data=monthly_data,
        daily_data=daily_ohlc,
        rets_full=rets_df,
        canonical=canonical,
        global_config=gc,
        rng=rng,
    )

    metrics, multi = _metrics_from_canonical(canonical)
    evaluator = BacktestEvaluator(
        metrics_to_optimize=metrics,
        is_multi_objective=multi,
        n_jobs=1,
        enable_parallel_optimization=False,
    )
    backtester = StrategyBacktester(global_config=gc, data_source=None, data_cache=data_cache)
    params = _mid_trial_params(list(canonical.optimize) if canonical.optimize else None)
    logger.info("Profiling evaluate_parameters with params: %s", params)

    pr = cProfile.Profile()
    pr.enable()
    t_eval = time.perf_counter()
    result = evaluator.evaluate_parameters(
        parameters=params,
        scenario_config=canonical,
        data=opt_data,
        backtester=backtester,
    )
    eval_s = time.perf_counter() - t_eval
    pr.disable()
    logger.info(
        "evaluate_parameters finished in %.2fs; objective=%s",
        eval_s,
        result.objective_value,
    )

    out_path = args.profile_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = io.StringIO()
    header.write(f"scenario_file={scenario_path}\n")
    header.write(f"trial_start={trial.get('start_date')} trial_end={trial.get('end_date')}\n")
    header.write(
        f"train_window_months={args.profile_train_months} "
        f"test_window_months={args.profile_test_months}\n"
    )
    header.write(f"n_windows={len(opt_data.windows)} metrics={metrics}\n")
    header.write(f"trial_params={params}\n")
    header.write(f"evaluate_parameters_wall_s={eval_s:.4f}\n")
    header.write(format_profile_header_metadata(gc))
    header.write("\n")

    buf = io.StringIO()
    buf.write(header.getvalue())
    buf.write(f"=== Sorted by cumulative time (top {args.top}) ===\n")
    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(
        args.top
    )
    buf.write(f"\n=== Sorted by tottime (top {args.top}) ===\n")
    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats(args.top)

    text = buf.getvalue()
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
