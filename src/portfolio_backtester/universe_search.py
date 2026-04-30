from __future__ import annotations

import argparse
import copy
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

import portfolio_backtester.config_loader as config_loader
from portfolio_backtester.backtester_logic.data_fetcher import DataFetcher
from portfolio_backtester.backtester_logic.reporting import _benchmark_returns
from portfolio_backtester.backtester_logic.strategy_logic import generate_signals, size_positions
from portfolio_backtester.backtester_logic.strategy_manager import StrategyManager
from portfolio_backtester.interfaces import create_cache_manager, create_data_source
from portfolio_backtester.reporting.report_directory_utils import (
    create_report_directory,
    generate_content_hash,
)
from portfolio_backtester.portfolio.rebalancing import rebalance
from portfolio_backtester.reporting.performance_metrics import calculate_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseSearchResult:
    subset_id: int
    tickers: list[str]
    metrics: pd.Series
    score: float


def normalize_candidates(candidates: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, str):
            raise ValueError(f"Universe candidates must be strings, got {type(item)}")
        ticker = item.strip().upper()
        if not ticker:
            continue
        if ticker in seen:
            continue
        seen.add(ticker)
        normalized.append(ticker)
    if not normalized:
        raise ValueError("Universe candidate list is empty after normalization.")
    return normalized


def sample_subsets(
    candidates: Sequence[str],
    subset_size: int,
    n_samples: int,
    rng: np.random.Generator,
) -> list[list[str]]:
    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if subset_size > len(candidates):
        raise ValueError("subset_size cannot exceed number of candidates.")

    if subset_size == len(candidates):
        return [list(candidates)]

    max_unique = int(math.comb(len(candidates), subset_size))
    target = min(n_samples, max_unique)
    subsets: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    max_attempts = target * 25

    while len(subsets) < target and attempts < max_attempts:
        attempts += 1
        idx = rng.choice(len(candidates), size=subset_size, replace=False)
        subset = [candidates[i] for i in sorted(idx)]
        key = tuple(subset)
        if key in seen:
            continue
        seen.add(key)
        subsets.append(subset)

    if len(subsets) < target:
        logger.warning(
            "Sampled only %d/%d unique subsets after %d attempts.",
            len(subsets),
            target,
            attempts,
        )
    return subsets


def build_weights_daily(
    signals: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    rebalance_frequency: str,
) -> pd.DataFrame:
    weights_monthly = rebalance(signals, rebalance_frequency)
    weights_daily = weights_monthly.reindex(price_index, method="ffill")
    return weights_daily.fillna(0.0)


def compute_symbol_contributions(
    weights_daily: pd.DataFrame, returns_daily: pd.DataFrame
) -> pd.DataFrame:
    weights_shifted = weights_daily.shift(1).fillna(0.0)
    aligned_returns = (
        returns_daily.reindex(weights_shifted.index).reindex(columns=weights_shifted.columns)
    ).fillna(0.0)
    contributions = weights_shifted * aligned_returns
    total = contributions.sum(axis=0)
    mean = contributions.mean(axis=0)
    positive_ratio = (contributions > 0.0).sum(axis=0) / max(len(contributions), 1)
    out = pd.DataFrame(
        {
            "total_contribution": total,
            "mean_contribution": mean,
            "positive_days_ratio": positive_ratio,
        }
    )
    return out.sort_values("total_contribution", ascending=False)


def compute_subset_returns(
    weights_daily: pd.DataFrame,
    returns_daily: pd.DataFrame,
    subset: Sequence[str],
    normalize_weights: bool,
) -> pd.Series:
    subset_list = list(subset)
    subset_weights = weights_daily.reindex(columns=subset_list).fillna(0.0)
    subset_returns = returns_daily.reindex(columns=subset_list).fillna(0.0)
    weights_shifted = subset_weights.shift(1).fillna(0.0)

    if normalize_weights:
        row_sum = weights_shifted.sum(axis=1)
        weights_shifted = weights_shifted.div(row_sum.where(row_sum > 0), axis=0).fillna(0.0)

    return (weights_shifted * subset_returns).sum(axis=1)


def evaluate_subsets(
    *,
    subsets: Iterable[list[str]],
    weights_daily: pd.DataFrame,
    returns_daily: pd.DataFrame,
    benchmark_returns: pd.Series,
    benchmark_ticker: str,
    metric_key: str,
    normalize_weights: bool,
) -> list[UniverseSearchResult]:
    results: list[UniverseSearchResult] = []
    for idx, subset in enumerate(subsets, start=1):
        subset_returns = compute_subset_returns(
            weights_daily, returns_daily, subset, normalize_weights
        )
        bench = benchmark_returns.reindex(subset_returns.index).fillna(0.0)
        metrics = calculate_metrics(subset_returns, bench, benchmark_ticker, name="Strategy")
        score = metrics.get(metric_key, np.nan)
        results.append(
            UniverseSearchResult(
                subset_id=idx,
                tickers=subset,
                metrics=metrics,
                score=float(score) if pd.notna(score) else float("nan"),
            )
        )
    return results


def _resolve_candidates(args: argparse.Namespace, scenario: dict[str, Any]) -> list[str]:
    if args.candidates:
        return normalize_candidates(args.candidates.split(","))
    if isinstance(scenario.get("universe"), list):
        return normalize_candidates(scenario["universe"])
    universe_config = scenario.get("universe_config")
    if isinstance(universe_config, dict) and universe_config.get("type") == "fixed":
        return normalize_candidates(universe_config.get("tickers", []))
    raise ValueError(
        "Universe candidates not provided. Use --candidates or a fixed universe_config."
    )


def run_universe_search(args: argparse.Namespace) -> Path:
    config_loader.load_config()
    global_config = config_loader.GLOBAL_CONFIG
    scenario_path = Path(args.scenario_filename)
    scenario = config_loader.load_scenario_from_file(scenario_path)

    candidates = _resolve_candidates(args, scenario)
    benchmark_ticker = str(global_config.get("benchmark", "SPY"))

    scenario_for_fetch = copy.deepcopy(scenario)
    scenario_for_fetch["universe"] = candidates

    strategy_manager = StrategyManager()
    data_source = create_data_source(global_config)
    data_fetcher = DataFetcher(global_config=global_config, data_source=data_source)
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig

    daily_ohlc, monthly_data, daily_closes = data_fetcher.prepare_data_for_backtesting(
        [CanonicalScenarioConfig.from_dict(scenario_for_fetch)], strategy_manager.get_strategy
    )

    data_cache = create_cache_manager()
    rets_full = data_cache.get_cached_returns(daily_closes, "full_period_returns")
    rets_daily = rets_full if isinstance(rets_full, pd.DataFrame) else pd.DataFrame(rets_full)

    strategy = strategy_manager.get_strategy(scenario["strategy"], scenario["strategy_params"])
    signals = generate_signals(
        strategy,
        scenario_for_fetch,
        daily_ohlc,
        candidates,
        benchmark_ticker,
        has_timed_out=lambda: False,
        global_config=global_config,
    )
    sized_signals = size_positions(
        signals,
        scenario,
        price_data_daily_ohlc=daily_ohlc,
        universe_tickers=candidates,
        benchmark_ticker=benchmark_ticker,
        global_config=global_config,
        strategy=strategy,
    )

    rebalance_frequency = scenario.get("timing_config", {}).get("rebalance_frequency", "ME")
    weights_daily = build_weights_daily(
        sized_signals, pd.DatetimeIndex(daily_ohlc.index), rebalance_frequency
    )
    weights_daily = weights_daily.reindex(columns=candidates).fillna(0.0)
    returns_daily = rets_daily.reindex(index=daily_ohlc.index, columns=candidates).fillna(0.0)
    benchmark_returns = _benchmark_returns(daily_ohlc, benchmark_ticker)

    baseline_returns = compute_subset_returns(
        weights_daily, returns_daily, candidates, normalize_weights=False
    )
    baseline_metrics = calculate_metrics(
        baseline_returns,
        benchmark_returns.reindex(baseline_returns.index).fillna(0.0),
        benchmark_ticker,
    )

    contributions = compute_symbol_contributions(weights_daily, returns_daily)

    rng = np.random.default_rng(int(args.random_seed))
    subsets = sample_subsets(
        candidates=candidates,
        subset_size=int(args.subset_size),
        n_samples=int(args.n_samples),
        rng=rng,
    )
    results = evaluate_subsets(
        subsets=subsets,
        weights_daily=weights_daily,
        returns_daily=returns_daily,
        benchmark_returns=benchmark_returns,
        benchmark_ticker=benchmark_ticker,
        metric_key=args.metric,
        normalize_weights=bool(args.normalize_weights),
    )

    results_sorted = sorted(results, key=lambda r: (-np.inf if np.isnan(r.score) else -r.score))
    top_results = results_sorted[: int(args.top_n)]

    # Generate content hash for version tracking
    content_hash = None
    strategy = strategy_manager.get_strategy(scenario["strategy"], scenario["strategy_params"])
    if strategy:
        try:
            content_hash = generate_content_hash(
                strategy_class=type(strategy), config_file_path=scenario_path
            )
            if content_hash:
                logger.info(f"Generated content hash: {content_hash}")
        except Exception as e:
            logger.warning(f"Could not generate content hash: {e}")

    # Create report directory with hash-based structure
    report_base_dir = Path(__file__).resolve().parents[2] / "data" / "reports"
    output_dir = create_report_directory(report_base_dir, "universe_search", content_hash)

    contributions.to_csv(output_dir / "symbol_contributions.csv")
    pd.DataFrame([baseline_metrics]).to_csv(output_dir / "baseline_metrics.csv")

    rows: list[dict[str, Any]] = []
    for res in results_sorted:
        row = {"subset_id": res.subset_id, "tickers": ",".join(res.tickers), "score": res.score}
        for key, value in res.metrics.items():
            row[str(key)] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "subset_results.csv", index=False)

    logger.info("Universe search complete. Output: %s", output_dir)
    for res in top_results:
        logger.info(
            "Top subset %d score=%.6f size=%d tickers=%s",
            res.subset_id,
            res.score,
            len(res.tickers),
            ",".join(res.tickers),
        )
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Approximate universe search (no retraining).")
    parser.add_argument(
        "--scenario-filename",
        required=True,
        help="Scenario YAML file to use as base.",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Comma-separated candidate tickers (overrides scenario universe).",
    )
    parser.add_argument("--subset-size", type=int, default=12)
    parser.add_argument("--n-samples", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="Sortino")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--normalize-weights",
        action="store_true",
        help="Renormalize weights after dropping symbols (best-case approximation).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_universe_search(args)


if __name__ == "__main__":
    main()
