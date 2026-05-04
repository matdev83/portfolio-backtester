"""One-off universe ablation for fixed-parameter dual-momentum sleeves.

For each ETF in the scenario's fixed universe, runs a full backtest with that ticker
removed (leave-one-out). Ranks tickers by change in **Sortino** vs the baseline
universe (matches typical ``optimization_metric: Sortino`` usage).

Does **not** re-optimize strategy params — only the ticker list changes.

Example::

    .venv\\Scripts\\python.exe scripts/country_etf_universe_ablation_sortino.py \\
        --scenario-filename config/scenarios/builtins/portfolio/dual_momentum_lagged_portfolio_strategy/country_etf_universe_spy_benchmark_no_sma_lag0.yaml \\
        --mdmp-cache-only --random-seed 42
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
import portfolio_backtester.config_loader as config_loader
from portfolio_backtester.config_loader import load_config, load_scenario_from_file
from portfolio_backtester.data_sources.base_data_source import BaseDataSource
from portfolio_backtester.backtester_logic.data_fetcher import DataFetcher
from portfolio_backtester.backtester_logic.strategy_manager import StrategyManager
from portfolio_backtester.interfaces import create_cache_manager, create_data_source
from portfolio_backtester.scenario_normalizer import ScenarioNormalizer

logger = logging.getLogger(__name__)

METRIC = "Sortino"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-out ETF ablation ranked by Sortino.")
    parser.add_argument(
        "--scenario-filename",
        type=str,
        default=(
            "config/scenarios/builtins/portfolio/dual_momentum_lagged_portfolio_strategy/"
            "country_etf_universe_spy_benchmark_no_sma_lag0.yaml"
        ),
        help="Base scenario YAML (fixed params; universe edited per run).",
    )
    parser.add_argument(
        "--mdmp-cache-only",
        action="store_true",
        help="Pass-through: MDMP reads parquet cache only (same as main backtester CLI).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed forwarded to global/backtester RNG conventions where applicable.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="How many 'best to remove' rows to print (highest Sortino lift when dropped).",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="",
        help="Optional path to write a CSV summary (UTF-8).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Root log level; portfolio_backtester can be noisy at INFO.",
    )
    return parser.parse_args()


def _scenario_dict_with_universe(
    base: dict[str, Any],
    *,
    name_suffix: str,
    tickers: list[str],
) -> dict[str, Any]:
    out = copy.deepcopy(base)
    out["name"] = f"{base['name']}{name_suffix}"
    ud = out.setdefault("universe_definition", {})
    if ud.get("type") != "fixed":
        raise ValueError("This script supports universe_definition.type == 'fixed' only.")
    ud["tickers"] = list(tickers)
    return out


def run() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    scenario_path = Path(args.scenario_filename)
    if not scenario_path.is_file():
        logger.error("Scenario file not found: %s", scenario_path.resolve())
        return 1

    load_config()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    gc: dict[str, Any] = config_loader.GLOBAL_CONFIG
    if args.mdmp_cache_only:
        dsc = gc.setdefault("data_source_config", {})
        dsc["cache_only"] = True
        logging.getLogger("portfolio_backtester").info("MDMP cache-only enabled.")

    raw = load_scenario_from_file(scenario_path)
    normalizer = ScenarioNormalizer()
    base_canonical = normalizer.normalize(scenario=raw, global_config=gc, source=str(scenario_path))
    base_dict = base_canonical.to_dict()
    fixed_tickers: list[str] = list(base_dict.get("universe_definition", {}).get("tickers") or [])
    if not fixed_tickers:
        logger.error("No universe_definition.tickers in scenario (expected fixed universe).")
        return 1

    data_source = create_data_source(gc)
    cache = create_cache_manager()
    strategy_manager = StrategyManager()
    fetcher = DataFetcher(global_config=gc, data_source=data_source)

    daily_ohlc, monthly_closes, daily_closes = fetcher.prepare_data_for_backtesting(
        [base_canonical],
        strategy_manager.get_strategy,
    )
    rets_full = cache.get_cached_returns(daily_closes, "full_period_returns")
    if isinstance(rets_full, pd.Series):
        rets_full = rets_full.to_frame()
    elif not isinstance(rets_full, pd.DataFrame):
        rets_full = pd.DataFrame(rets_full)

    class _DummySource(BaseDataSource):
        def get_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
            return daily_ohlc

    dummy = _DummySource()
    bt = StrategyBacktester(gc, dummy)

    def eval_one(label: str, tickers: list[str]) -> dict[str, float]:
        scen_dict = _scenario_dict_with_universe(base_dict, name_suffix=label, tickers=tickers)
        cfg = normalizer.normalize(scenario=scen_dict, global_config=gc, source=str(scenario_path))
        res = bt.backtest_strategy(cfg, monthly_closes, daily_ohlc, rets_full, track_trades=False)
        if res is None:
            return {}
        raw = res.metrics
        if raw is None:
            return {}
        md = raw.to_dict() if isinstance(raw, pd.Series) else dict(raw)
        if not md:
            return {}
        return {
            METRIC: float(md.get(METRIC, float("nan"))),
            "Ann. Return": float(md.get("Ann. Return", float("nan"))),
            "Max Drawdown": float(md.get("Max Drawdown", float("nan"))),
            "Calmar": float(md.get("Calmar", float("nan"))),
        }

    logger.warning("Baseline (%d names)...", len(fixed_tickers))
    baseline = eval_one("_baseline", fixed_tickers)
    base_sortino = baseline.get(METRIC, float("nan"))
    logger.warning("Baseline Sortino=%.6f", base_sortino)

    rows: list[dict[str, Any]] = []
    for t in fixed_tickers:
        reduced = [x for x in fixed_tickers if x != t]
        m = eval_one(f"_minus_{t}", reduced)
        s = m.get(METRIC, float("nan"))
        rows.append(
            {
                "removed": t,
                METRIC: s,
                f"delta_{METRIC}": (
                    s - base_sortino if pd.notna(s) and pd.notna(base_sortino) else float("nan")
                ),
                "Ann. Return": m.get("Ann. Return"),
                "Max Drawdown": m.get("Max Drawdown"),
                "Calmar": m.get("Calmar"),
            }
        )
        logger.warning("minus %-5s Sortino=%.6f (Δ=%+.6f)", t, s, s - base_sortino)

    df = pd.DataFrame(rows)
    df = df.sort_values(f"delta_{METRIC}", ascending=False, na_position="last")

    print("\n=== Baseline ===")
    print(f"universe_size={len(fixed_tickers)}  {METRIC}={base_sortino:.6f}")
    print(f"\n=== Top {args.top} removal candidates (largest Sortino lift when dropped) ===")
    cols = ["removed", METRIC, f"delta_{METRIC}", "Ann. Return", "Max Drawdown"]
    print(df[cols].head(args.top).to_string(index=False))
    worst = df.tail(min(args.top, len(df))).iloc[::-1]
    print(f"\n=== Bottom {min(args.top, len(df))} (dropping these hurts Sortino most) ===")
    print(worst[cols].to_string(index=False))

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_row = pd.DataFrame(
            [
                {
                    "removed": "(baseline)",
                    METRIC: base_sortino,
                    f"delta_{METRIC}": 0.0,
                    "Ann. Return": baseline.get("Ann. Return"),
                    "Max Drawdown": baseline.get("Max Drawdown"),
                    "Calmar": baseline.get("Calmar"),
                }
            ]
        )
        pd.concat([baseline_row, df], ignore_index=True).to_csv(out_path, index=False)
        logger.warning("Wrote %s", out_path.resolve())

    return 0


if __name__ == "__main__":
    sys.exit(run())
