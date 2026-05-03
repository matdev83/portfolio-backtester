"""Detailed BacktestRunner check for the discovered SPY intramonth baseline.

The detailed run disables the optional generate_signal_matrix hook so BacktestRunner
uses per-date generate_signals without changing strategy parameters.
"""

from __future__ import annotations

# ruff: noqa: E402

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from portfolio_backtester import config_loader
from portfolio_backtester.backtester_logic.backtest_runner import BacktestRunner
from portfolio_backtester.backtester_logic.data_fetcher import DataFetcher
from portfolio_backtester.backtester_logic.strategy_manager import StrategyManager
from portfolio_backtester.interfaces import create_cache_manager, create_data_source
from portfolio_backtester.reporting.fast_objective_metrics import calculate_optimizer_metrics_fast
from portfolio_backtester.scenario_normalizer import ScenarioNormalizer

PORT = 100_000.0
CPS = 0.005
CMIN = 1.0
CMAX = 0.005
SLIP = 0.5
START = "2005-01-01"
END = "2024-12-31"
BITMASK = 3958
ENTRY_BY_MONTH = {1: 19, 2: 7, 3: 16, 4: 1, 5: 17, 6: 20, 7: 7, 8: 17, 9: 7, 10: 19, 11: 14, 12: 14}
HOLD_BY_MONTH = {1: 13, 2: 5, 3: 10, 4: 12, 5: 5, 6: 17, 7: 9, 8: 14, 9: 5, 10: 7, 11: 6, 12: 5}
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def strategy_params() -> dict[str, Any]:
    params: dict[str, Any] = {
        "direction": "long",
        "month_local_seasonal_windows": False,
        "entry_day": 1,
        "hold_days": max(HOLD_BY_MONTH.values()),
        "entry_day_by_month": ENTRY_BY_MONTH,
        "hold_days_by_month": HOLD_BY_MONTH,
        "simple_high_low_stop_loss": False,
        "simple_high_low_take_profit": False,
        "stop_loss_atr_multiple": 0.0,
        "take_profit_atr_multiple": 0.0,
    }
    for month in range(1, 13):
        params[f"trade_month_{month}"] = bool((BITMASK >> (month - 1)) & 1)
    return params


def scenario(*, force_daily_generation: bool) -> dict[str, Any]:
    return {
        "name": (
            "spy_intramonth_cost_aware_best_detailed"
            if force_daily_generation
            else "spy_intramonth_cost_aware_best_fast"
        ),
        "strategy": "SeasonalSignalStrategy",
        "start_date": START,
        "end_date": END,
        "benchmark_ticker": "SPY",
        "rebalance_frequency": "ME",
        "position_sizer": "equal_weight",
        "timing_config": {
            "mode": "signal_based",
            "scan_frequency": "D",
            "min_holding_period": 1,
            "trade_execution_timing": "bar_close",
        },
        "universe_config": {"type": "fixed", "tickers": ["SPY"]},
        "train_window_months": 36,
        "test_window_months": 12,
        "extras": {"is_wfo": False, "risk_free_metrics_enabled": False},
        "optimization_targets": [{"name": "Sortino", "direction": "maximize"}],
        "strategy_params": strategy_params(),
    }


def run_one(
    *,
    force_daily_generation: bool,
    shared_inputs: tuple[Any, Any, Any, Any, Any] | None = None,
) -> tuple[pd.Series, dict[str, float], tuple[Any, Any, Any, Any, Any]]:
    config_loader.load_config()
    gc = config_loader.GLOBAL_CONFIG
    gc.setdefault("data_source_config", {})["cache_only"] = True
    gc.update(
        {
            "portfolio_value": PORT,
            "commission_per_share": CPS,
            "commission_min_per_order": CMIN,
            "commission_max_percent_of_trade": CMAX,
            "slippage_bps": SLIP,
        }
    )
    strategy_manager = StrategyManager()
    data_cache = create_cache_manager()
    runner = BacktestRunner(gc, data_cache, strategy_manager, lambda: False)
    normalizer = ScenarioNormalizer()
    canonical = normalizer.normalize(
        scenario=scenario(force_daily_generation=force_daily_generation),
        global_config=gc,
    )

    if shared_inputs is None:
        fetcher = DataFetcher(gc, create_data_source(gc))
        daily_ohlc, monthly_data, daily_closes = fetcher.prepare_data_for_backtesting(
            [canonical], strategy_manager.get_strategy
        )
        cached_rets = data_cache.get_cached_returns(daily_closes, "full_period_returns")
        rets_full = (
            cached_rets.to_frame()
            if isinstance(cached_rets, pd.Series)
            else pd.DataFrame(cached_rets)
        )
        shared_inputs = (daily_ohlc, monthly_data, daily_closes, rets_full, data_cache)
    else:
        daily_ohlc, monthly_data, daily_closes, rets_full, _ = shared_inputs

    if force_daily_generation:
        from portfolio_backtester.strategies.builtins.signal.seasonal_signal_strategy import (
            SeasonalSignalStrategy,
        )

        original_matrix = SeasonalSignalStrategy.generate_signal_matrix

        def _disable_signal_matrix(self: SeasonalSignalStrategy, *args: Any, **kwargs: Any) -> None:
            return None

        SeasonalSignalStrategy.generate_signal_matrix = _disable_signal_matrix  # type: ignore[method-assign]
        try:
            returns = runner.run_scenario(
                canonical, monthly_data, daily_ohlc, rets_full, verbose=False
            )
        finally:
            SeasonalSignalStrategy.generate_signal_matrix = original_matrix  # type: ignore[method-assign]
    else:
        returns = runner.run_scenario(canonical, monthly_data, daily_ohlc, rets_full, verbose=False)
    if returns is None:
        raise RuntimeError("Backtest returned no returns")
    period_mask = _period_filter(returns.index)
    returns = returns.loc[period_mask.to_numpy()].astype(float)
    close = daily_ohlc.xs("Close", level="Field", axis=1)["SPY"].loc[returns.index]
    benchmark = close.pct_change(fill_method=None).fillna(0.0)
    metrics_raw = calculate_optimizer_metrics_fast(
        returns.astype(float),
        benchmark.astype(float),
        "SPY",
        risk_free_rets=None,
        requested_metrics={
            "Sortino",
            "Sharpe",
            "Calmar",
            "Total Return",
            "Max Drawdown",
            "Ann. Return",
        },
    )
    metrics = {str(k): float(v) for k, v in metrics_raw.items()}
    return returns, metrics, shared_inputs


def _period_filter(index: pd.Index) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        cmp = pd.DatetimeIndex(
            [pd.Timestamp(ts).tz_convert(idx.tz).replace(tzinfo=None) for ts in idx]
        )
    else:
        cmp = idx
    return pd.Series((cmp >= pd.Timestamp(START)) & (cmp <= pd.Timestamp(END)), index=index)


def trade_stats(returns: pd.Series) -> dict[str, Any]:
    active = returns[returns.abs() > 1e-12]
    return {
        "non_zero_return_days": int(active.shape[0]),
        "first_return": str(returns.index.min()),
        "last_return": str(returns.index.max()),
        "mean_daily_return": float(returns.mean()),
        "down_days": int((returns < 0).sum()),
        "up_days": int((returns > 0).sum()),
    }


def main() -> int:
    out = _REPO / "dev" / "spy_intramonth_cost_aware_best_detailed_backtest"
    out.mkdir(parents=True, exist_ok=True)

    fast_returns, fast_metrics, shared_inputs = run_one(force_daily_generation=False)
    detailed_returns, detailed_metrics, _ = run_one(
        force_daily_generation=True,
        shared_inputs=shared_inputs,
    )

    common = fast_returns.index.intersection(detailed_returns.index)
    diff = (
        fast_returns.loc[common].astype(float) - detailed_returns.loc[common].astype(float)
    ).abs()
    summary = {
        "bitmask": BITMASK,
        "months": [MONTH_NAMES[m - 1] for m in range(1, 13) if (BITMASK >> (m - 1)) & 1],
        "excluded_months": [
            MONTH_NAMES[m - 1] for m in range(1, 13) if not ((BITMASK >> (m - 1)) & 1)
        ],
        "entry_by_month": ENTRY_BY_MONTH,
        "hold_by_month": HOLD_BY_MONTH,
        "costs": {
            "portfolio_value": PORT,
            "commission_per_share": CPS,
            "commission_min_per_order": CMIN,
            "commission_max_percent_of_trade": CMAX,
            "slippage_bps": SLIP,
        },
        "fast_matrix_metrics": fast_metrics,
        "detailed_daily_metrics": detailed_metrics,
        "return_diff": {
            "common_rows": int(common.shape[0]),
            "max_abs_diff": float(diff.max()),
            "sum_abs_diff": float(diff.sum()),
            "rows_diff_gt_1e_12": int((diff > 1e-12).sum()),
        },
        "fast_stats": trade_stats(fast_returns),
        "detailed_stats": trade_stats(detailed_returns),
        "detail_forced_by": "SeasonalSignalStrategy.generate_signal_matrix temporarily disabled; strategy params unchanged.",
    }

    fast_returns.to_csv(out / "fast_matrix_returns.csv", header=["return"])
    detailed_returns.to_csv(out / "detailed_daily_returns.csv", header=["return"])
    (out / "detailed_backtest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"WROTE={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
