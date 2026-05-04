#!/usr/bin/env python3
"""
Final comprehensive test of directional trade statistics.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TypedDict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from portfolio_backtester.reporting.performance_metrics import calculate_metrics
from portfolio_backtester.trading.trade_tracker import TradeTracker

logger = logging.getLogger(__name__)


class _Scenario(TypedDict):
    date: pd.Timestamp
    weights: dict[str, float]
    prices: dict[str, float]


def test_comprehensive_directional_functionality() -> None:
    """Exercise directional trade statistics, table, summary, and metrics integration."""
    logger.debug("Testing comprehensive directional functionality")

    tracker = TradeTracker(initial_portfolio_value=500000.0)  # $500K portfolio

    dates = pd.date_range("2023-01-01", periods=30, freq="D")

    test_scenarios: list[_Scenario] = [
        {
            "date": dates[0],
            "weights": {"AAPL": 0.2, "MSFT": 0.15},
            "prices": {"AAPL": 150, "MSFT": 300},
        },
        {
            "date": dates[5],
            "weights": {"AAPL": 0.0, "MSFT": 0.0},
            "prices": {"AAPL": 165, "MSFT": 320},
        },
        {"date": dates[6], "weights": {"GOOGL": 0.25}, "prices": {"GOOGL": 2500}},
        {
            "date": dates[10],
            "weights": {"GOOGL": 0.0},
            "prices": {"GOOGL": 2400},
        },
        {
            "date": dates[11],
            "weights": {"AMZN": -0.15, "NVDA": -0.1},
            "prices": {"AMZN": 3000, "NVDA": 400},
        },
        {
            "date": dates[15],
            "weights": {"AMZN": 0.0, "NVDA": 0.0},
            "prices": {"AMZN": 2900, "NVDA": 380},
        },
        {"date": dates[16], "weights": {"TSLA": -0.2}, "prices": {"TSLA": 200}},
        {
            "date": dates[20],
            "weights": {"TSLA": 0.0},
            "prices": {"TSLA": 220},
        },
    ]

    for scenario in test_scenarios:
        weights = pd.Series(scenario["weights"])
        prices = pd.Series(scenario["prices"])
        tracker.update_positions(scenario["date"], weights, prices, 10.0)

        for i in range(3):
            future_date = scenario["date"] + pd.DateOffset(days=i + 1)
            if future_date <= dates[-1]:
                mfe_mae_prices = prices * (1 + np.random.normal(0, 0.01, len(prices)))
                tracker.update_mfe_mae(future_date, mfe_mae_prices)

    final_prices = pd.Series(
        {"AAPL": 160, "MSFT": 310, "GOOGL": 2450, "AMZN": 2950, "NVDA": 390, "TSLA": 210}
    )
    tracker.close_all_positions(dates[-1], final_prices)

    stats = tracker.get_trade_statistics()
    table = tracker.get_trade_statistics_table()
    summary = tracker.get_directional_summary()

    assert stats["long_num_trades"] > 0, "Should have long trades"
    assert stats["short_num_trades"] > 0, "Should have short trades"
    assert stats["all_num_trades"] == stats["long_num_trades"] + stats["short_num_trades"]
    logger.debug(
        "Trade counts: All=%s, Long=%s, Short=%s",
        stats["all_num_trades"],
        stats["long_num_trades"],
        stats["short_num_trades"],
    )

    required_metrics = [
        "largest_profit",
        "largest_loss",
        "mean_profit",
        "mean_loss",
        "reward_risk_ratio",
    ]
    for direction in ["all", "long", "short"]:
        for metric in required_metrics:
            key = f"{direction}_{metric}"
            assert key in stats, f"Missing metric: {key}"

    assert not table.empty, "Table should not be empty"
    assert "Metric" in table.columns, "Table should have Metric column"
    assert "All" in table.columns, "Table should have All column"
    assert "Long" in table.columns, "Table should have Long column"
    assert "Short" in table.columns, "Table should have Short column"
    logger.debug("Table format verified (%s rows)", len(table))

    assert "All" in summary, "Summary should have All direction"
    assert "Long" in summary, "Summary should have Long direction"
    assert "Short" in summary, "Summary should have Short direction"

    returns = pd.Series(
        np.random.normal(0.001, 0.02, 100), index=pd.date_range("2023-01-01", periods=100)
    )
    benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 100), index=returns.index)

    trade_stats = tracker.get_trade_statistics()
    metrics = calculate_metrics(returns, benchmark_returns, "SPY", trade_stats=trade_stats)

    assert "Number of Trades (All)" in metrics.index
    assert "Reward/Risk Ratio (Long)" in metrics.index
    assert "Total P&L Net (Short)" in metrics.index


def main() -> int:
    """CLI entrypoint mirroring pytest assertions (for manual runs)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        test_comprehensive_directional_functionality()
    except AssertionError as e:
        logger.error("TESTS FAILED: %s", e)
        return 1
    logger.info("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
