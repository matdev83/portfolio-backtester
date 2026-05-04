"""Documents interaction between meta execution mode and trade execution timing."""

from __future__ import annotations

from portfolio_backtester.backtester_logic.meta_execution import (
    MetaExecutionMode,
    portfolio_execution_mode_for_strategy,
)
from portfolio_backtester.strategies.builtins.meta.simple_meta_strategy import SimpleMetaStrategy


def test_document_meta_trade_aggregation_coexists_with_next_bar_open_timing_flag() -> None:
    """Meta strategies report ``next_bar_open`` like other strategies but returns use TRADE_AGGREGATION.

    Canonical portfolio simulation applies ``next_bar_open`` via ``build_portfolio_simulation_input``
    and the Numba kernel. Meta strategies routed through ``MetaExecutionMode.TRADE_AGGREGATION`` rebuild
    returns from intercepted sub-strategy trades and close-based valuation paths unless/until unified
    with the canonical engine (see ``docs/simulation_execution_paths.md``). This test locks the
    *coexistence* of those two facts—it does not assert open-priced fills on the aggregation ledger.
    """

    meta_cfg = {
        "initial_capital": 1_000_000,
        "timing_config": {
            "mode": "time_based",
            "rebalance_frequency": "M",
            "trade_execution_timing": "next_bar_open",
        },
        "allocations": [
            {
                "strategy_id": "momentum",
                "strategy_class": "CalmarMomentumPortfolioStrategy",
                "strategy_params": {
                    "rolling_window": 3,
                    "num_holdings": 1,
                    "price_column_asset": "Close",
                    "price_column_benchmark": "Close",
                    "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
                },
                "weight": 1.0,
            },
        ],
    }
    meta = SimpleMetaStrategy(meta_cfg)
    assert meta.get_trade_execution_timing() == "next_bar_open"
    assert portfolio_execution_mode_for_strategy(meta) is MetaExecutionMode.TRADE_AGGREGATION
