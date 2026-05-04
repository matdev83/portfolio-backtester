from __future__ import annotations

import pandas as pd
from frozendict import frozendict

from portfolio_backtester.backtester_logic.strategy_logic import generate_signals
from portfolio_backtester.canonical_config import CanonicalScenarioConfig
from portfolio_backtester.strategies._core.target_generation import StrategyContext
from portfolio_backtester.strategies.user.signal.hello_world_signal_strategy import (
    HelloWorldSignalStrategy,
)


class _ForbiddenSignalMatrixHelloWorld(HelloWorldSignalStrategy):
    def generate_signal_matrix(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("generate_signal_matrix must not be invoked by StrategyLogic")


def test_generate_signals_uses_generate_target_weights_not_signal_matrix() -> None:
    idx = pd.date_range("2024-06-03", periods=12, freq="B")
    universe_tickers = ["AAA", "BBB"]
    benchmark_ticker = "AAA"
    ohlc = pd.DataFrame(
        {
            "AAA": pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float),
            "BBB": pd.Series(range(200, 200 + len(idx)), index=idx, dtype=float),
        }
    )
    canonical = CanonicalScenarioConfig.from_dict(
        {
            "name": "tw_api",
            "strategy": "HelloWorldSignalStrategy",
            "benchmark_ticker": benchmark_ticker,
            "strategy_params": {"leverage": 1.0},
            "timing_config": {
                "mode": "signal_based",
                "scan_frequency": "D",
                "min_holding_period": 1,
            },
        }
    )
    strat = HelloWorldSignalStrategy(canonical)
    forbidden = _ForbiddenSignalMatrixHelloWorld(canonical)

    dense = strat.generate_target_weights(
        StrategyContext.from_standard_inputs(
            asset_data=ohlc[universe_tickers],
            benchmark_data=ohlc[[benchmark_ticker]],
            non_universe_data=pd.DataFrame(),
            rebalance_dates=pd.DatetimeIndex(idx),
            universe_tickers=universe_tickers,
            benchmark_ticker=benchmark_ticker,
            wfo_start_date=None,
            wfo_end_date=None,
            use_sparse_nan_for_inactive_rows=True,
        )
    )

    assert dense is not None
    got = generate_signals(
        forbidden,
        canonical,
        ohlc,
        universe_tickers,
        benchmark_ticker,
        lambda: False,
        global_config=frozendict({"feature_flags": frozendict()}),
    )

    aligned = dense.reindex(index=got.index, columns=universe_tickers)
    pd.testing.assert_frame_equal(
        got.fillna(0.0),
        aligned.fillna(0.0),
        rtol=0.0,
        atol=1e-12,
    )
