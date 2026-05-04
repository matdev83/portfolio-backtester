"""StrategyBacktester passes realized signed weights into calculate_metrics."""

from unittest.mock import MagicMock, patch

import pandas as pd

from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester


def test_backtest_strategy_threads_signed_weights_to_calculate_metrics():
    dates = pd.bdate_range("2023-01-03", periods=5)
    daily = pd.DataFrame(
        {"AAA": [100.0, 101.0, 102.0, 103.0, 104.0], "BMK": [50.0, 51.0, 52.0, 53.0, 54.0]},
        index=dates,
    )
    rets_full = daily.pct_change(fill_method=None).fillna(0.0)

    gc = {"benchmark": "BMK", "portfolio_value": 100_000.0}
    bt = StrategyBacktester(gc, data_source=MagicMock())

    strat_cfg = {
        "name": "exposure_thread_smoke",
        "strategy": "Dummy",
        "strategy_params": {},
        "benchmark_ticker": "BMK",
        "universe": [("AAA", 1.0)],
        "timing_config": {"mode": "signal_based"},
    }

    pr = pd.Series(0.001, index=dates)
    signed = pd.DataFrame({"AAA": [1.0] * len(dates)}, index=dates)

    strat_mock = MagicMock()
    strat_mock.get_universe.return_value = [("AAA", 1.0)]

    with (
        patch.object(bt, "_get_strategy", return_value=strat_mock),
        patch(
            "portfolio_backtester.backtesting.strategy_backtester.generate_signals",
            return_value=pd.DataFrame({"AAA": [1.0]}, index=[dates[0]]),
        ),
        patch(
            "portfolio_backtester.backtesting.strategy_backtester.size_positions",
            return_value=pd.DataFrame({"AAA": [1.0]}, index=[dates[0]]),
        ),
        patch(
            "portfolio_backtester.backtesting.strategy_backtester.calculate_portfolio_returns",
            return_value=(pr, None, signed),
        ),
        patch("portfolio_backtester.backtesting.strategy_backtester.calculate_metrics") as mock_cm,
    ):
        mock_cm.return_value = pd.Series({"Total Return": 0.01})
        bt.backtest_strategy(
            strat_cfg, monthly_data=daily[["AAA"]], daily_data=daily, rets_full=rets_full
        )

    mock_cm.assert_called_once()
    got = mock_cm.call_args.kwargs.get("exposure")
    pd.testing.assert_frame_equal(got, signed)
