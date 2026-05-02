import numpy as np
import pandas as pd
from unittest.mock import patch

from portfolio_backtester.strategies.builtins.portfolio.dual_momentum_lagged_portfolio_strategy import (
    DualMomentumLaggedPortfolioStrategy,
)


def _make_ohlc_multiindex(close: pd.Series, ticker: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1_000_000,
        }
    )
    df.columns = pd.MultiIndex.from_product([[ticker], df.columns], names=["Ticker", "Field"])
    return df


def test_vol_targeting_scales_gross_exposure_down() -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range(start="2019-01-01", periods=420, freq="B")

    bench_rets = rng.normal(loc=0.0002, scale=0.02, size=len(dates))
    bench_close = pd.Series(100.0 * np.cumprod(1.0 + bench_rets), index=dates)

    # Two assets that outperform the benchmark so that we get holdings.
    a_rets = bench_rets + 0.0040
    b_rets = bench_rets + 0.0030
    a_close = pd.Series(50.0 * np.cumprod(1.0 + a_rets), index=dates)
    b_close = pd.Series(70.0 * np.cumprod(1.0 + b_rets), index=dates)

    all_historical_data = pd.concat(
        [_make_ohlc_multiindex(a_close, "AAA"), _make_ohlc_multiindex(b_close, "BBB")], axis=1
    )
    benchmark_historical_data = _make_ohlc_multiindex(bench_close, "SPX")

    strategy = DualMomentumLaggedPortfolioStrategy(
        {
            "strategy_params": {
                "lookback_months": 12,
                "lag_months": 0,
                "max_holdings": 2,
                "use_200sma_filter": False,
                "min_absolute_momentum": 0.0,
                "vol_target_enabled": True,
                "target_vol_annual": 0.05,
                "vol_lookback_days": 63,
                "vol_max_gross_exposure": 1.0,
            }
        }
    )

    current_date = dates[-1]

    with patch(
        "portfolio_backtester.strategies.builtins.portfolio.dual_momentum_lagged_portfolio_strategy.get_top_weight_sp500_components",
        return_value=["AAA", "BBB"],
    ):
        weights = strategy.generate_signals(
            all_historical_data=all_historical_data,
            benchmark_historical_data=benchmark_historical_data,
            current_date=current_date,
        ).iloc[0]

    gross = float(weights.abs().sum())
    assert gross > 0.0
    assert gross < 1.0


def test_portfolio_proxy_vol_targeting_scales_gross_exposure_down() -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range(start="2019-01-01", periods=420, freq="B")

    bench_rets = rng.normal(loc=0.0002, scale=0.01, size=len(dates))
    bench_close = pd.Series(100.0 * np.cumprod(1.0 + bench_rets), index=dates)

    # High-vol assets so the portfolio proxy vol exceeds target.
    a_rets = rng.normal(loc=0.0004, scale=0.03, size=len(dates))
    b_rets = rng.normal(loc=0.0003, scale=0.025, size=len(dates))
    a_close = pd.Series(50.0 * np.cumprod(1.0 + a_rets), index=dates)
    b_close = pd.Series(70.0 * np.cumprod(1.0 + b_rets), index=dates)

    all_historical_data = pd.concat(
        [_make_ohlc_multiindex(a_close, "AAA"), _make_ohlc_multiindex(b_close, "BBB")], axis=1
    )
    benchmark_historical_data = _make_ohlc_multiindex(bench_close, "SPX")

    strategy = DualMomentumLaggedPortfolioStrategy(
        {
            "strategy_params": {
                "lookback_months": 12,
                "lag_months": 0,
                "max_holdings": 2,
                "use_200sma_filter": False,
                "min_absolute_momentum": -1.0,
                "vol_target_enabled": True,
                "vol_target_source": "portfolio_proxy",
                "target_vol_annual": 0.08,
                "vol_lookback_days": 63,
                "vol_max_gross_exposure": 1.0,
            }
        }
    )

    current_date = dates[-1]

    with patch(
        "portfolio_backtester.strategies.builtins.portfolio.dual_momentum_lagged_portfolio_strategy.get_top_weight_sp500_components",
        return_value=["AAA", "BBB"],
    ):
        weights = strategy.generate_signals(
            all_historical_data=all_historical_data,
            benchmark_historical_data=benchmark_historical_data,
            current_date=current_date,
        ).iloc[0]

    gross = float(weights.abs().sum())
    assert gross > 0.0
    assert gross < 1.0


def test_residual_momentum_ranking_prefers_alpha_over_beta() -> None:
    dates = pd.date_range(start="2020-01-01", periods=300, freq="B")
    bench_rets = np.full(len(dates), 0.0005, dtype=float)
    bench_close = pd.Series(100.0 * np.cumprod(1.0 + bench_rets), index=dates)

    # Alpha stock: market + constant drift; Beta stock: leveraged market.
    alpha_rets = bench_rets + 0.0010
    beta_rets = 2.0 * bench_rets
    alpha_close = pd.Series(50.0 * np.cumprod(1.0 + alpha_rets), index=dates)
    beta_close = pd.Series(50.0 * np.cumprod(1.0 + beta_rets), index=dates)

    asset_prices = pd.DataFrame({"ALPHA": alpha_close, "BETA": beta_close})
    benchmark_prices = pd.DataFrame({"Close": bench_close})

    strategy = DualMomentumLaggedPortfolioStrategy({"strategy_params": {}})
    current_date = dates[-1]
    params = {
        "lookback_months": 12,
        "min_absolute_momentum": 0.0,
        "ranking_method": "residual_momentum",
        "price_column_benchmark": "Close",
    }

    candidates = strategy._get_dual_momentum_candidates(
        asset_prices=asset_prices,
        benchmark_prices=benchmark_prices,
        current_date=current_date,
        params=params,
        current_holdings=set(),
    )

    assert len(candidates) >= 1
    assert candidates[0][0] == "ALPHA"


def test_exit_hysteresis_keeps_held_names_in_borderline_cases() -> None:
    dates = pd.date_range(start="2021-01-01", periods=260, freq="B")

    # Construct prices so benchmark momentum over lookback is slightly higher than the asset.
    bench_close = pd.Series(np.linspace(100.0, 110.0, len(dates)), index=dates)
    borderline_close = pd.Series(np.linspace(100.0, 109.0, len(dates)), index=dates)

    asset_prices = pd.DataFrame({"BORDER": borderline_close})
    benchmark_prices = pd.DataFrame({"Close": bench_close})

    strategy = DualMomentumLaggedPortfolioStrategy({"strategy_params": {}})
    current_date = dates[-1]

    base_params = {
        "lookback_months": 6,
        "min_absolute_momentum": 0.0,
        "ranking_method": "excess_total_return",
        "price_column_benchmark": "Close",
    }

    # Not held -> should fail the strict relative momentum check.
    candidates_no_hold = strategy._get_dual_momentum_candidates(
        asset_prices=asset_prices,
        benchmark_prices=benchmark_prices,
        current_date=current_date,
        params={**base_params, "relative_exit_buffer": 0.02, "absolute_exit_buffer": 0.0},
        current_holdings=set(),
    )
    assert candidates_no_hold == []

    # Held -> should pass with hysteresis buffer.
    candidates_held = strategy._get_dual_momentum_candidates(
        asset_prices=asset_prices,
        benchmark_prices=benchmark_prices,
        current_date=current_date,
        params={**base_params, "relative_exit_buffer": 0.02, "absolute_exit_buffer": 0.0},
        current_holdings={"BORDER"},
    )
    assert len(candidates_held) == 1
    assert candidates_held[0][0] == "BORDER"


def test_fixed_universe_does_not_apply_dynamic_sp500_filter() -> None:
    dates = pd.date_range(start="2020-01-01", periods=300, freq="B")
    bench_close = pd.Series(np.linspace(100.0, 120.0, len(dates)), index=dates)
    asset_close = pd.Series(np.linspace(100.0, 150.0, len(dates)), index=dates)

    all_historical_data = _make_ohlc_multiindex(asset_close, "QQQ")
    benchmark_historical_data = _make_ohlc_multiindex(bench_close, "SPY")
    strategy = DualMomentumLaggedPortfolioStrategy(
        {
            "universe_config": {"type": "fixed", "tickers": ["QQQ"]},
            "strategy_params": {
                "lookback_months": 6,
                "lag_months": 0,
                "max_holdings": 5,
                "use_200sma_filter": False,
                "min_absolute_momentum": 0.0,
            },
        }
    )

    with patch(
        "portfolio_backtester.strategies.builtins.portfolio.dual_momentum_lagged_portfolio_strategy.get_top_weight_sp500_components",
        return_value=["AAPL", "MSFT"],
    ) as top_components:
        weights = strategy.generate_signals(
            all_historical_data=all_historical_data,
            benchmark_historical_data=benchmark_historical_data,
            current_date=dates[-1],
        ).iloc[0]

    top_components.assert_not_called()
    assert float(weights["QQQ"]) > 0.0


def test_missing_benchmark_data_does_not_crash_sma_filter() -> None:
    dates = pd.date_range(start="2020-01-01", periods=300, freq="B")
    asset_close = pd.Series(np.linspace(100.0, 150.0, len(dates)), index=dates)

    all_historical_data = _make_ohlc_multiindex(asset_close, "QQQ")
    benchmark_historical_data = pd.DataFrame(index=dates)
    strategy = DualMomentumLaggedPortfolioStrategy(
        {
            "universe_config": {"type": "fixed", "tickers": ["QQQ"]},
            "strategy_params": {
                "lookback_months": 6,
                "lag_months": 0,
                "max_holdings": 5,
                "use_200sma_filter": True,
                "min_absolute_momentum": 0.0,
            },
        }
    )

    weights = strategy.generate_signals(
        all_historical_data=all_historical_data,
        benchmark_historical_data=benchmark_historical_data,
        current_date=dates[-1],
    ).iloc[0]

    assert float(weights["QQQ"]) == 0.0


def test_named_universe_does_not_apply_sp500_topn_filter() -> None:
    dates = pd.date_range(start="2020-01-01", periods=300, freq="B")
    bench_close = pd.Series(np.linspace(100.0, 120.0, len(dates)), index=dates)
    asset_close = pd.Series(np.linspace(100.0, 150.0, len(dates)), index=dates)

    all_historical_data = _make_ohlc_multiindex(asset_close, "QQQ")
    benchmark_historical_data = _make_ohlc_multiindex(bench_close, "SPY")
    strategy = DualMomentumLaggedPortfolioStrategy(
        {
            "universe_config": {"type": "named", "universe_name": "custom"},
            "strategy_params": {
                "lookback_months": 6,
                "lag_months": 0,
                "max_holdings": 5,
                "use_200sma_filter": False,
                "min_absolute_momentum": 0.0,
            },
        }
    )

    with patch(
        "portfolio_backtester.strategies.builtins.portfolio.dual_momentum_lagged_portfolio_strategy.get_top_weight_sp500_components",
        return_value=["AAPL", "MSFT"],
    ) as top_components:
        weights = strategy.generate_signals(
            all_historical_data=all_historical_data,
            benchmark_historical_data=benchmark_historical_data,
            current_date=dates[-1],
        ).iloc[0]

    top_components.assert_not_called()
    assert float(weights["QQQ"]) > 0.0


def test_momentum_skip_months_excludes_recent_reversal() -> None:
    dates = pd.date_range(start="2020-01-31", periods=15, freq="ME")
    current_date = dates[-1]
    asset_close = pd.Series(100.0, index=dates)
    benchmark_close = pd.Series(100.0, index=dates)

    formation_end = current_date - pd.DateOffset(months=1)
    lookback_start = formation_end - pd.DateOffset(months=12)
    asset_close.loc[asset_close.index <= lookback_start] = 100.0
    asset_close.loc[(asset_close.index > lookback_start) & (asset_close.index <= formation_end)] = (
        150.0
    )
    asset_close.iloc[-1] = 75.0
    benchmark_close.loc[benchmark_close.index <= lookback_start] = 100.0
    benchmark_close.loc[
        (benchmark_close.index > lookback_start) & (benchmark_close.index <= formation_end)
    ] = 110.0
    benchmark_close.iloc[-1] = 110.0

    asset_prices = pd.DataFrame({"REVERSAL": asset_close})
    benchmark_prices = pd.DataFrame({"Close": benchmark_close})
    strategy = DualMomentumLaggedPortfolioStrategy({"strategy_params": {}})
    base_params = {
        "lookback_months": 12,
        "min_absolute_momentum": 0.0,
        "ranking_method": "excess_total_return",
        "price_column_benchmark": "Close",
    }

    no_skip_candidates = strategy._get_dual_momentum_candidates(
        asset_prices=asset_prices,
        benchmark_prices=benchmark_prices,
        current_date=current_date,
        params={**base_params, "momentum_skip_months": 0},
        current_holdings=set(),
    )
    skip_candidates = strategy._get_dual_momentum_candidates(
        asset_prices=asset_prices,
        benchmark_prices=benchmark_prices,
        current_date=current_date,
        params={**base_params, "momentum_skip_months": 1},
        current_holdings=set(),
    )

    assert no_skip_candidates == []
    assert len(skip_candidates) == 1
    assert skip_candidates[0][0] == "REVERSAL"
