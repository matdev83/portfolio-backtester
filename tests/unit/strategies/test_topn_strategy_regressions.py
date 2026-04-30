from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pandas.testing as pdt
from frozendict import frozendict

from portfolio_backtester.canonical_config import CanonicalScenarioConfig
from portfolio_backtester.strategies.builtins.portfolio.calmar_momentum_portfolio_strategy import (
    CalmarMomentumPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.dual_momentum_lagged_portfolio_strategy import (
    DualMomentumLaggedPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.sharpe_momentum_portfolio_strategy import (
    SharpeMomentumPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.portfolio.sortino_momentum_portfolio_strategy import (
    SortinoMomentumPortfolioStrategy,
)


def _make_ohlc_multiindex(close: pd.Series, ticker: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1_000_000,
        }
    )
    frame.columns = pd.MultiIndex.from_product([[ticker], frame.columns], names=["Ticker", "Field"])
    return frame


def test_sortino_strategy_uses_last_available_trading_day_for_missing_rebalance_date() -> None:
    strategy = SortinoMomentumPortfolioStrategy({"strategy_params": {"rolling_window": 3}})
    mock_scores = pd.DataFrame(
        {"StockA": [1.2, 1.8], "StockB": [0.7, 0.4]},
        index=pd.to_datetime(["2020-01-30", "2020-01-31"]),
    )
    asset_prices = pd.DataFrame(
        {"StockA": [100.0, 101.0], "StockB": [100.0, 99.0]},
        index=pd.to_datetime(["2020-01-30", "2020-01-31"]),
    )

    with patch.object(strategy.sortino_feature, "compute", return_value=mock_scores):
        scores = strategy._calculate_scores(asset_prices, pd.Timestamp("2020-02-01"))

    expected = pd.Series({"StockA": 1.8, "StockB": 0.4})
    pdt.assert_series_equal(scores, expected, check_names=False)


def test_calmar_strategy_uses_last_available_trading_day_for_missing_rebalance_date() -> None:
    strategy = CalmarMomentumPortfolioStrategy({"strategy_params": {"rolling_window": 6}})
    mock_scores = pd.DataFrame(
        {"StockA": [0.5, 0.6], "StockB": [0.3, 0.4]},
        index=pd.to_datetime(["2020-01-30", "2020-01-31"]),
    )
    asset_prices = pd.DataFrame(
        {"StockA": [100.0, 101.0], "StockB": [100.0, 99.0]},
        index=pd.to_datetime(["2020-01-30", "2020-01-31"]),
    )

    with patch.object(strategy.calmar_feature, "compute", return_value=mock_scores):
        scores = strategy._calculate_scores(asset_prices, pd.Timestamp("2020-02-01"))

    expected = pd.Series({"StockA": 0.6, "StockB": 0.4})
    pdt.assert_series_equal(scores, expected, check_names=False)


def test_dual_momentum_dynamic_universe_respects_configured_top_n() -> None:
    dates = pd.bdate_range("2023-01-02", periods=5)
    aaa = pd.Series(np.linspace(100.0, 104.0, len(dates)), index=dates)
    bbb = pd.Series(np.linspace(90.0, 94.0, len(dates)), index=dates)
    spx = pd.Series(np.linspace(4000.0, 4010.0, len(dates)), index=dates)

    all_historical_data = pd.concat(
        [_make_ohlc_multiindex(aaa, "AAA"), _make_ohlc_multiindex(bbb, "BBB")],
        axis=1,
    )
    benchmark_historical_data = _make_ohlc_multiindex(spx, "SPX")

    strategy = DualMomentumLaggedPortfolioStrategy(
        {
            "universe_config": {
                "type": "method",
                "method_name": "get_top_weight_sp500_components",
                "n_holdings": 20,
                "exact": True,
            },
            "strategy_params": {
                "lookback_months": 6,
                "lag_months": 0,
                "max_holdings": 2,
                "use_200sma_filter": False,
                "min_absolute_momentum": -1.0,
            },
        }
    )

    risk_off_signal_generator = MagicMock()
    risk_off_signal_generator.generate_risk_off_signal.return_value = False

    with (
        patch.object(strategy, "validate_data_sufficiency", return_value=(True, "")),
        patch.object(strategy, "filter_universe_by_data_availability", return_value=["AAA", "BBB"]),
        patch.object(strategy, "_get_dual_momentum_candidates", return_value=[]),
        patch.object(
            strategy, "get_risk_off_signal_generator", return_value=risk_off_signal_generator
        ),
        patch(
            "portfolio_backtester.strategies.builtins.portfolio.dual_momentum_lagged_portfolio_strategy.get_top_weight_sp500_components",
            return_value=["AAA", "BBB"],
        ) as mock_top_components,
    ):
        strategy.generate_signals(
            all_historical_data=all_historical_data,
            benchmark_historical_data=benchmark_historical_data,
            current_date=dates[-1],
        )

    assert mock_top_components.call_args.kwargs["n"] == 20
    assert mock_top_components.call_args.kwargs["exact"] is True


def test_dual_momentum_canonical_config_uses_universe_definition_for_top_n() -> None:
    dates = pd.bdate_range("2023-01-02", periods=5)
    aaa = pd.Series(np.linspace(100.0, 104.0, len(dates)), index=dates)
    bbb = pd.Series(np.linspace(90.0, 94.0, len(dates)), index=dates)
    spx = pd.Series(np.linspace(4000.0, 4010.0, len(dates)), index=dates)

    all_historical_data = pd.concat(
        [_make_ohlc_multiindex(aaa, "AAA"), _make_ohlc_multiindex(bbb, "BBB")],
        axis=1,
    )
    benchmark_historical_data = _make_ohlc_multiindex(spx, "SPX")

    canonical = CanonicalScenarioConfig(
        name="test_canonical_topn",
        strategy="DualMomentumLaggedPortfolioStrategy",
        universe_definition=frozendict(
            {
                "type": "method",
                "method_name": "get_top_weight_sp500_components",
                "n_holdings": 50,
                "exact": True,
            }
        ),
        strategy_params=frozendict(
            {
                "lookback_months": 6,
                "lag_months": 0,
                "max_holdings": 10,
                "use_200sma_filter": False,
                "min_absolute_momentum": -1.0,
            }
        ),
    )

    strategy = DualMomentumLaggedPortfolioStrategy(canonical)

    risk_off_signal_generator = MagicMock()
    risk_off_signal_generator.generate_risk_off_signal.return_value = False

    with (
        patch.object(strategy, "validate_data_sufficiency", return_value=(True, "")),
        patch.object(strategy, "filter_universe_by_data_availability", return_value=["AAA", "BBB"]),
        patch.object(strategy, "_get_dual_momentum_candidates", return_value=[]),
        patch.object(
            strategy, "get_risk_off_signal_generator", return_value=risk_off_signal_generator
        ),
        patch(
            "portfolio_backtester.strategies.builtins.portfolio.dual_momentum_lagged_portfolio_strategy.get_top_weight_sp500_components",
            return_value=["AAA", "BBB"],
        ) as mock_top_components,
    ):
        strategy.generate_signals(
            all_historical_data=all_historical_data,
            benchmark_historical_data=benchmark_historical_data,
            current_date=dates[-1],
        )

    assert mock_top_components.call_args.kwargs["n"] == 50
    assert mock_top_components.call_args.kwargs["exact"] is True


def test_data_sufficiency_treats_sma_filter_window_as_days_not_months() -> None:
    dates = pd.bdate_range("2023-01-02", periods=130)
    asset_prices = pd.DataFrame(
        {"A": np.linspace(100.0, 130.0, len(dates))},
        index=dates,
    )
    benchmark_prices = pd.DataFrame(
        {"Close": np.linspace(400.0, 430.0, len(dates))},
        index=dates,
    )

    strategy = SharpeMomentumPortfolioStrategy(
        {"strategy_params": {"rolling_window": 3, "sma_filter_window": 20}}
    )

    is_sufficient, reason = strategy.validate_data_sufficiency(
        asset_prices,
        benchmark_prices,
        dates[-1],
    )

    assert is_sufficient, reason
