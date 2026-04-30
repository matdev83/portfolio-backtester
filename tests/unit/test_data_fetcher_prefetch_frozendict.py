"""Regression: canonical (frozendict) universes must trigger SP500 prefetch."""

from unittest.mock import MagicMock

from frozendict import frozendict

from portfolio_backtester.backtester_logic.data_fetcher import DataFetcher
from portfolio_backtester.canonical_config import CanonicalScenarioConfig


def test_collect_required_tickers_prefetches_when_universe_is_frozendict(monkeypatch) -> None:
    captured: dict = {}

    def fake_collect(*, start_date, end_date, n_holdings, exact):  # noqa: ARG001
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        captured["n_holdings"] = n_holdings
        captured["exact"] = exact
        return ["ZZZ_UNIQUE_PREFETCH"]

    monkeypatch.setattr(
        "portfolio_backtester.backtester_logic.data_fetcher._collect_sp500_top_components_over_range",
        fake_collect,
    )

    global_config = {
        "benchmark": "SPY",
        "start_date": "2019-01-01",
        "end_date": "2019-12-31",
    }
    scenario = CanonicalScenarioConfig(
        name="prefetch_test",
        strategy="DualMomentumLaggedPortfolioStrategy",
        start_date="2020-01-01",
        end_date="2020-12-31",
        benchmark_ticker="SPX",
        universe_definition=frozendict(
            {
                "type": "method",
                "method_name": "get_top_weight_sp500_components",
                "n_holdings": 50,
                "exact": False,
            }
        ),
    )

    fetcher = DataFetcher(global_config, MagicMock())

    def strategy_getter(_name, _cfg):
        strat = MagicMock()
        strat.get_non_universe_data_requirements.return_value = []
        return strat

    tickers, has_univ = fetcher.collect_required_tickers([scenario], strategy_getter)

    assert has_univ is True
    assert "ZZZ_UNIQUE_PREFETCH" in tickers
    assert captured["start_date"] == "2020-01-01"
    assert captured["end_date"] == "2020-12-31"
    assert captured["n_holdings"] == 50
    assert captured["exact"] is False
