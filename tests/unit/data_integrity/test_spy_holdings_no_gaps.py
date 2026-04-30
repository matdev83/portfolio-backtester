import pandas as pd
import pytest

import portfolio_backtester.universe_data.spy_holdings as spy_holdings


def test_spy_holdings_dense_business_days(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("market_data_multi_provider")
    from market_data_multi_provider.sp500 import builder as sp500_builder

    start = pd.Timestamp("2015-03-16")
    end = pd.Timestamp("2015-03-31")
    dates_b = pd.date_range(start, end, freq="B")
    frames = [
        pd.DataFrame({"date": [d], "ticker": ["AAPL"], "weight_pct": [100.0]}) for d in dates_b
    ]
    hist = pd.concat(frames, ignore_index=True)

    spy_holdings.reset_history_cache()

    monkeypatch.setattr(sp500_builder, "_HISTORY_DF", hist, raising=False)
    monkeypatch.setattr(sp500_builder, "_ensure_history_loaded", lambda: None)

    available = set(pd.to_datetime(hist["date"]).dt.normalize().unique())
    expected = set(pd.date_range(start, end, freq="B"))
    assert available == expected

    sample = dates_b[len(dates_b) // 2]
    res = spy_holdings.get_spy_holdings(sample, exact=True)
    assert not res.empty
