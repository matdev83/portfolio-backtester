import pandas as pd
import datetime as dt
from pathlib import Path

import pytest

import portfolio_backtester.spy_holdings as spy_holdings

pytest.skip("SPY holdings tests disabled", allow_module_level=True)


FILLED_PARQUET = Path("data/data/spy_holdings_full_filled.parquet")

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _load_filled_history() -> pd.DataFrame:
    if not FILLED_PARQUET.exists():
        pytest.skip(f"Filled parquet missing: {FILLED_PARQUET}")
    return pd.read_parquet(FILLED_PARQUET)


def _business_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="B")  # NYSE approximation


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_spy_holdings_coverage_no_gaps():
    """Ensure every NYSE business day since 2004-01-01 has holdings data."""
    df = _load_filled_history()

    # Patch the in-memory cache so get_spy_holdings uses the filled dataset
    spy_holdings.reset_history_cache()
    spy_holdings._HISTORY_DF = df.copy()  # type: ignore[attr-defined]  # noqa: SLF001

    start = pd.Timestamp("2004-01-02")  # first business day in 2004
    end = df["date"].max()

    # Fast vectorised check: compare unique dates
    available = set(df["date"].dt.normalize().unique())
    expected = set(_business_days(start, end))

    missing = sorted(expected - available)
    assert not missing, f"Missing {len(missing)} business days – e.g. {missing[:5]}"

    # Spot-check the get_spy_holdings() helper on a few random dates to ensure function API still works
    sample_dates = [start, start + pd.Timedelta(days=123), end - pd.Timedelta(days=17)]
    for date in sample_dates:
        res = spy_holdings.get_spy_holdings(date, exact=True)
        assert not res.empty, f"No holdings returned for {date}" 