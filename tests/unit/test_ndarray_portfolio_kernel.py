import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.numba_kernels import position_and_pnl_kernel
from portfolio_backtester.ndarray_adapter import to_ndarrays


def _pandas_path(weights_for_returns: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    common_cols = [c for c in weights_for_returns.columns if c in rets.columns]
    if not common_cols:
        return pd.Series(0.0, index=weights_for_returns.index)
    # Use the same pandas formulation as in portfolio_logic fallback
    prod_df = pd.DataFrame(weights_for_returns[common_cols]).mul(pd.DataFrame(rets[common_cols]), axis=0)
    return pd.Series(prod_df.sum(axis="columns"), index=weights_for_returns.index)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_position_and_pnl_kernel_matches_pandas(dtype):
    # Small synthetic dataset
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    tickers = ["AAA", "BBB", "CCC"]

    # Build weights_daily and returns
    rng = np.random.default_rng(0)
    weights_daily = pd.DataFrame(rng.random((len(dates), len(tickers))), index=dates, columns=tickers)
    # Normalize rows to sum to 1.0 to mimic portfolio weights
    weights_daily = weights_daily.div(weights_daily.sum(axis=1), axis=0)

    # Shift to get weights_for_returns (previous day weights, first day zeros)
    weights_for_returns = weights_daily.shift(1).fillna(0.0)

    # Returns
    rets = pd.DataFrame(rng.normal(0, 0.01, (len(dates), len(tickers))), index=dates, columns=tickers)

    # Adapter
    adapter = to_ndarrays(weights_for_returns, rets, tickers, dates, use_float32=(dtype == np.float32))
    w = adapter["weights"].astype(dtype, copy=False)
    r = adapter["rets"].astype(dtype, copy=False)
    m = adapter["mask"]

    # Kernels
    # Use single optimized implementation
    gross_arr, equity, turnover = position_and_pnl_kernel(w, r, m)

    gross_series_kernel = pd.Series(gross_arr, index=dates)

    # Pandas baseline
    gross_series_pd = _pandas_path(weights_for_returns, rets)

    # Compare
    pd.testing.assert_index_equal(gross_series_kernel.index, gross_series_pd.index)
    # Allow tiny tolerance for float32
    atol = 1e-6 if dtype is np.float64 else 1e-4
    assert np.allclose(gross_series_kernel.values, gross_series_pd.values, atol=atol, rtol=0)

    # Basic sanity checks on outputs
    assert len(equity) == len(dates)
    assert len(turnover) == len(dates)
    # Equity starts at 1 + first return
    assert np.isfinite(equity).all()
    assert np.isfinite(turnover).all()
