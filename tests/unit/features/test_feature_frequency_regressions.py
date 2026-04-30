import numpy as np
import pandas as pd
import pandas.testing as pdt

from portfolio_backtester.features.calmar_ratio import CalmarRatio
from portfolio_backtester.features.sortino_ratio import SortinoRatio
from portfolio_backtester.numba_optimized import calmar_batch_fixed, sortino_fast_fixed


def _make_price_data() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=40)
    returns = np.array(
        [
            0.0,
            0.01,
            -0.005,
            0.012,
            -0.007,
            0.006,
            0.004,
            -0.003,
            0.009,
            -0.002,
        ]
        * 4,
        dtype=float,
    )
    prices = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame({"A": prices}, index=dates)


def test_sortino_ratio_interprets_rolling_window_in_months_on_daily_data() -> None:
    price_data = _make_price_data()
    result = SortinoRatio(rolling_window=1, target_return=0.0).compute(price_data)

    rets = price_data.pct_change(fill_method=None).infer_objects().fillna(0.0)
    expected_np = sortino_fast_fixed(
        rets.to_numpy(dtype=np.float64),
        21,
        0.0,
        annualization_factor=252.0,
    )
    expected = pd.DataFrame(expected_np, index=rets.index, columns=rets.columns).clip(-10.0, 10.0)
    valid_rows = np.where(np.arange(len(expected)) >= 20)[0]
    expected.iloc[valid_rows] = expected.iloc[valid_rows].fillna(0.0)

    pdt.assert_frame_equal(result, expected)


def test_calmar_ratio_interprets_rolling_window_in_months_on_daily_data() -> None:
    price_data = _make_price_data()
    result = CalmarRatio(rolling_window=1).compute(price_data)

    rets = price_data.pct_change(fill_method=None).fillna(0.0)
    expected_np = calmar_batch_fixed(rets.to_numpy(dtype=np.float64), 21, cal_factor=252.0)
    expected = pd.DataFrame(expected_np, index=rets.index, columns=rets.columns).clip(-10.0, 10.0)

    pdt.assert_frame_equal(result, expected)
