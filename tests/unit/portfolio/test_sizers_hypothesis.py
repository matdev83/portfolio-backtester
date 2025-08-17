import numpy as np
import pandas as pd

from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from portfolio_backtester.portfolio.position_sizer import (
    RollingSharpeSizer,
    RollingBetaSizer,
)


@st.composite
def price_signal_benchmark(draw):
    rows = draw(st.integers(min_value=40, max_value=120))
    cols = draw(st.integers(min_value=2, max_value=6))
    ret_elements = st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False)
    rets = draw(hnp.arrays(dtype=float, shape=(rows, cols), elements=ret_elements))
    # Build prices from returns to avoid constant series
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    prices_df = pd.DataFrame(prices, columns=[f"A{i}" for i in range(cols)])

    # Non-negative signals to simplify invariants
    signal_elements = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    signals = draw(hnp.arrays(dtype=float, shape=(rows, cols), elements=signal_elements))
    signals_df = pd.DataFrame(signals, columns=prices_df.columns)

    # Benchmark derived from first asset to ensure relationship
    bench_rets = rets[:, 0]
    bench_prices = 100.0 * np.cumprod(1.0 + bench_rets)
    benchmark = pd.Series(bench_prices, index=prices_df.index, name="BENCH")
    return prices_df, signals_df, benchmark


@given(price_signal_benchmark())
@settings(deadline=None)
def test_rolling_sharpe_sizer_outputs_normalized_nonnegative(data):
    prices, signals, _ = data
    sizer = RollingSharpeSizer()
    weights = sizer.calculate_weights(signals=signals, prices=prices, window=10)

    assert isinstance(weights, pd.DataFrame)
    assert not weights.isna().any().any()
    assert (weights.values >= -1e-12).all()  # numerical noise tolerance
    row_sums = weights.abs().sum(axis=1)
    pre_norm_sum = (signals.abs() * 1.0).sum(axis=1)  # placeholder to align masks
    # If sizer produced any positive pre-normalization mass, normalized row sum should be ~1, else 0
    # We approximate by checking if resulting weights have any positive entries
    has_mass = (weights.abs().sum(axis=1) > 0)
    if has_mass.any():
        assert np.allclose(row_sums[has_mass].values, 1.0, rtol=1e-6, atol=1e-9)
    zero_mass = ~has_mass
    if zero_mass.any():
        assert np.all(row_sums[zero_mass].values == 0.0)


@given(price_signal_benchmark())
@settings(deadline=None)
def test_rolling_beta_sizer_outputs_normalized_when_benchmark_present(data):
    prices, signals, benchmark = data
    sizer = RollingBetaSizer()
    weights = sizer.calculate_weights(signals=signals, prices=prices, window=5, benchmark=benchmark)

    assert isinstance(weights, pd.DataFrame)
    assert not weights.isna().any().any()
    row_sums = weights.abs().sum(axis=1)
    has_mass = (weights.abs().sum(axis=1) > 0)
    if has_mass.any():
        assert np.allclose(row_sums[has_mass].values, 1.0, rtol=1e-6, atol=1e-9)
    zero_mass = ~has_mass
    if zero_mass.any():
        assert np.all(row_sums[zero_mass].values == 0.0)


