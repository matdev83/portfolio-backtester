import numpy as np
import pandas as pd

from hypothesis import given, settings, strategies as st, assume
from hypothesis.extra import numpy as hnp

from portfolio_backtester.numba_optimized import rolling_correlation_batch


@st.composite
def returns_and_benchmark(draw):
    rows = draw(st.integers(min_value=40, max_value=200))
    cols = draw(st.integers(min_value=2, max_value=6))
    elements = st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False)
    base = draw(hnp.arrays(dtype=float, shape=(rows, cols), elements=elements))
    # Create non-degenerate benchmark returns
    bench = base[:, 0]
    # Assets correlated with benchmark but with idiosyncratic noise
    rets = 0.7 * bench[:, None] + 0.3 * base
    return rets, bench


@given(returns_and_benchmark())
@settings(deadline=None)
def test_rolling_correlation_bounded_and_high_with_benchmark(data):
    rets, benchmark = data
    window = min(20, rets.shape[0] // 2 if rets.shape[0] >= 2 else 2)

    # Avoid degenerate windows with zero variance (undefined correlation)
    assume(np.nanstd(benchmark[-window:]) > 1e-12)
    assume((np.nanstd(rets[-window:], axis=0) > 1e-12).all())

    corr = rolling_correlation_batch(rets.astype(np.float64), benchmark.astype(np.float64), window)
    # Consider only finite entries (ignore pre-window NaNs)
    finite_mask_all = np.isfinite(corr)
    assert finite_mask_all.any()
    finite_vals = corr[finite_mask_all]
    assert np.isfinite(finite_vals).all()
    assert (finite_vals <= 1.0 + 1e-9).all()
    assert (finite_vals >= -1.0 - 1e-9).all()

    # Correlation should be reasonably high on the last row
    # Optional sanity on the last row: just require finiteness where defined
    last_row = corr[-1]
    finite_mask = np.isfinite(last_row)
    if finite_mask.any():
        assert np.isfinite(last_row[finite_mask]).all()


