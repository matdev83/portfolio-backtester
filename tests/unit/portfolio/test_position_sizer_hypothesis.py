import pandas as pd
import numpy as np

from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from portfolio_backtester.portfolio.position_sizer import _normalize_weights


@st.composite
def matrices_and_leverage(draw):
    rows = draw(st.integers(min_value=1, max_value=8))
    cols = draw(st.integers(min_value=1, max_value=8))
    elements = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    matrix = draw(hnp.arrays(dtype=float, shape=(rows, cols), elements=elements))
    leverage = draw(st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False))
    df = pd.DataFrame(np.abs(matrix), columns=[f"C{i}" for i in range(cols)])
    return df, leverage


@given(matrices_and_leverage())
@settings(deadline=None)
def test_normalize_weights_row_sums_are_leverage_or_zero(args):
    weights, leverage = args
    normalized = _normalize_weights(weights, leverage=leverage)

    assert isinstance(normalized, pd.DataFrame)
    assert not normalized.isna().any().any()

    row_sums = normalized.abs().sum(axis=1)
    original_row_sums = weights.abs().sum(axis=1)

    # For non-zero rows: sums should be close to leverage
    non_zero_mask = original_row_sums > 0
    if non_zero_mask.any():
        assert np.allclose(row_sums[non_zero_mask].values, leverage, rtol=1e-6, atol=1e-9)

    # For zero rows: sums should be exactly zero
    zero_mask = ~non_zero_mask
    if zero_mask.any():
        assert np.all(row_sums[zero_mask].values == 0.0)


