import numpy as np
import pandas as pd
import warnings

from portfolio_backtester.reporting.metrics import (
    compensated_skew_fast,
    compensated_kurtosis_fast,
)


def test_numba_moments_on_random_data():
    rng = np.random.default_rng(123)
    data = rng.normal(0, 0.01, 500)
    sk = compensated_skew_fast(data)
    ku = compensated_kurtosis_fast(data)
    assert np.isfinite(sk)
    assert np.isfinite(ku)


def test_numba_moments_warn_on_degenerate():
    const = np.array([0.0] * 10, dtype=float)
    # The Python layer emits RuntimeWarning for degenerate inputs; Numba functions return 0.0
    sk = compensated_skew_fast(const)
    ku = compensated_kurtosis_fast(const)
    assert sk == 0.0
    assert ku == 0.0


