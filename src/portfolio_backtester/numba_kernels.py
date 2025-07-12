"""Numba-accelerated kernels for high-frequency walk-forward evaluation.

These low-level functions are intentionally isolated from the rest of the
codebase so they can be JIT-compiled once and reused across optimisation
trials without pulling in large Python objects or pandas.
"""

from __future__ import annotations

import numba as _nb
import numpy as _np


@_nb.njit(cache=True, fastmath=True, parallel=True)
def window_mean_std(daily_rets: _np.ndarray, test_starts: _np.ndarray, test_ends: _np.ndarray) -> _np.ndarray:
    """Compute mean and standard deviation of daily portfolio returns for each
    walk-forward *test* segment.

    Parameters
    ----------
    daily_rets : (n_days,) float32
        Daily portfolio return series.
    test_starts, test_ends : (n_windows,) int64
        Inclusive start / end indices (row offsets) for each test window in
        *daily_rets*.

    Returns
    -------
    metrics : (n_windows, 2) float32
        ``metrics[i, 0]`` – mean return for window *i*.
        ``metrics[i, 1]`` – standard deviation for window *i*.
    """
    n_windows = test_starts.shape[0]
    out = _np.empty((n_windows, 2), dtype=_np.float32)

    for i in _nb.prange(n_windows):
        s_idx = test_starts[i]
        e_idx = test_ends[i] + 1  # inclusive → slice end
        seg = daily_rets[s_idx:e_idx]
        n = seg.size
        if n == 0:
            out[i, 0] = _np.nan
            out[i, 1] = _np.nan
            continue
        # Mean
        mu = _np.float32(0.0)
        for j in range(n):
            mu += seg[j]
        mu /= n
        # Std (two-pass; n small so fine)
        var = _np.float32(0.0)
        for j in range(n):
            diff = seg[j] - mu
            var += diff * diff
        var /= n
        out[i, 0] = mu
        out[i, 1] = _np.sqrt(var)
    return out 