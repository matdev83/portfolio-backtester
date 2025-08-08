import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.trading.unified_commission_calculator import get_unified_commission_calculator
from portfolio_backtester.numba_kernels import (
    detailed_commission_slippage_kernel,
)


def _build_prices(dates, tickers, seed=0):
    rng = np.random.default_rng(seed)
    # Positive prices in a reasonable range
    arr = rng.uniform(10.0, 200.0, size=(len(dates), len(tickers)))
    return pd.DataFrame(arr, index=dates, columns=tickers)


def _build_weights(dates, tickers, seed=1):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.2, size=(len(dates), len(tickers)))
    # Soft normalize rows to sum to ~1 by clipping and re-normalization of positives
    w = np.clip(w, -1.0, 1.0)
    # Make them long-only for simpler comparison with calculator (not required but common)
    w = np.abs(w)
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    w = w / row_sums
    return pd.DataFrame(w, index=dates, columns=tickers)


@pytest.mark.parametrize("use_numba", [True, False])
def test_detailed_commissions_matches_calculator_series(use_numba):
    # Synthetic small case
    dates = pd.date_range("2021-01-01", periods=15, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD"]

    prices = _build_prices(dates, tickers)
    weights = _build_weights(dates, tickers)

    # Global config parameters matching UnifiedCommissionCalculator defaults
    global_config = {
        "commission_per_share": 0.005,
        "commission_min_per_order": 1.0,
        "commission_max_percent_of_trade": 0.005,
        "slippage_bps": 2.5,
        "portfolio_value": 100000.0,
    }

    # Calculate reference commissions with UnifiedCommissionCalculator
    calc = get_unified_commission_calculator(global_config)

    # Turnover per ticker per day is abs diff in weights; calculator expects daily series fraction or per-asset series
    # We pass weights_daily and prices daily, calculator derives quantities from weights diff and prices
    turnover_series = pd.Series(1.0, index=dates, dtype=float)  # not used in detailed path except as fallback
    ref_total_costs, breakdown, _ = calc.calculate_portfolio_commissions(
        turnover=turnover_series,
        weights_daily=weights,
        price_data=prices,
        portfolio_value=float(global_config["portfolio_value"]),
        transaction_costs_bps=None,  # force detailed path
    )
    ref_total_costs = ref_total_costs.astype(float).reindex(dates).fillna(0.0)

    # Prepare ndarray inputs for fast path: current weights (not shifted), prices, masks
    weights_arr = weights.to_numpy(copy=True)
    prices_arr = prices.to_numpy(copy=True)
    mask_arr = np.isfinite(prices_arr) & (prices_arr > 0.0)

    # Use single optimized implementation (no fallback needed)
    tc_frac = detailed_commission_slippage_kernel(
        weights_current=weights_arr,
        close_prices=prices_arr,
        portfolio_value=float(global_config["portfolio_value"]),
        commission_per_share=float(global_config["commission_per_share"]),
        commission_min_per_order=float(global_config["commission_min_per_order"]),
        commission_max_percent=float(global_config["commission_max_percent_of_trade"]),
        slippage_bps=float(global_config["slippage_bps"]),
        price_mask=mask_arr,
    )

    fast_series = pd.Series(tc_frac, index=dates)

    # Compare with a reasonable tolerance – the calculator iterates per-asset similarly; the math should match
    # Tolerance accounts for minor float ops differences
    atol = 1e-10
    rtol = 1e-10
    pd.testing.assert_index_equal(ref_total_costs.index, fast_series.index)
    ref_vals = np.asarray(ref_total_costs.to_numpy(dtype=float))
    fast_vals = np.asarray(fast_series.to_numpy(dtype=float))
    assert np.allclose(ref_vals, fast_vals, atol=atol, rtol=rtol)


def test_zero_weights_zero_costs():
    dates = pd.date_range("2021-01-01", periods=5, freq="B")
    tickers = ["A", "B"]
    prices = pd.DataFrame([[100.0, 50.0]] * len(dates), index=dates, columns=tickers)
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)

    global_config = {
        "commission_per_share": 0.005,
        "commission_min_per_order": 1.0,
        "commission_max_percent_of_trade": 0.005,
        "slippage_bps": 2.5,
        "portfolio_value": 100000.0,
    }

    calc = get_unified_commission_calculator(global_config)
    turnover_series = pd.Series(1.0, index=dates, dtype=float)

    ref_total_costs, breakdown, _ = calc.calculate_portfolio_commissions(
        turnover=turnover_series,
        weights_daily=weights,
        price_data=prices,
        portfolio_value=float(global_config["portfolio_value"]),
        transaction_costs_bps=None,
    )
    ref_total_costs = ref_total_costs.astype(float).reindex(dates).fillna(0.0)

    weights_arr = weights.to_numpy(copy=True)
    prices_arr = prices.to_numpy(copy=True)
    mask_arr = np.isfinite(prices_arr) & (prices_arr > 0.0)

    tc_frac = detailed_commission_slippage_kernel(
        weights_current=weights_arr,
        close_prices=prices_arr,
        portfolio_value=float(global_config["portfolio_value"]),
        commission_per_share=float(global_config["commission_per_share"]),
        commission_min_per_order=float(global_config["commission_min_per_order"]),
        commission_max_percent=float(global_config["commission_max_percent_of_trade"]),
        slippage_bps=float(global_config["slippage_bps"]),
        price_mask=mask_arr,
    )
    fast_series = pd.Series(tc_frac, index=dates)

    # All zeros
    assert np.allclose(np.asarray(ref_total_costs.to_numpy(dtype=float)), 0.0)
    assert np.allclose(np.asarray(fast_series.to_numpy(dtype=float)), 0.0)
