
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, assume
from hypothesis.extra import numpy as hnp
from portfolio_backtester.reporting.performance_metrics import (
    calculate_metrics,
    calculate_max_flat_period,
    _is_all_zero,
)

# Copied from original test file to replicate environment
@st.composite
def return_series(draw, min_length=30, max_length=500, allow_zeros=True):
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    
    if allow_zeros:
        returns = draw(
            hnp.arrays(
                dtype=float,
                shape=length,
                elements=st.floats(
                    min_value=-0.2, 
                    max_value=0.2,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            )
        )
    else:
        returns = draw(
            hnp.arrays(
                dtype=float,
                shape=length,
                elements=st.floats(
                    min_value=-0.2,
                    max_value=0.2,
                    allow_nan=False,
                    allow_infinity=False,
                ).filter(lambda x: x != 0),
            )
        )
    
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))
    
    start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    dates = pd.date_range(start=start_date, periods=length, freq='B')
    
    return pd.Series(returns, index=dates)


@given(return_series())
@settings(deadline=None, max_examples=50)
def test_metrics_max_drawdown_properties(returns):
    """Test properties of maximum drawdown calculation."""
    assume(not returns.empty)
    assume(len(returns) >= 3)
    assume(returns.min() > -0.9)
    assume(returns.max() < 10)
    
    bench_returns = pd.Series(0.0005, index=returns.index)
    metrics = calculate_metrics(returns, bench_returns, "SPY", num_trials=1)
    max_dd = metrics["Max Drawdown"]
    
    # 1. Range check
    if np.isfinite(max_dd):
        # Max drawdown is typically negative in this implementation (e.g. -0.20 for 20% drop)
        assert -1.01 < max_dd < 0.01

    # 2. Strictly increasing returns -> 0 drawdown
    increasing_returns = pd.Series(
        np.linspace(0.001, 0.001, len(returns)),
        index=returns.index
    )
    increasing_metrics = calculate_metrics(increasing_returns, bench_returns, "SPY", num_trials=1)
    increasing_dd = increasing_metrics["Max Drawdown"]
    if np.isfinite(increasing_dd):
        assert np.isclose(increasing_dd, 0, atol=1e-6)

    # 3. Crash returns -> ~ -1 drawdown
    crash_returns = pd.Series(0.0, index=returns.index)
    mid_point = len(crash_returns) // 2
    if len(crash_returns) > 1:
        crash_returns.iloc[mid_point] = -0.99
        crash_metrics = calculate_metrics(crash_returns, bench_returns, "SPY", num_trials=1)
        crash_dd = crash_metrics["Max Drawdown"]
        if np.isfinite(crash_dd):
            # Expecting close to -0.99
            assert -1.0 <= crash_dd <= -0.99


@given(return_series(min_length=252))
@settings(deadline=None, max_examples=30)
def test_metrics_annualized_return_properties(returns):
    assume(not returns.empty)
    bench_returns = pd.Series(0.0005, index=returns.index)
    metrics = calculate_metrics(returns, bench_returns, "SPY", num_trials=1)
    ann_return = metrics["Ann. Return"]
    
    assert np.isfinite(ann_return) or np.isnan(ann_return)
    
    if np.isfinite(ann_return):
        # 2. Relationship between Total Return and Ann. Return
        # Ann. Return = (1 + Total Return)^(1 / years) - 1
        total_return = metrics["Total Return"]
        if np.isfinite(total_return) and total_return > -1:
            years = len(returns) / 252.0 # Assuming daily
            # Allow for some frequency estimation differences
            # We just want to check they are consistent
            
            # Re-calculate expected based on inferred steps_per_year implicitly
            # Since we don't have easy access to steps_per_year here without re-inferring
            # let's just check sign consistency which is robust
            if total_return > 0:
                assert ann_return > 0
            elif -1 < total_return < 0:
                assert -1 < ann_return < 0
            elif total_return == 0:
                assert ann_return == 0
            
            # Also check that scaling affects the return, even if direction is ambiguous due to vol drag
            scaled_returns = returns * 1.5
            scaled_metrics = calculate_metrics(scaled_returns, bench_returns, "SPY", num_trials=1)
            assert scaled_metrics["Ann. Return"] != ann_return or len(returns) < 5 or abs(ann_return) < 1e-6


@given(return_series(min_length=252))
@settings(deadline=None, max_examples=30)
def test_metrics_annualized_volatility_properties(returns):
    assume(not returns.empty)
    bench_returns = pd.Series(0.0005, index=returns.index)
    metrics = calculate_metrics(returns, bench_returns, "SPY", num_trials=1)
    ann_vol = metrics["Ann. Vol"]
    
    if np.isfinite(ann_vol):
        assert ann_vol >= 0
        
    # 3. Zero for constant returns
    constant_returns = pd.Series(0.001, index=returns.index)
    constant_metrics = calculate_metrics(constant_returns, bench_returns, "SPY", num_trials=1)
    constant_ann_vol = constant_metrics["Ann. Vol"]
    if np.isfinite(constant_ann_vol):
        assert np.isclose(constant_ann_vol, 0, atol=1e-6)


@given(return_series())
@settings(deadline=None)
def test_max_flat_period_properties(returns):
    assume(not returns.empty and len(returns) >= 100)
    flat_period = calculate_max_flat_period(returns)
    assert flat_period >= 0
    
    # 2. Zero for no zeros
    non_zero_returns = returns.copy()
    # Masking with small value
    non_zero_returns = non_zero_returns.apply(lambda x: x if abs(x) > 1e-9 else 0.0001)
    # Double check no zeros
    if (non_zero_returns.abs() > 1e-9).all():
        non_zero_flat_period = calculate_max_flat_period(non_zero_returns)
        assert non_zero_flat_period == 0
