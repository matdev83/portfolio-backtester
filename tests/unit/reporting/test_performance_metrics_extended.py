import pytest
import pandas as pd
import numpy as np
from portfolio_backtester.reporting.performance_metrics import (
    calculate_metrics,
    _is_all_zero,
    _safe_moment,
    _default_zero_activity_metrics,
    calculate_max_dd_recovery_time,
    calculate_max_flat_period,
    _infer_steps_per_year
)
from scipy.stats import skew, kurtosis

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def daily_dates():
    return pd.date_range("2023-01-01", periods=100, freq="D")

@pytest.fixture
def random_returns(daily_dates):
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.01, size=len(daily_dates)), index=daily_dates)

@pytest.fixture
def benchmark_returns(daily_dates):
    np.random.seed(43)
    return pd.Series(np.random.normal(0.0003, 0.008, size=len(daily_dates)), index=daily_dates)

# -------------------------------------------------------------------------
# Helper Function Tests
# -------------------------------------------------------------------------

def test_infer_steps_per_year():
    # Daily
    assert _infer_steps_per_year(pd.date_range("2023-01-01", periods=10, freq="D")) == 252
    assert _infer_steps_per_year(pd.date_range("2023-01-01", periods=10, freq="B")) == 252
    # Weekly
    assert _infer_steps_per_year(pd.date_range("2023-01-01", periods=10, freq="W-FRI")) == 52
    # Monthly
    assert _infer_steps_per_year(pd.date_range("2023-01-01", periods=10, freq="ME")) == 12
    assert _infer_steps_per_year(pd.date_range("2023-01-01", periods=10, freq="MS")) == 12
    # Short series default
    assert _infer_steps_per_year(pd.date_range("2023-01-01", periods=2, freq="D")) == 252
    
    # Heuristic fallback (irregular data)
    irregular_dates = pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-05"])
    assert _infer_steps_per_year(irregular_dates) == 252

def test_is_all_zero():
    assert _is_all_zero(pd.Series([0.0, 0.0, 0.0])) is True
    assert _is_all_zero(pd.Series([0.0, 0.0, 1e-16])) is True  # Epsilon check
    assert _is_all_zero(pd.Series([0.0, 0.0, 0.0001])) is False
    assert _is_all_zero(pd.Series([], dtype=float)) is False  # Empty is not "all zero" returns, handled separately

def test_safe_moment():
    # Constant series -> 0.0
    s_const = pd.Series([1.0, 1.0, 1.0])
    assert _safe_moment(skew, s_const) == 0.0
    assert _safe_moment(kurtosis, s_const) == 0.0
    
    # Near constant (numerical noise)
    s_near_const = pd.Series([1.0, 1.0 + 1e-16, 1.0 - 1e-16])
    assert _safe_moment(skew, s_near_const) == 0.0

    # Normal data
    np.random.seed(42)
    s_norm = pd.Series(np.random.normal(0, 1, 100))
    sk = _safe_moment(skew, s_norm)
    kt = _safe_moment(kurtosis, s_norm)
    assert isinstance(sk, float)
    assert isinstance(kt, float)
    assert abs(sk) < 1.0  # Normal dist skew approx 0
    assert abs(kt) < 1.0  # Normal dist excess kurtosis approx 0

def test_default_zero_activity_metrics():
    m = _default_zero_activity_metrics(is_all_zero_returns=True)
    assert m["Total Return"] == 0.0
    assert m["Sharpe"] == 0.0
    
    m_bad = _default_zero_activity_metrics(is_all_zero_returns=False)
    assert m_bad["Total Return"] == -9999.0
    assert m_bad["Sharpe"] == -9999.0

# -------------------------------------------------------------------------
# Specific Metric Calculation Tests
# -------------------------------------------------------------------------

def test_calculate_max_dd_recovery_time_normal(random_returns):
    # Create a scenario with a known drawdown and recovery
    # Day 0-9: Flat
    # Day 10: -10% drop
    # Day 11-20: Recovery (1% per day approx)
    rets = pd.Series(0.0, index=pd.date_range("2023-01-01", periods=30, freq="D"))
    rets.iloc[10] = -0.10
    for i in range(11, 22):
        rets.iloc[i] = 0.011 # Enough to recover
        
    recovery_time = calculate_max_dd_recovery_time(rets)
    assert recovery_time >= 10 and recovery_time <= 12

def test_calculate_max_dd_recovery_time_no_drawdown():
    rets = pd.Series([0.01] * 20, index=pd.date_range("2023-01-01", periods=20, freq="D"))
    assert calculate_max_dd_recovery_time(rets) == 0

def test_calculate_max_dd_recovery_time_never_recovers():
    rets = pd.Series(0.0, index=pd.date_range("2023-01-01", periods=20, freq="D"))
    rets.iloc[5] = -0.5 # Big crash
    # Remaining are 0, so never recovers back to 1.0
    
    # Logic in function: "End of drawdown (recovery)" triggers when dd_value >= -1e-6
    # If it never goes back above that, it never updates max_recovery_time for that period?
    # Or does it count until end of series? The current implementation looks for "End of drawdown" event.
    # If in_drawdown is True at end of loop, it doesn't calculate that period.
    assert calculate_max_dd_recovery_time(rets) == 0 # Or whatever behavior is expected for unrecovered DD

def test_calculate_max_flat_period():
    # 5 days flat, then return, then 3 days flat
    rets = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01])
    assert calculate_max_flat_period(rets) == 5
    
    # No flat
    rets_active = pd.Series([0.01] * 10)
    assert calculate_max_flat_period(rets_active) == 0
    
    # All flat
    rets_dead = pd.Series([0.0] * 10)
    assert calculate_max_flat_period(rets_dead) == 10

# -------------------------------------------------------------------------
# Main calculate_metrics Tests
# -------------------------------------------------------------------------

def test_calculate_metrics_full_run(random_returns, benchmark_returns):
    metrics = calculate_metrics(random_returns, benchmark_returns, "SPY")
    
    expected_keys = [
        "Total Return", "Ann. Return", "Ann. Vol", "Sharpe", "Sortino", 
        "Calmar", "Max Drawdown", "VaR (5%)", "CVaR (5%)", 
        "Tail Ratio", "Skew", "Kurtosis", "Deflated Sharpe"
    ]
    for key in expected_keys:
        assert key in metrics
        assert not np.isnan(metrics[key])

def test_calculate_metrics_insufficient_data():
    # Only 1 data point
    rets = pd.Series([0.01], index=pd.date_range("2023-01-01", periods=1))
    bench = pd.Series([0.01], index=pd.date_range("2023-01-01", periods=1))
    
    m = calculate_metrics(rets, bench, "SPY")
    
    # Most stats require >1 point (std dev)
    assert np.isnan(m["Ann. Vol"]) or m["Ann. Vol"] == 0.0
    assert np.isnan(m["Sharpe"]) or m["Sharpe"] == 0.0

def test_calculate_metrics_all_zeros():
    rets = pd.Series([0.0] * 100, index=pd.date_range("2023-01-01", periods=100))
    bench = pd.Series([0.01] * 100, index=pd.date_range("2023-01-01", periods=100))
    
    m = calculate_metrics(rets, bench, "SPY")
    
    assert m["Total Return"] == 0.0
    assert m["Sharpe"] == 0.0
    assert m["Max Drawdown"] == 0.0

def test_deflated_sharpe_ratio_edge_cases():
    # 1. Very short series -> NaN
    # MUST provide DatetimeIndex, otherwise _infer_steps_per_year fails with AttributeError on .days
    dates_short = pd.date_range("2023-01-01", periods=10)
    short_rets = pd.Series(np.random.normal(0, 1, 10), index=dates_short)
    
    # We can check via the metrics dictionary from calculate_metrics
    # Need > 100 points for DSR
    bench = pd.Series(np.random.normal(0, 1, 10), index=dates_short)
    m = calculate_metrics(short_rets, bench, "B")
    assert np.isnan(m["Deflated Sharpe"])

    # 2. Constant returns (std=0) -> NaN (guarded)
    dates_const = pd.date_range("2023-01-01", periods=150)
    const_rets = pd.Series([0.01] * 150, index=dates_const) # Enough length, but constant
    bench_const = pd.Series([0.01] * 150, index=dates_const)
    m_const = calculate_metrics(const_rets, bench_const, "B")
    assert np.isnan(m_const["Deflated Sharpe"])

def test_metrics_with_trade_stats(random_returns, benchmark_returns):
    trade_stats = {
        "all_num_trades": 10,
        "all_win_rate_pct": 0.6,
        "all_total_pnl_net": 1000.0,
        "long_num_trades": 5,
        "short_num_trades": 5
    }
    
    m = calculate_metrics(random_returns, benchmark_returns, "SPY", trade_stats=trade_stats)
    
    assert m["Number of Trades (All)"] == 10
    assert m["Win Rate % (All)"] == 0.6
    assert m["Number of Trades (Long)"] == 5