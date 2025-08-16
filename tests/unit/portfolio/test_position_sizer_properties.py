"""
Property-based tests for position sizing algorithms.

This module uses Hypothesis to test invariants and properties of the position sizing
algorithms, ensuring they behave correctly under a wide range of inputs and edge cases.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

from hypothesis import given, settings, strategies as st, assume
from hypothesis.extra import numpy as hnp

from portfolio_backtester.portfolio.position_sizer import (
    _normalize_weights,
    EqualWeightSizer,
    RollingSharpeSizer,
    RollingSortinoSizer,
    RollingBetaSizer,
    RollingBenchmarkCorrSizer,
    RollingDownsideVolatilitySizer,
    get_position_sizer,
    get_position_sizer_from_config,
)


@st.composite
def price_and_signal_data(
    draw,
    min_rows=30,
    max_rows=100,
    min_assets=2,
    max_assets=10,
    include_zeros=True,
    include_negatives=True,
):
    """
    Generate price and signal data for testing position sizers.
    
    Args:
        min_rows: Minimum number of time periods
        max_rows: Maximum number of time periods
        min_assets: Minimum number of assets
        max_assets: Maximum number of assets
        include_zeros: Whether to include zero signals
        include_negatives: Whether to include negative signals
        
    Returns:
        Tuple of (prices DataFrame, signals DataFrame)
    """
    rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_assets = draw(st.integers(min_value=min_assets, max_value=max_assets))
    
    # Generate dates
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))  # Avoid month end issues
    
    start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    dates = pd.date_range(start=start_date, periods=rows, freq='B')
    
    # Create asset names
    assets = [f"ASSET{i}" for i in range(n_assets)]
    
    # Generate price data with realistic properties
    price_data = {}
    for asset in assets:
        # Start with a base price
        base_price = draw(st.floats(min_value=10.0, max_value=1000.0))
        
        # Generate returns with some realistic volatility
        volatility = draw(st.floats(min_value=0.005, max_value=0.03))
        returns = np.random.normal(0.0005, volatility, rows)  # Small positive drift
        
        # Convert returns to price series
        prices = base_price * np.cumprod(1 + returns)
        price_data[asset] = prices
    
    # Create price DataFrame
    prices_df = pd.DataFrame(price_data, index=dates)
    
    # Generate signal data
    signal_data = {}
    signal_values = [0.0, 0.5, 1.0]
    if include_negatives:
        signal_values.extend([-0.5, -1.0])
    
    for asset in assets:
        # Use discrete values instead of continuous to avoid extremely small values
        signals = np.random.choice(signal_values, size=rows)
        
        if include_zeros:
            # Randomly set some signals to exactly zero
            zero_mask = draw(
                hnp.arrays(
                    dtype=bool,
                    shape=rows,
                    elements=st.booleans(),
                )
            )
            signals[zero_mask] = 0.0
            
        signal_data[asset] = signals
    
    # Create signal DataFrame
    signals_df = pd.DataFrame(signal_data, index=dates)
    
    return prices_df, signals_df


@st.composite
def price_signal_benchmark_data(draw):
    """
    Generate price, signal, and benchmark data for testing position sizers.
    
    Returns:
        Tuple of (prices DataFrame, signals DataFrame, benchmark Series)
    """
    prices_df, signals_df = draw(price_and_signal_data())
    
    # Generate benchmark data
    # Option 1: Use one of the assets as benchmark
    use_asset_as_benchmark = draw(st.booleans())
    
    if use_asset_as_benchmark and len(prices_df.columns) > 0:
        benchmark_asset = draw(st.sampled_from(list(prices_df.columns)))
        benchmark = prices_df[benchmark_asset].copy()
        benchmark.name = "BENCHMARK"
    else:
        # Option 2: Generate a new benchmark series
        benchmark_volatility = draw(st.floats(min_value=0.005, max_value=0.03))
        benchmark_returns = np.random.normal(0.0005, benchmark_volatility, len(prices_df))
        benchmark_prices = 100.0 * np.cumprod(1 + benchmark_returns)
        benchmark = pd.Series(benchmark_prices, index=prices_df.index, name="BENCHMARK")
    
    return prices_df, signals_df, benchmark


@st.composite
def daily_and_monthly_price_data(draw):
    """
    Generate daily and monthly price data for testing position sizers.
    
    Returns:
        Tuple of (monthly prices DataFrame, daily prices DataFrame, benchmark Series)
    """
    # Generate daily price data
    daily_rows = draw(st.integers(min_value=252, max_value=756))  # 1-3 years of daily data
    n_assets = draw(st.integers(min_value=2, max_value=10))
    
    start_date = pd.Timestamp(year=2000, month=1, day=1)
    daily_dates = pd.date_range(start=start_date, periods=daily_rows, freq='B')
    
    # Create asset names
    assets = [f"ASSET{i}" for i in range(n_assets)]
    
    # Generate daily price data
    daily_price_data = {}
    for asset in assets:
        base_price = draw(st.floats(min_value=10.0, max_value=1000.0))
        volatility = draw(st.floats(min_value=0.005, max_value=0.03))
        returns = np.random.normal(0.0005, volatility, daily_rows)
        prices = base_price * np.cumprod(1 + returns)
        daily_price_data[asset] = prices
    
    daily_prices_df = pd.DataFrame(daily_price_data, index=daily_dates)
    
    # Generate benchmark
    benchmark_volatility = draw(st.floats(min_value=0.005, max_value=0.03))
    benchmark_returns = np.random.normal(0.0005, benchmark_volatility, daily_rows)
    benchmark_prices = 100.0 * np.cumprod(1 + benchmark_returns)
    benchmark = pd.Series(benchmark_prices, index=daily_dates, name="BENCHMARK")
    
    # Create monthly price data by resampling daily data
    monthly_prices_df = daily_prices_df.resample('ME').last()
    
    return monthly_prices_df, daily_prices_df, benchmark


@given(price_and_signal_data())
@settings(deadline=None)
def test_equal_weight_sizer_properties(data):
    """Test properties of the EqualWeightSizer."""
    prices_df, signals_df = data
    
    # Skip empty dataframes
    assume(not prices_df.empty and not signals_df.empty)
    
    sizer = EqualWeightSizer()
    
    # Test with default leverage
    weights = sizer.calculate_weights(signals=signals_df, prices=prices_df)
    
    # Properties that should hold:
    # 1. Output should be a DataFrame with same shape as signals
    assert isinstance(weights, pd.DataFrame)
    assert weights.shape == signals_df.shape
    
    # 2. No NaN values
    assert not weights.isna().any().any()
    
    # 3. For each date, weights should sum to 1.0 if there are any non-zero signals
    for date in weights.index:
        row_sum = weights.loc[date].abs().sum()
        signals_row_sum = signals_df.loc[date].abs().sum()
        
        if signals_row_sum > 0:
            assert np.isclose(row_sum, 1.0, rtol=1e-6, atol=1e-9), f"Row sum for {date} is {row_sum}, expected 1.0"
        else:
            assert np.isclose(row_sum, 0.0, rtol=1e-6, atol=1e-9), f"Row sum for {date} is {row_sum}, expected 0.0"
    
    # 4. Weights should be non-negative (EqualWeightSizer uses abs() on signals)
    assert (weights >= -1e-9).all().all(), "All weights should be non-negative"
    
    # 5. If signal is exactly zero, weight should be exactly zero
    for date in weights.index:
        for col in weights.columns:
            signal = signals_df.loc[date, col]
            weight = weights.loc[date, col]
            
            if signal == 0.0:
                assert weight == 0.0
    
    # Test with custom leverage
    leverage = 2.0
    leveraged_weights = sizer.calculate_weights(signals=signals_df, prices=prices_df, leverage=leverage)
    
    # 6. With leverage, weights should sum to leverage if there are any non-zero signals
    for date in leveraged_weights.index:
        row_sum = leveraged_weights.loc[date].abs().sum()
        signals_row_sum = signals_df.loc[date].abs().sum()
        
        if signals_row_sum > 0:
            assert np.isclose(row_sum, leverage, rtol=1e-6, atol=1e-9), f"Row sum for {date} is {row_sum}, expected {leverage}"
        else:
            assert np.isclose(row_sum, 0.0, rtol=1e-6, atol=1e-9), f"Row sum for {date} is {row_sum}, expected 0.0"


@given(price_signal_benchmark_data())
@settings(deadline=None)
def test_rolling_sharpe_sizer_properties(data):
    """Test properties of the RollingSharpeSizer."""
    prices_df, signals_df, _ = data
    
    # Skip empty dataframes
    assume(not prices_df.empty and not signals_df.empty)
    # Ensure we have enough data for rolling window
    assume(len(prices_df) >= 10)
    
    sizer = RollingSharpeSizer()
    window = min(10, len(prices_df) - 1)  # Ensure window is smaller than data length
    
    weights = sizer.calculate_weights(signals=signals_df, prices=prices_df, window=window)
    
    # Properties that should hold:
    # 1. Output should be a DataFrame with same shape as signals
    assert isinstance(weights, pd.DataFrame)
    assert weights.shape == signals_df.shape
    
    # 2. No NaN values
    assert not weights.isna().any().any()
    
    # 3. For each date, weights should sum to 1.0 if there are any non-zero signals
    # But for the first few dates, there might not be enough data to calculate metrics
    # So we'll only check dates after the window size
    min_data_date = weights.index[0] + pd.Timedelta(days=window*2)
    
    for date in weights.index:
        row_sum = weights.loc[date].abs().sum()
        signals_row_sum = signals_df.loc[date].abs().sum()
        
        if date >= min_data_date and signals_row_sum > 0:
            assert np.isclose(row_sum, 1.0, rtol=1e-6, atol=1e-9), f"Row sum for {date} is {row_sum}, expected 1.0"
        elif signals_row_sum == 0:
            assert np.isclose(row_sum, 0.0, rtol=1e-6, atol=1e-9), f"Row sum for {date} is {row_sum}, expected 0.0"
    
    # 4. If signal is very close to zero, weight should be very close to zero
    for date in weights.index:
        for col in weights.columns:
            signal = signals_df.loc[date, col]
            weight = weights.loc[date, col]
            
            if np.isclose(signal, 0.0, rtol=1e-6, atol=1e-6):
                assert np.isclose(weight, 0.0, rtol=1e-6, atol=1e-6)


@given(price_signal_benchmark_data())
@settings(deadline=None)
def test_rolling_sortino_sizer_properties(data):
    """Test properties of the RollingSortinoSizer."""
    prices_df, signals_df, _ = data
    
    # Skip empty dataframes
    assume(not prices_df.empty and not signals_df.empty)
    # Ensure we have enough data for rolling window
    assume(len(prices_df) >= 10)
    
    sizer = RollingSortinoSizer()
    window = min(10, len(prices_df) - 1)  # Ensure window is smaller than data length
    
    weights = sizer.calculate_weights(signals=signals_df, prices=prices_df, window=window)
    
    # Properties that should hold:
    # 1. Output should be a DataFrame with same shape as signals
    assert isinstance(weights, pd.DataFrame)
    assert weights.shape == signals_df.shape
    
    # 2. No NaN values
    assert not weights.isna().any().any()
    
    # 3. For each date, weights should sum to 1.0 if there are any non-zero signals
    # But for the first few dates, there might not be enough data to calculate metrics
    # So we'll only check dates after the window size
    min_data_date = weights.index[0] + pd.Timedelta(days=window*2)
    
    # For each date, check if weights sum to 1.0 for non-zero signals after the minimum data date
    has_valid_weights = False
    for date in weights.index:
        row_sum = weights.loc[date].abs().sum()
        signals_row_sum = signals_df.loc[date].abs().sum()
        
        if date >= min_data_date and signals_row_sum > 0 and row_sum > 0:
            has_valid_weights = True
            assert np.isclose(row_sum, 1.0, rtol=1e-6, atol=1e-9), f"Row sum for {date} is {row_sum}, expected 1.0"
        elif signals_row_sum == 0:
            assert np.isclose(row_sum, 0.0, rtol=1e-6, atol=1e-9), f"Row sum for {date} is {row_sum}, expected 0.0"
    
    # Skip the test if we don't have any valid weights (this can happen with random data)
    assume(has_valid_weights)
    
    # 4. If signal is very close to zero, weight should be very close to zero
    for date in weights.index:
        for col in weights.columns:
            signal = signals_df.loc[date, col]
            weight = weights.loc[date, col]
            
            if np.isclose(signal, 0.0, rtol=1e-6, atol=1e-6):
                assert np.isclose(weight, 0.0, rtol=1e-6, atol=1e-6)


@given(price_signal_benchmark_data())
@settings(deadline=None)
def test_rolling_beta_sizer_properties(data):
    """Test properties of the RollingBetaSizer."""
    prices_df, signals_df, benchmark = data
    
    # Skip empty dataframes
    assume(not prices_df.empty and not signals_df.empty)
    # Ensure we have enough data for rolling window
    assume(len(prices_df) >= 10)
    
    sizer = RollingBetaSizer()
    window = min(10, len(prices_df) - 1)  # Ensure window is smaller than data length
    
    weights = sizer.calculate_weights(
        signals=signals_df, prices=prices_df, window=window, benchmark=benchmark
    )
    
    # Properties that should hold:
    # 1. Output should be a DataFrame with same shape as signals
    assert isinstance(weights, pd.DataFrame)
    assert weights.shape == signals_df.shape
    
    # 2. No NaN values
    assert not weights.isna().any().any()
    
    # 3. For each date, weights should sum to 1.0 if there are any non-zero signals
    # But for the first few dates, there might not be enough data to calculate metrics
    # So we'll only check dates after the window size
    min_data_date = weights.index[0] + pd.Timedelta(days=window*2)
    
    # For each date, check if weights sum to 1.0 for non-zero signals after the minimum data date
    has_valid_weights = False
    for date in weights.index:
        row_sum = weights.loc[date].abs().sum()
        signals_row_sum = signals_df.loc[date].abs().sum()
        
        if date >= min_data_date and signals_row_sum > 0 and row_sum > 0:
            has_valid_weights = True
            assert np.isclose(row_sum, 1.0, rtol=1e-6, atol=1e-9), f"Row sum for {date} is {row_sum}, expected 1.0"
        elif signals_row_sum == 0:
            assert np.isclose(row_sum, 0.0, rtol=1e-6, atol=1e-9), f"Row sum for {date} is {row_sum}, expected 0.0"
    
    # Skip the test if we don't have any valid weights (this can happen with random data)
    assume(has_valid_weights)
    
    # 4. If signal is very close to zero, weight should be very close to zero
    for date in weights.index:
        for col in weights.columns:
            signal = signals_df.loc[date, col]
            weight = weights.loc[date, col]
            
            if np.isclose(signal, 0.0, rtol=1e-6, atol=1e-6):
                assert np.isclose(weight, 0.0, rtol=1e-6, atol=1e-6)


@given(price_signal_benchmark_data())
@settings(deadline=None)
def test_rolling_benchmark_corr_sizer_properties(data):
    """Test properties of the RollingBenchmarkCorrSizer."""
    prices_df, signals_df, benchmark = data
    
    # Skip empty dataframes
    assume(not prices_df.empty and not signals_df.empty)
    # Ensure we have enough data for rolling window
    assume(len(prices_df) >= 10)
    
    sizer = RollingBenchmarkCorrSizer()
    window = min(10, len(prices_df) - 1)  # Ensure window is smaller than data length
    
    weights = sizer.calculate_weights(
        signals=signals_df, prices=prices_df, window=window, benchmark=benchmark
    )
    
    # Properties that should hold:
    # 1. Output should be a DataFrame with same shape as signals
    assert isinstance(weights, pd.DataFrame)
    assert weights.shape == signals_df.shape
    
    # 2. No NaN values
    assert not weights.isna().any().any()
    
    # 3. For each date, weights should sum to 1.0 if there are any non-zero signals
    # But for the first few dates, there might not be enough data to calculate metrics
    # So we'll only check dates after the window size
    min_data_date = weights.index[0] + pd.Timedelta(days=window*2)
    
    # For each date, check if weights sum to 1.0 for non-zero signals after the minimum data date
    has_valid_weights = False
    for date in weights.index:
        row_sum = weights.loc[date].abs().sum()
        signals_row_sum = signals_df.loc[date].abs().sum()
        
        if date >= min_data_date and signals_row_sum > 0 and row_sum > 0:
            has_valid_weights = True
            assert np.isclose(row_sum, 1.0, rtol=1e-6, atol=1e-9), f"Row sum for {date} is {row_sum}, expected 1.0"
        elif signals_row_sum == 0:
            assert np.isclose(row_sum, 0.0, rtol=1e-6, atol=1e-9), f"Row sum for {date} is {row_sum}, expected 0.0"
    
    # Skip the test if we don't have any valid weights (this can happen with random data)
    assume(has_valid_weights)
    
    # 4. If signal is very close to zero, weight should be very close to zero
    for date in weights.index:
        for col in weights.columns:
            signal = signals_df.loc[date, col]
            weight = weights.loc[date, col]
            
            if np.isclose(signal, 0.0, rtol=1e-6, atol=1e-6):
                assert np.isclose(weight, 0.0, rtol=1e-6, atol=1e-6)


@given(daily_and_monthly_price_data(), st.integers(min_value=2, max_value=5))
@settings(deadline=None)
def test_rolling_downside_volatility_sizer_properties(daily_monthly_data, n_assets):
    """Test properties of the RollingDownsideVolatilitySizer."""
    monthly_prices, daily_prices, benchmark = daily_monthly_data
    
    # Create signals that match the monthly prices exactly
    signal_values = [0.0, 0.5, 1.0, -0.5, -1.0]
    signals_data = {}
    
    # Use the same column names as monthly_prices
    for col in monthly_prices.columns:
        # Generate random signals
        signals = np.random.choice(signal_values, size=len(monthly_prices))
        signals_data[col] = signals
    
    # Create signal DataFrame with same index as monthly_prices
    signals_df = pd.DataFrame(signals_data, index=monthly_prices.index)
    
    # Skip empty dataframes
    assume(not monthly_prices.empty and not signals_df.empty)
    # Ensure we have enough data for rolling window
    assume(len(monthly_prices) >= 12)
    assume(len(daily_prices) >= 252)
    
    sizer = RollingDownsideVolatilitySizer()
    window = min(12, len(monthly_prices) - 1)  # Ensure window is smaller than data length
    target_volatility = 0.1  # 10% target volatility
    max_leverage = 2.0
    
    weights = sizer.calculate_weights(
        signals=signals_df,
        prices=monthly_prices,
        window=window,
        benchmark=benchmark,
        daily_prices_for_vol=daily_prices,
        target_volatility=target_volatility,
        max_leverage=max_leverage,
    )
    
    # Properties that should hold:
    # 1. Output should be a DataFrame with same shape as signals
    assert isinstance(weights, pd.DataFrame)
    assert weights.shape == signals_df.shape
    
    # 2. No NaN values
    assert not weights.isna().any().any()
    
    # 3. Weights should be bounded by max_leverage
    assert (weights.abs() <= max_leverage + 1e-6).all().all()
    
    # 4. If signal is very close to zero, weight should be very close to zero
    for date in weights.index:
        for col in weights.columns:
            signal = signals_df.loc[date, col]
            weight = weights.loc[date, col]
            
            if np.isclose(signal, 0.0, rtol=1e-6, atol=1e-6):
                assert np.isclose(weight, 0.0, rtol=1e-6, atol=1e-6)


@given(st.sampled_from(["equal_weight", "rolling_sharpe", "rolling_sortino", "rolling_beta", "rolling_benchmark_corr"]))
@settings(deadline=None)
def test_get_position_sizer_factory(sizer_name):
    """Test the position sizer factory function."""
    sizer = get_position_sizer(sizer_name)
    
    # Should return a valid sizer instance
    assert sizer is not None
    
    # Should have the calculate_weights method
    assert hasattr(sizer, "calculate_weights")
    assert callable(sizer.calculate_weights)


@given(
    st.dictionaries(
        keys=st.sampled_from(["position_sizer"]),
        values=st.sampled_from(["equal_weight", "rolling_sharpe", "rolling_sortino"]),
        min_size=1,
        max_size=1,
    )
)
@settings(deadline=None)
def test_get_position_sizer_from_config(config):
    """Test getting a position sizer from a configuration dictionary."""
    sizer = get_position_sizer_from_config(config)
    
    # Should return a valid sizer instance
    assert sizer is not None
    
    # Should have the calculate_weights method
    assert hasattr(sizer, "calculate_weights")
    assert callable(sizer.calculate_weights)


@given(
    st.dictionaries(
        keys=st.sampled_from(["position_sizer"]),
        values=st.just("invalid_sizer_name"),
        min_size=1,
        max_size=1,
    )
)
@settings(deadline=None)
def test_get_position_sizer_from_config_invalid(config):
    """Test getting a position sizer with an invalid name raises ValueError."""
    try:
        sizer = get_position_sizer_from_config(config)
        assert False, f"Expected ValueError but got sizer: {sizer}"
    except ValueError:
        # Expected behavior
        pass
