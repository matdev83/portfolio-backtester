import pandas as pd
import numpy as np
import pytest
from src.portfolio_backtester.strategies.builtins.portfolio.drift_regime_conditional_factor_portfolio_strategy import (
    DriftRegimeConditionalFactorPortfolioStrategy,
)

@pytest.fixture
def drift_test_data():
    """Generate test data with a clear drift regime for some stocks."""
    dates = pd.date_range(start="2020-01-01", periods=300, freq="B")
    tickers = ["TrendStock", "MeanStock", "LoserStock"]
    
    data_frames = []
    for ticker in tickers:
        if ticker == "TrendStock":
            # Consistent uptrend (positive drift)
            # Drift for TrendStock should be > 0.6
            returns = np.random.normal(0.002, 0.01, size=len(dates))
        elif ticker == "MeanStock":
            # Sideways
            returns = np.random.normal(0, 0.01, size=len(dates))
        else:
            # Downtrend
            returns = np.random.normal(-0.002, 0.01, size=len(dates))
            
        prices = (1 + returns).cumprod() * 100
        
        df = pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": 1000
        }, index=dates)
        
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns], names=["Ticker", "Field"])
        data_frames.append(df)
        
    all_data = pd.concat(data_frames, axis=1)
    
    # Mock benchmark
    bench_returns = np.random.normal(0.0005, 0.01, size=len(dates))
    bench_prices = (1 + bench_returns).cumprod() * 100
    benchmark_df = pd.DataFrame({
        "Open": bench_prices * 0.99,
        "High": bench_prices * 1.01,
        "Low": bench_prices * 0.98,
        "Close": bench_prices,
        "Volume": 10000
    }, index=dates)
    benchmark_df.columns = pd.MultiIndex.from_product([["SPY"], benchmark_df.columns], names=["Ticker", "Field"])
    
    return {
        "dates": dates,
        "all_data": all_data,
        "benchmark_data": benchmark_df
    }

class TestDriftRegimeConditionalFactorPortfolioStrategy:
    
    def test_initialization(self):
        config = {"strategy_params": {"drift_window": 63}}
        strategy = DriftRegimeConditionalFactorPortfolioStrategy(config)
        params = strategy.strategy_params.get("strategy_params", strategy.strategy_params)
        assert params["drift_window"] == 63
        assert params["drift_threshold"] == 0.6
        
    def test_generate_signals_basic(self, drift_test_data):
        config = {
            "strategy_params": {
                "drift_window": 63,
                "drift_threshold": 0.5, # Lower threshold to ensure we get signals
                "num_holdings": 2,
                "reversal_window": 21,
                "value_window": 126
            }
        }
        strategy = DriftRegimeConditionalFactorPortfolioStrategy(config)
        
        current_date = drift_test_data["dates"][-1]
        signals = strategy.generate_signals(
            drift_test_data["all_data"],
            drift_test_data["benchmark_data"],
            current_date=current_date
        )
        
        assert isinstance(signals, pd.DataFrame)
        assert signals.index[0] == current_date
        assert not signals.empty
        
    def test_insufficient_data(self, drift_test_data):
        config = {"strategy_params": {"drift_window": 300, "value_window": 300}}
        strategy = DriftRegimeConditionalFactorPortfolioStrategy(config)
        
        # Only 300 days available, so 300-day windows might fail or be borderline
        # Let's use an even earlier date
        current_date = drift_test_data["dates"][50]
        signals = strategy.generate_signals(
            drift_test_data["all_data"],
            drift_test_data["benchmark_data"],
            current_date=current_date
        )
        
        assert (signals == 0).all().all()
        
    def test_drift_threshold_filtering(self, drift_test_data):
        # Set threshold very high to ensure NO stocks pass (unless extremely lucky)
        config = {
            "strategy_params": {
                "drift_threshold": 0.99,
                "drift_window": 63
            }
        }
        strategy = DriftRegimeConditionalFactorPortfolioStrategy(config)
        
        current_date = drift_test_data["dates"][-1]
        signals = strategy.generate_signals(
            drift_test_data["all_data"],
            drift_test_data["benchmark_data"],
            current_date=current_date
        )
        
    # Should be zero as no stock is likely to have 99% positive days
        assert (signals == 0).all().all()

    def test_long_short_signals(self, drift_test_data):
        # 1. Test Long/Short (Default)
        config_ls = {
            "strategy_params": {
                "drift_threshold": 0.4, # Lower to ensure enough candidates
                "num_holdings": 1,
                "trade_longs": True,
                "trade_shorts": True
            }
        }
        strategy_ls = DriftRegimeConditionalFactorPortfolioStrategy(config_ls)
        current_date = drift_test_data["dates"][-1]
        signals_ls = strategy_ls.generate_signals(
            drift_test_data["all_data"],
            drift_test_data["benchmark_data"],
            current_date=current_date
        )
        
        # Should have both positive and negative weights (if enough candidates)
        assert (signals_ls > 0).any().any()
        assert (signals_ls < 0).any().any()
        # Sum should be near 0 (dollar neutral style with equal holdings)
        # Note: if only one side has candidates, sum will follow that side.
        # But in our test data, we should have multiple.
        
        # 2. Test Long Only
        config_lo = {
            "strategy_params": {
                "drift_threshold": 0.4,
                "num_holdings": 1,
                "trade_longs": True,
                "trade_shorts": False
            }
        }
        strategy_lo = DriftRegimeConditionalFactorPortfolioStrategy(config_lo)
        signals_lo = strategy_lo.generate_signals(
            drift_test_data["all_data"],
            drift_test_data["benchmark_data"],
            current_date=current_date
        )
        assert (signals_lo >= 0).all().all()
        assert (signals_lo > 0).any().any()
        
        # 3. Test Short Only
        config_so = {
            "strategy_params": {
                "drift_threshold": 0.4,
                "num_holdings": 1,
                "trade_longs": False,
                "trade_shorts": True
            }
        }
        strategy_so = DriftRegimeConditionalFactorPortfolioStrategy(config_so)
        signals_so = strategy_so.generate_signals(
            drift_test_data["all_data"],
            drift_test_data["benchmark_data"],
            current_date=current_date
        )
        assert (signals_so <= 0).all().all()
        assert (signals_so < 0).any().any()

