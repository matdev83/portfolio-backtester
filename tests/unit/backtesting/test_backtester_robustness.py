import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def corrupted_data():
    """Provides data with NaNs, Infs, and misalignment."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    
    # Prices with NaNs and Infs
    prices = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    prices.iloc[2, 0] = np.nan
    prices.iloc[3, 0] = np.inf
    prices.iloc[4, 0] = -np.inf
    prices.iloc[5, 0] = 0.0 # Zero price edge case
    
    # Returns with Extreme values
    rets = prices.pct_change().fillna(0.0)
    rets.iloc[6, 0] = 1000.0 # 100000% return
    
    return {
        "monthly": prices.resample("ME").last().ffill(),
        "daily": prices,
        "rets": rets
    }

@pytest.fixture
def robust_backtester():
    global_config = {"benchmark": "SPY"}
    data_source = MagicMock()
    return StrategyBacktester(global_config, data_source)

# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

def test_backtest_with_corrupted_prices(robust_backtester, corrupted_data):
    """Test backtester resilience against NaNs, Infs, and Zero prices."""
    config = {
        "strategy": "TestStrategy",
        "strategy_params": {},
        "name": "CorruptTest",
        "universe": ["A", "B"]
    }
    
    with patch.object(robust_backtester, "_get_strategy") as mock_get_strat, \
         patch("portfolio_backtester.backtesting.strategy_backtester.generate_signals") as mock_sigs, \
         patch("portfolio_backtester.backtesting.strategy_backtester.size_positions") as mock_size, \
         patch("portfolio_backtester.backtesting.strategy_backtester.calculate_portfolio_returns") as mock_calc:
        
        # Mock strategy
        mock_strat = MagicMock()
        mock_strat.get_universe.return_value = [("A", 1.0), ("B", 1.0)]
        mock_get_strat.return_value = mock_strat
        
        # Mock logic to proceed despite bad data
        mock_sigs.return_value = pd.DataFrame(0.5, index=corrupted_data["daily"].index, columns=["A", "B"])
        mock_size.return_value = pd.DataFrame(0.5, index=corrupted_data["daily"].index, columns=["A", "B"])
        
        # calculate_portfolio_returns usually handles math, but let's see if it crashes before calling it
        # or if we mock it to return something valid, ensuring pipeline integrity
        mock_calc.return_value = (
            pd.Series(0.01, index=corrupted_data["daily"].index),
            None # trade_tracker
        )
        
        result = robust_backtester.backtest_strategy(
            config,
            corrupted_data["monthly"],
            corrupted_data["daily"],
            corrupted_data["rets"]
        )
        
        assert not result.returns.empty
        # Ensure it didn't crash

def test_backtest_with_misaligned_dates(robust_backtester):
    """Test handling of disjoint daily vs monthly data."""
    daily = pd.DataFrame(100.0, index=pd.date_range("2023-01-01", periods=10), columns=["A", "SPY"])
    monthly = pd.DataFrame(100.0, index=pd.date_range("2022-01-01", periods=10, freq="ME"), columns=["A", "SPY"])
    rets = daily.pct_change().fillna(0.0)
    
    config = {"strategy": "Test", "strategy_params": {}, "universe": ["A"]}
    
    with patch.object(robust_backtester, "_get_strategy") as mock_get, \
         patch("portfolio_backtester.backtesting.strategy_backtester.calculate_portfolio_returns") as mock_calc:
        
        mock_strat = MagicMock()
        mock_strat.get_universe.return_value = [("A", 1.0)]
        mock_get.return_value = mock_strat
        # Mock return to avoid 'None' crash if it gets that far
        mock_calc.return_value = (pd.Series(0.0, index=daily.index), None)
        
        # Should likely proceed but might align data internally or produce warnings
        result = robust_backtester.backtest_strategy(config, monthly, daily, rets)
        
        assert not result.returns.empty

def test_universe_collection_failure(robust_backtester, corrupted_data):
    """Test resilience when universe collection raises exception."""
    config = {"strategy": "Test", "strategy_params": {}, "universe": "INVALID_FILE"}
    
    with patch.object(robust_backtester, "_get_strategy") as mock_get_strat, \
         patch("portfolio_backtester.interfaces.ticker_collector.TickerCollectorFactory.create_collector") as mock_factory:
        
        mock_strat = MagicMock()
        mock_strat.get_universe.return_value = [] # Return empty universe to test the empty path
        mock_get_strat.return_value = mock_strat
        mock_factory.side_effect = Exception("File Read Error")
        
        # Should catch exception and log error, then proceed with empty universe or handle gracefully
        # Implementation: try/except -> universe_tickers = [] -> _create_empty_backtest_result
        
        result = robust_backtester.backtest_strategy(
            config, corrupted_data["monthly"], corrupted_data["daily"], corrupted_data["rets"]
        )
        
        assert result.returns.empty
        assert result.metrics == {}

def test_missing_all_tickers(robust_backtester):
    """Test scenario where requested universe tickers are entirely missing from data."""
    daily = pd.DataFrame(100.0, index=pd.date_range("2023-01-01", periods=5), columns=["A", "B"])
    config = {"strategy": "Test", "strategy_params": {}, "universe": ["X", "Y"]}
    
    with patch.object(robust_backtester, "_get_strategy"):
        result = robust_backtester.backtest_strategy(
            config, daily, daily, daily # monthly/rets same for structure
        )
        
        # Should detect missing tickers -> filter -> empty -> return empty result
        assert result.returns.empty

def test_signal_generation_failure(robust_backtester, corrupted_data):
    """Test resilience when signal generation crashes."""
    config = {"strategy": "Test", "strategy_params": {}, "universe": ["A"]}
    
    with patch.object(robust_backtester, "_get_strategy") as mock_get_strat, \
         patch("portfolio_backtester.backtesting.strategy_backtester.generate_signals") as mock_sigs:
        
        mock_strat = MagicMock()
        mock_strat.get_universe.return_value = [("A", 1.0)]
        mock_get_strat.return_value = mock_strat
        
        mock_sigs.side_effect = ValueError("Strategy Logic Error")
        
        with pytest.raises(ValueError, match="Strategy Logic Error"):
            robust_backtester.backtest_strategy(
                config, corrupted_data["monthly"], corrupted_data["daily"], corrupted_data["rets"]
            )
        # Note: Current implementation might not catch this inside backtest_strategy 
        # but let it bubble up. That is acceptable design, verifying it here.

def test_trade_history_creation_bad_types(robust_backtester):
    """Test trade history creation with non-numeric positions (garbage data)."""
    dates = pd.date_range("2023-01-01", periods=2)
    daily = pd.DataFrame(100.0, index=dates, columns=["A"])
    
    # Sized signals with string/garbage
    sized = pd.DataFrame({"A": ["INVALID", 0.5]}, index=dates)
    
    history = robust_backtester._create_trade_history(sized, daily)
    
    # Should skip the "INVALID" row and process the 0.5 row
    assert len(history) == 1
    assert history.iloc[0]["position"] == 0.5
