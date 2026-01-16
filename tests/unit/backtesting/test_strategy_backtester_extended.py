import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester

@pytest.fixture
def backtester():
    global_config = {"benchmark": "A"}
    data_source = MagicMock()
    return StrategyBacktester(global_config, data_source)

@pytest.fixture
def mock_data():
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    data = pd.DataFrame(
        100.0,
        index=dates,
        columns=["A", "B"]
    )
    return {
        "daily": data,
        "monthly": data,
        "rets": data.pct_change().fillna(0.0)
    }

def test_backtest_strategy_missing_universe(backtester, mock_data):
    # Strategy with no universe
    config = {
        "strategy": "TestStrategy",
        "strategy_params": {},
        "name": "Test"
    }
    
    with patch.object(backtester, "_get_strategy") as mock_get_strat:
        mock_strat = MagicMock()
        # Returns empty universe list
        mock_strat.get_universe.return_value = []
        mock_get_strat.return_value = mock_strat
        
        result = backtester.backtest_strategy(
            config,
            mock_data["monthly"],
            mock_data["daily"],
            mock_data["rets"]
        )
        
        # Should return empty result
        assert result.returns.empty
        assert result.metrics == {}

def test_backtest_strategy_empty_returns(backtester, mock_data):
    config = {
        "strategy": "TestStrategy",
        "strategy_params": {},
        "name": "Test",
        "universe": ["A"]
    }
    
    with patch.object(backtester, "_get_strategy"), \
         patch("portfolio_backtester.backtesting.strategy_backtester.generate_signals"), \
         patch("portfolio_backtester.backtesting.strategy_backtester.size_positions"), \
         patch("portfolio_backtester.backtesting.strategy_backtester.calculate_portfolio_returns") as mock_calc:
        
        # Return empty/None returns
        mock_calc.return_value = (None, None)
        
        result = backtester.backtest_strategy(
            config,
            mock_data["monthly"],
            mock_data["daily"],
            mock_data["rets"]
        )
        
        assert result.returns.empty
        assert "Total Return" not in result.metrics # Empty metrics dict

def test_create_trade_history_fallback(backtester):
    # Test _create_trade_history logic
    dates = pd.date_range("2023-01-01", periods=2)
    daily = pd.DataFrame(100.0, index=dates, columns=["A"])
    
    sized = pd.DataFrame({"A": [0.5, 0.0]}, index=dates)
    
    history = backtester._create_trade_history(sized, daily)
    
    assert len(history) == 1
    assert history.iloc[0]["ticker"] == "A"
    assert history.iloc[0]["position"] == 0.5
    assert history.iloc[0]["price"] == 100.0

def test_create_performance_stats_empty(backtester):
    # Empty returns
    stats = backtester._create_performance_stats(pd.Series(dtype=float), {})
    assert stats["total_return"] == 0.0
    assert stats["max_drawdown"] == 0.0

def test_calculate_rolling_sharpe(backtester):
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    rets = pd.Series(np.random.normal(0.001, 0.01, 300), index=dates)
    
    sharpe = backtester._calculate_rolling_sharpe(rets, window=252)
    
    # First 251 should be 0/NaN (filled to 0)
    assert (sharpe.iloc[:251] == 0).all()
    # Last one should be calculated
    assert sharpe.iloc[-1] != 0

def test_run_scenario_for_window_failure(backtester, mock_data):
    # Test window failure path
    config = {"strategy": "Test", "strategy_params": {}, "universe": ["MISSING"]}
    
    with patch.object(backtester, "_get_strategy") as mock_get_strat:
        # Return a dummy strategy
        mock_get_strat.return_value = MagicMock()
        mock_get_strat.return_value.get_universe.return_value = ["MISSING"]

        # Missing tickers -> returns None
        result = backtester._run_scenario_for_window(
            config,
            mock_data["monthly"],
            mock_data["daily"]
        )
        
        assert result is None
