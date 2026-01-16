import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from portfolio_backtester.backtesting.window_evaluator import WindowEvaluator
from portfolio_backtester.optimization.wfo_window import WFOWindow
from portfolio_backtester.backtesting.results import WindowResult

@pytest.fixture
def mock_backtester():
    backtester = MagicMock()
    # Mock BacktestResult
    result = MagicMock()
    result.returns = pd.Series([0.01, 0.02])
    result.trade_history = pd.DataFrame()
    result.performance_stats = {"final_weights": {"A": 0.5}}
    result.metrics = {"sharpe": 2.0}
    backtester.backtest_strategy.return_value = result
    return backtester

def test_window_evaluator_prepare_caching(mock_backtester):
    evaluator = WindowEvaluator(backtester=mock_backtester)
    
    window = WFOWindow(
        train_start=pd.Timestamp("2023-01-01"),
        train_end=pd.Timestamp("2023-01-10"),
        test_start=pd.Timestamp("2023-01-11"),
        test_end=pd.Timestamp("2023-01-20")
    )
    
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    daily_data = pd.DataFrame(index=dates, columns=["A"])
    universe = ["A"]
    
    # First call
    eval_dates, is_mi, tickers = evaluator._prepare_window_evaluation(window, daily_data, universe)
    assert not is_mi
    assert tickers == ["A"]
    
    # Second call - should use cache
    with patch.object(window, "get_evaluation_dates", wraps=window.get_evaluation_dates) as mock_get:
        evaluator._prepare_window_evaluation(window, daily_data, universe)
        assert mock_get.call_count == 0

def test_evaluate_window_basic(mock_backtester):
    evaluator = WindowEvaluator(backtester=mock_backtester)
    window = WFOWindow(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02"), 
                       pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04"))
    
    strategy = MagicMock()
    strategy.config = {"param": 1}
    
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    daily_data = pd.DataFrame(index=dates, columns=pd.MultiIndex.from_product([["A"], ["Close"]]))
    
    result = evaluator.evaluate_window(
        window, strategy, daily_data, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ["A"], "SPY"
    )
    
    assert isinstance(result, WindowResult)
    assert result.metrics["sharpe"] == 2.0
    assert result.final_weights == {"A": 0.5}

def test_get_current_prices_multiindex():
    evaluator = WindowEvaluator(backtester=MagicMock())
    dates = pd.date_range("2023-01-01", periods=2)
    cols = pd.MultiIndex.from_product([["A", "B"], ["Close"]])
    df = pd.DataFrame([[10.0, 20.0], [11.0, 21.0]], index=dates, columns=cols)
    
    prices = evaluator._get_current_prices(df, dates[0], is_multiindex=True)
    assert prices["A"] == 10.0
    assert prices["B"] == 20.0
