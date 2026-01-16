import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from portfolio_backtester.trading.trade_tracker import TradeTracker, Trade

@pytest.fixture
def tracker():
    return TradeTracker(initial_portfolio_value=100000.0, allocation_mode="reinvestment")

def test_tracker_initialization(tracker):
    assert tracker.initial_portfolio_value == 100000.0
    assert tracker.allocation_mode == "reinvestment"
    assert tracker.current_portfolio_value == 100000.0

def test_populate_from_kernel_results(tracker):
    dates = pd.date_range("2023-01-01", periods=3)
    portfolio_values = pd.Series([100000.0, 101000.0, 102000.0], index=dates)
    positions = pd.DataFrame(np.zeros((3, 2)), index=dates, columns=["A", "B"])
    prices = pd.DataFrame(np.ones((3, 2)) * 100, index=dates, columns=["A", "B"])
    tickers = np.array(["A", "B"])
    
    # Create structured array for completed trades
    dtype = [
        ("ticker_idx", "i8"),
        ("entry_date", "i8"),
        ("exit_date", "i8"),
        ("entry_price", "f8"),
        ("exit_price", "f8"),
        ("quantity", "f8"),
        ("pnl", "f8"),
        ("commission", "f8"),
    ]
    completed_trades = np.zeros(1, dtype=dtype)
    completed_trades[0] = (
        0, # ticker_idx A
        dates[0].value,
        dates[1].value,
        100.0,
        110.0,
        10.0,
        100.0, # Gross PnL
        5.0 # Commission
    )
    
    tracker.populate_from_kernel_results(
        portfolio_values, positions, completed_trades, tickers, prices
    )
    
    # Verify portfolio values set
    assert tracker.current_portfolio_value == 102000.0
    
    # Verify trades populated
    trades = tracker.trade_lifecycle_manager.get_completed_trades()
    assert len(trades) == 1
    t = trades[0]
    assert t.ticker == "A"
    assert t.pnl_gross == 100.0
    assert t.commission_exit == 5.0
    assert t.pnl_net == 95.0

def test_update_positions_logic(tracker):
    date = pd.Timestamp("2023-01-01")
    new_weights = pd.Series({"A": 0.5}, dtype=float)
    prices = pd.Series({"A": 100.0}, dtype=float)
    commissions = {"A": 1.0}
    
    # Mock lifecycle manager interactions to verify calls
    with patch.object(tracker.trade_lifecycle_manager, "open_position") as mock_open, \
         patch.object(tracker.trade_lifecycle_manager, "close_position") as mock_close, \
         patch.object(tracker.portfolio_value_tracker, "update_daily_metrics") as mock_update_metrics:
        
        tracker.update_positions(date, new_weights, prices, commissions)
        
        # Should attempt to open position for A
        # Quantity = 0.5 * 100000 / 100 = 500
        mock_open.assert_called_once()
        args, _ = mock_open.call_args
        assert args[1] == "A" # ticker
        assert args[2] == 500.0 # quantity
        
        mock_update_metrics.assert_called_once()

def test_get_trade_statistics_integration(tracker):
    # Setup some trades
    trade = Trade(
        ticker="A",
        entry_date=pd.Timestamp("2023-01-01"),
        exit_date=pd.Timestamp("2023-01-02"),
        entry_price=100.0,
        exit_price=110.0,
        quantity=10,
        entry_value=1000.0,
        pnl_net=95.0,
        pnl_gross=100.0,
        commission_entry=2.5,
        commission_exit=2.5,
        mfe=10.0,
        mae=0.0
    )
    tracker.trade_lifecycle_manager.trades.append(trade)
    
    stats = tracker.get_trade_statistics()
    assert stats["all_num_trades"] == 1
    assert stats["all_total_pnl_net"] == 95.0
    assert stats["initial_capital"] == 100000.0

def test_close_all_positions(tracker):
    date = pd.Timestamp("2023-01-10")
    prices = pd.Series({"A": 120.0})
    
    with patch.object(tracker.trade_lifecycle_manager, "close_all_positions") as mock_close_all, \
         patch.object(tracker.portfolio_value_tracker, "update_portfolio_value") as mock_update_val:
        
        mock_trade = MagicMock()
        mock_trade.pnl_net = 50.0
        mock_close_all.return_value = [mock_trade]
        
        tracker.close_all_positions(date, prices)
        
        mock_close_all.assert_called_once_with(date, prices, None)
        mock_update_val.assert_called_once_with(50.0)
