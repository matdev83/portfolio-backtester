import pytest
import numpy as np
import pandas as pd
from portfolio_backtester.trading.trade_statistics_calculator import TradeStatisticsCalculator
from portfolio_backtester.trading.trade_lifecycle_manager import Trade

@pytest.fixture
def calculator():
    return TradeStatisticsCalculator()

def create_trade(pnl=100.0, winner=True, quantity=10):
    return Trade(
        ticker="A",
        entry_date=pd.Timestamp("2023-01-01"),
        exit_date=pd.Timestamp("2023-01-02"),
        entry_price=100.0,
        exit_price=110.0 if winner else 90.0,
        quantity=quantity,
        entry_value=1000.0,
        pnl_net=pnl,
        pnl_gross=pnl + 5.0, # simplistic
        commission_entry=2.5,
        commission_exit=2.5,
        mfe=0.0,
        mae=0.0,
        is_winner=winner
    )

def test_calculate_statistics_empty(calculator):
    stats = calculator.calculate_statistics([], 100000.0, "reinvestment")
    assert stats["all_num_trades"] == 0
    assert stats["allocation_mode"] == "reinvestment"

def test_calculate_statistics_basic(calculator):
    trades = [
        create_trade(pnl=100.0, winner=True, quantity=10),
        create_trade(pnl=-50.0, winner=False, quantity=10)
    ]
    
    stats = calculator.calculate_statistics(trades, 100000.0, "reinvestment")
    
    assert stats["all_num_trades"] == 2
    assert stats["all_num_winners"] == 1
    assert stats["all_num_losers"] == 1
    assert stats["all_win_rate_pct"] == 50.0
    assert stats["all_total_pnl_net"] == 50.0
    assert stats["all_largest_profit"] == 100.0
    assert stats["all_largest_loss"] == -50.0

def test_calculate_statistics_directional(calculator):
    trades = [
        create_trade(pnl=100.0, winner=True, quantity=10), # Long
        create_trade(pnl=50.0, winner=True, quantity=-10)  # Short winner
    ]
    
    stats = calculator.calculate_statistics(trades, 100000.0, "reinvestment")
    
    assert stats["long_num_trades"] == 1
    assert stats["short_num_trades"] == 1
    assert stats["long_total_pnl_net"] == 100.0
    assert stats["short_total_pnl_net"] == 50.0

def test_calculate_statistics_edge_cases(calculator):
    # Trade with no exit date (should be filtered out)
    open_trade = create_trade()
    open_trade.exit_date = None
    
    stats = calculator.calculate_statistics([open_trade], 100000.0, "reinvestment")
    assert stats["all_num_trades"] == 0
    
    # Division by zero guards
    # Losing trade with 0 mean loss? Impossible if pnl < 0.
    # But if mean loss is 0 (no losers), reward risk ratio handles it
    trades = [create_trade(pnl=100.0, winner=True)]
    stats = calculator.calculate_statistics(trades, 100000.0, "reinvestment")
    assert stats["all_reward_risk_ratio"] == np.inf
    
    # Information score with std=0 (single trade)
    assert stats["all_information_score"] == 0.0

def test_trades_per_month(calculator):
    t1 = create_trade()
    t1.entry_date = pd.Timestamp("2023-01-01")
    t1.exit_date = pd.Timestamp("2023-01-02")
    
    t2 = create_trade()
    t2.entry_date = pd.Timestamp("2023-02-01")
    t2.exit_date = pd.Timestamp("2023-02-02") # ~1 month later
    
    stats = calculator.calculate_statistics([t1, t2], 100000.0, "reinvestment")
    # Span is approx 32 days ~ 1.05 months. 2 trades / 1.05 ~ 1.9
    assert 1.0 < stats["all_trades_per_month"] < 3.0
