import pandas as pd
import numpy as np

from portfolio_backtester.strategies.base.trade_aggregator import TradeAggregator
from portfolio_backtester.strategies.base.trade_record import TradeRecord, TradeSide


def _make_trade(date, qty, price, side):
    return TradeRecord(
        date=pd.Timestamp(date),
        asset="SPY",
        quantity=qty,
        price=price,
        side=side,
        strategy_id="s1",
        allocated_capital=10000,
    )


def test_no_trades_returns_initial_capital():
    agg = TradeAggregator(initial_capital=10000)
    val = agg.calculate_portfolio_value(pd.Timestamp("2020-01-01"))
    assert val == 10000


def test_buy_and_sell_closes_position_and_updates_pnl():
    agg = TradeAggregator(initial_capital=10000)
    buy = _make_trade("2020-01-01", qty=10, price=100, side=TradeSide.BUY)
    agg.track_sub_strategy_trade(buy)
    # After buy, portfolio value same (no mark to market, position held)
    val_after_buy = agg.calculate_portfolio_value(pd.Timestamp("2020-01-01"))
    assert val_after_buy == 10000  # cash reduced but asset value added

    sell = _make_trade("2020-01-10", qty=-10, price=110, side=TradeSide.SELL)
    agg.track_sub_strategy_trade(sell)
    val_after_sell = agg.calculate_portfolio_value(pd.Timestamp("2020-01-10"))
    # P&L should be 10*10 = 100
    assert np.isclose(val_after_sell, 10100)


def test_weighted_performance_keys_present():
    agg = TradeAggregator(initial_capital=10000)
    keys = agg.calculate_weighted_performance().keys()
    for k in ["total_return","total_pnl"]:
        assert k in keys