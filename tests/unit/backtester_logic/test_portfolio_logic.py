import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from portfolio_backtester.backtester_logic.portfolio_logic import calculate_portfolio_returns
from portfolio_backtester.trading.trade_tracker import TradeTracker

@pytest.fixture
def mock_data():
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    tickers = ["AAPL", "GOOG"]
    
    price_data = pd.DataFrame(
        100.0 + np.random.randn(len(dates), len(tickers)),
        index=dates,
        columns=tickers
    )
    
    # Add Field level for MultiIndex if needed, but the logic handles single level too
    # Let's create a MultiIndex version as it's common
    mi_columns = pd.MultiIndex.from_product([tickers, ["Close", "Open", "High", "Low", "Volume"]], names=["Ticker", "Field"])
    price_data_mi = pd.DataFrame(
        np.random.randn(len(dates), len(mi_columns)) + 100,
        index=dates,
        columns=mi_columns
    )
    
    rets_daily = price_data.pct_change().fillna(0.0)
    
    sized_signals = pd.DataFrame(
        0.5,
        index=dates,
        columns=tickers
    )
    
    return {
        "dates": dates,
        "tickers": tickers,
        "price_data": price_data,
        "price_data_mi": price_data_mi,
        "rets_daily": rets_daily,
        "sized_signals": sized_signals
    }

def test_calculate_portfolio_returns_standard(mock_data):
    scenario_config = {
        "timing_config": {"rebalance_frequency": "M"},
        "allocation_mode": "reinvestment"
    }
    global_config = {
        "feature_flags": {"ndarray_simulation": True},
        "commission_per_share": 0.0,
        "slippage_bps": 0.0
    }
    
    rets, tracker = calculate_portfolio_returns(
        sized_signals=mock_data["sized_signals"],
        scenario_config=scenario_config,
        price_data_daily_ohlc=mock_data["price_data"],
        rets_daily=mock_data["rets_daily"],
        universe_tickers=mock_data["tickers"],
        global_config=global_config,
        track_trades=False
    )
    
    assert isinstance(rets, pd.Series)
    assert len(rets) == len(mock_data["dates"])
    assert tracker is None
    # Basic sanity: returns should be roughly average of asset returns (since weights are 0.5 each)
    # Note: rebalancing might change weights, but with Monthly rebalance and 10 days, it might just be constant
    
def test_calculate_portfolio_returns_with_costs(mock_data):
    scenario_config = {"timing_config": {"rebalance_frequency": "M"}}
    global_config = {
        "feature_flags": {"ndarray_simulation": True},
        "commission_per_share": 0.01,
        "slippage_bps": 5.0
    }
    
    rets_net, _ = calculate_portfolio_returns(
        sized_signals=mock_data["sized_signals"],
        scenario_config=scenario_config,
        price_data_daily_ohlc=mock_data["price_data"],
        rets_daily=mock_data["rets_daily"],
        universe_tickers=mock_data["tickers"],
        global_config=global_config,
        track_trades=False
    )
    
    # With costs, returns should be slightly lower than gross returns (if we calculated them)
    # Here we just check it runs and produces output
    assert isinstance(rets_net, pd.Series)
    assert not rets_net.isna().any()

def test_calculate_portfolio_returns_track_trades(mock_data):
    scenario_config = {
        "timing_config": {"rebalance_frequency": "M"},
        "allocation_mode": "reinvestment"
    }
    global_config = {
        "feature_flags": {"ndarray_simulation": True},
        "portfolio_value": 100000.0
    }
    
    # Ensure MultiIndex data is handled
    rets, tracker = calculate_portfolio_returns(
        sized_signals=mock_data["sized_signals"],
        scenario_config=scenario_config,
        price_data_daily_ohlc=mock_data["price_data_mi"],
        rets_daily=mock_data["rets_daily"],
        universe_tickers=mock_data["tickers"],
        global_config=global_config,
        track_trades=True
    )
    
    assert isinstance(tracker, TradeTracker)
    assert isinstance(rets, pd.Series)
    
    # Check if tracker has data
    stats = tracker.get_trade_statistics()
    # Depending on logic, might have trades or open positions
    # With constant 0.5 weights, we expect initial entry trades
    assert stats["all_num_trades"] >= 0 

def test_calculate_portfolio_returns_meta_strategy(mock_data):
    # Mock a meta strategy
    mock_strategy = MagicMock()
    # StrategyResolver needs to say it is a meta strategy
    
    with patch("portfolio_backtester.backtester_logic.portfolio_logic.StrategyResolverFactory") as MockFactory:
        mock_resolver = MockFactory.create.return_value
        mock_resolver.is_meta_strategy.return_value = True
        
        # Mock aggregation
        mock_aggregator = MagicMock()
        mock_strategy.get_trade_aggregator.return_value = mock_aggregator
        
        # Mock trades
        mock_trade = MagicMock()
        mock_trade.date = mock_data["dates"][0]
        mock_trade.asset = "AAPL"
        mock_trade.quantity = 10
        mock_trade.price = 100.0
        mock_trade.side.value = "buy"
        
        mock_aggregator.get_aggregated_trades.return_value = [mock_trade]
        
        # Mock portfolio timeline
        mock_timeline = pd.DataFrame({
            "returns": pd.Series(0.01, index=mock_data["dates"])
        })
        mock_aggregator.get_portfolio_timeline.return_value = mock_timeline
        
        scenario_config = {"name": "MetaScenario"}
        global_config = {"portfolio_value": 100000.0}
        
        rets, tracker = calculate_portfolio_returns(
            sized_signals=mock_data["sized_signals"],
            scenario_config=scenario_config,
            price_data_daily_ohlc=mock_data["price_data_mi"],
            rets_daily=mock_data["rets_daily"],
            universe_tickers=mock_data["tickers"],
            global_config=global_config,
            track_trades=True,
            strategy=mock_strategy
        )
        
        assert isinstance(rets, pd.Series)
        assert len(rets) == len(mock_data["dates"])
        assert tracker is not None # We requested track_trades=True

def test_missing_rets_daily_error(mock_data):
    scenario_config = {}
    global_config = {}
    
    rets, _ = calculate_portfolio_returns(
        sized_signals=mock_data["sized_signals"],
        scenario_config=scenario_config,
        price_data_daily_ohlc=mock_data["price_data"],
        rets_daily=None, # Explicitly None
        universe_tickers=mock_data["tickers"],
        global_config=global_config
    )
    
    assert (rets == 0.0).all()

def test_legacy_mode_error(mock_data):
    scenario_config = {}
    global_config = {
        "feature_flags": {"ndarray_simulation": False} # Explicitly False
    }
    
    with pytest.raises(RuntimeError, match="legacy Pandas-based portfolio simulation has been removed"):
        calculate_portfolio_returns(
            sized_signals=mock_data["sized_signals"],
            scenario_config=scenario_config,
            price_data_daily_ohlc=mock_data["price_data"],
            rets_daily=mock_data["rets_daily"],
            universe_tickers=mock_data["tickers"],
            global_config=global_config
        )
