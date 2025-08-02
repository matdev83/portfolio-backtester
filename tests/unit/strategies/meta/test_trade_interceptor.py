"""Tests for MetaStrategyTradeInterceptor class."""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock

from src.portfolio_backtester.strategies.base.trade_interceptor import MetaStrategyTradeInterceptor
from src.portfolio_backtester.strategies.base.trade_record import TradeRecord, TradeSide
from src.portfolio_backtester.strategies.base.base_strategy import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, strategy_params=None):
        super().__init__(strategy_params or {})
        self.generate_signals_calls = []
    
    def generate_signals(self, all_historical_data, benchmark_historical_data, 
                        non_universe_historical_data, current_date, 
                        start_date=None, end_date=None):
        """Mock generate_signals method."""
        self.generate_signals_calls.append({
            'current_date': current_date,
            'all_historical_data': all_historical_data,
        })
        
        # Return mock signals
        if current_date == pd.Timestamp("2023-01-15"):
            return pd.DataFrame({
                'AAPL': [0.5],
                'MSFT': [0.3],
                'GOOGL': [0.0]
            }, index=[current_date])
        elif current_date == pd.Timestamp("2023-01-16"):
            return pd.DataFrame({
                'AAPL': [0.3],  # Reduced from 0.5
                'MSFT': [0.3],  # No change
                'GOOGL': [0.2]  # Increased from 0.0
            }, index=[current_date])
        else:
            return pd.DataFrame()


class TestMetaStrategyTradeInterceptor:
    """Test cases for MetaStrategyTradeInterceptor class."""
    
    def test_interceptor_initialization(self):
        """Test basic interceptor initialization."""
        mock_strategy = MockStrategy()
        trade_callback = Mock()
        
        interceptor = MetaStrategyTradeInterceptor(
            sub_strategy=mock_strategy,
            strategy_id="test_strategy",
            allocated_capital=100000.0,
            trade_callback=trade_callback,
            transaction_cost_bps=10.0
        )
        
        assert interceptor.strategy_id == "test_strategy"
        assert interceptor.allocated_capital == 100000.0
        assert interceptor.transaction_cost_bps == 10.0
        assert interceptor.trade_callback == trade_callback
        assert interceptor._previous_signals is None
    
    def test_interceptor_wraps_generate_signals(self):
        """Test that interceptor properly wraps generate_signals method."""
        mock_strategy = MockStrategy()
        trade_callback = Mock()
        
        # Store original method
        original_method = mock_strategy.generate_signals
        
        interceptor = MetaStrategyTradeInterceptor(
            sub_strategy=mock_strategy,
            strategy_id="test_strategy",
            allocated_capital=100000.0,
            trade_callback=trade_callback
        )
        
        # Method should be wrapped
        assert mock_strategy.generate_signals != original_method
        assert hasattr(mock_strategy.generate_signals, '__wrapped__')
    
    def test_first_signal_generation_creates_trades(self):
        """Test that first signal generation creates trades for all non-zero signals."""
        mock_strategy = MockStrategy()
        trade_callback = Mock()
        
        interceptor = MetaStrategyTradeInterceptor(
            sub_strategy=mock_strategy,
            strategy_id="momentum",
            allocated_capital=100000.0,
            trade_callback=trade_callback
        )
        
        # Create mock historical data
        dates = pd.date_range("2023-01-01", "2023-01-20", freq="D")
        historical_data = pd.DataFrame({
            ('AAPL', 'Close'): [150.0] * len(dates),
            ('MSFT', 'Close'): [200.0] * len(dates),
            ('GOOGL', 'Close'): [2500.0] * len(dates)
        }, index=dates)
        historical_data.columns = pd.MultiIndex.from_tuples(
            historical_data.columns, names=['Ticker', 'Field']
        )
        
        # Call generate_signals (this should trigger trade detection)
        current_date = pd.Timestamp("2023-01-15")
        signals = mock_strategy.generate_signals(
            all_historical_data=historical_data,
            benchmark_historical_data=pd.DataFrame(),
            non_universe_historical_data=pd.DataFrame(),
            current_date=current_date
        )
        
        # Should have called trade_callback for AAPL and MSFT (non-zero signals)
        assert trade_callback.call_count == 2
        
        # Check the trades
        calls = trade_callback.call_args_list
        trades = [call[0][0] for call in calls]  # Extract TradeRecord from each call
        
        # Should have trades for AAPL and MSFT
        assets_traded = [trade.asset for trade in trades]
        assert 'AAPL' in assets_traded
        assert 'MSFT' in assets_traded
        assert 'GOOGL' not in assets_traded  # Zero signal
        
        # Check trade details for AAPL
        aapl_trade = next(trade for trade in trades if trade.asset == 'AAPL')
        assert aapl_trade.side == TradeSide.BUY
        assert aapl_trade.strategy_id == "momentum"
        assert aapl_trade.allocated_capital == 100000.0
        assert aapl_trade.price == 150.0
        # Trade value should be 0.5 * 100000 = 50000
        assert aapl_trade.trade_value == 50000.0
        # Quantity should be 50000 / 150 = 333.33
        assert abs(aapl_trade.quantity - 333.33) < 0.01
    
    def test_signal_changes_create_appropriate_trades(self):
        """Test that signal changes create appropriate buy/sell trades."""
        mock_strategy = MockStrategy()
        trade_callback = Mock()
        
        interceptor = MetaStrategyTradeInterceptor(
            sub_strategy=mock_strategy,
            strategy_id="momentum",
            allocated_capital=100000.0,
            trade_callback=trade_callback
        )
        
        # Create mock historical data
        dates = pd.date_range("2023-01-01", "2023-01-20", freq="D")
        historical_data = pd.DataFrame({
            ('AAPL', 'Close'): [150.0] * len(dates),
            ('MSFT', 'Close'): [200.0] * len(dates),
            ('GOOGL', 'Close'): [2500.0] * len(dates)
        }, index=dates)
        historical_data.columns = pd.MultiIndex.from_tuples(
            historical_data.columns, names=['Ticker', 'Field']
        )
        
        # First call - establishes initial positions
        current_date1 = pd.Timestamp("2023-01-15")
        mock_strategy.generate_signals(
            all_historical_data=historical_data,
            benchmark_historical_data=pd.DataFrame(),
            non_universe_historical_data=pd.DataFrame(),
            current_date=current_date1
        )
        
        # Reset mock for second call
        trade_callback.reset_mock()
        
        # Second call - changes in signals
        current_date2 = pd.Timestamp("2023-01-16")
        mock_strategy.generate_signals(
            all_historical_data=historical_data,
            benchmark_historical_data=pd.DataFrame(),
            non_universe_historical_data=pd.DataFrame(),
            current_date=current_date2
        )
        
        # Should have trades for changes:
        # AAPL: 0.5 -> 0.3 (sell 0.2)
        # MSFT: 0.3 -> 0.3 (no change, no trade)
        # GOOGL: 0.0 -> 0.2 (buy 0.2)
        assert trade_callback.call_count == 2
        
        calls = trade_callback.call_args_list
        trades = [call[0][0] for call in calls]
        
        # Check AAPL sell trade
        aapl_trade = next(trade for trade in trades if trade.asset == 'AAPL')
        assert aapl_trade.side == TradeSide.SELL
        assert aapl_trade.trade_value == 20000.0  # 0.2 * 100000
        assert aapl_trade.metadata['signal_change'] == -0.2
        
        # Check GOOGL buy trade
        googl_trade = next(trade for trade in trades if trade.asset == 'GOOGL')
        assert googl_trade.side == TradeSide.BUY
        assert googl_trade.trade_value == 20000.0  # 0.2 * 100000
        assert googl_trade.metadata['signal_change'] == 0.2
    
    def test_update_allocated_capital(self):
        """Test updating allocated capital."""
        mock_strategy = MockStrategy()
        trade_callback = Mock()
        
        interceptor = MetaStrategyTradeInterceptor(
            sub_strategy=mock_strategy,
            strategy_id="test_strategy",
            allocated_capital=100000.0,
            trade_callback=trade_callback
        )
        
        assert interceptor.allocated_capital == 100000.0
        
        interceptor.update_allocated_capital(150000.0)
        assert interceptor.allocated_capital == 150000.0
    
    def test_get_strategy_info(self):
        """Test getting strategy information."""
        mock_strategy = MockStrategy()
        trade_callback = Mock()
        
        interceptor = MetaStrategyTradeInterceptor(
            sub_strategy=mock_strategy,
            strategy_id="test_strategy",
            allocated_capital=100000.0,
            trade_callback=trade_callback
        )
        
        info = interceptor.get_strategy_info()
        
        assert info['strategy_id'] == "test_strategy"
        assert info['strategy_class'] == "MockStrategy"
        assert info['allocated_capital'] == 100000.0
        assert info['has_previous_signals'] == False
        assert info['previous_signal_count'] == 0
    
    def test_reset_state(self):
        """Test resetting interceptor state."""
        mock_strategy = MockStrategy()
        trade_callback = Mock()
        
        interceptor = MetaStrategyTradeInterceptor(
            sub_strategy=mock_strategy,
            strategy_id="test_strategy",
            allocated_capital=100000.0,
            trade_callback=trade_callback
        )
        
        # Set some previous signals
        interceptor._previous_signals = pd.Series([0.5, 0.3], index=['AAPL', 'MSFT'])
        
        assert interceptor._previous_signals is not None
        
        interceptor.reset_state()
        assert interceptor._previous_signals is None
    
    def test_attribute_delegation(self):
        """Test that interceptor delegates attributes to wrapped strategy."""
        mock_strategy = MockStrategy()
        mock_strategy.custom_attribute = "test_value"
        trade_callback = Mock()
        
        interceptor = MetaStrategyTradeInterceptor(
            sub_strategy=mock_strategy,
            strategy_id="test_strategy",
            allocated_capital=100000.0,
            trade_callback=trade_callback
        )
        
        # Should delegate to wrapped strategy
        assert interceptor.custom_attribute == "test_value"
        assert interceptor.strategy_params == mock_strategy.strategy_params
    
    def test_no_price_available_skips_trade(self):
        """Test that trades are skipped when price is not available."""
        mock_strategy = MockStrategy()
        trade_callback = Mock()
        
        interceptor = MetaStrategyTradeInterceptor(
            sub_strategy=mock_strategy,
            strategy_id="momentum",
            allocated_capital=100000.0,
            trade_callback=trade_callback
        )
        
        # Create historical data without AAPL price
        dates = pd.date_range("2023-01-01", "2023-01-20", freq="D")
        historical_data = pd.DataFrame({
            ('MSFT', 'Close'): [200.0] * len(dates),
            ('GOOGL', 'Close'): [2500.0] * len(dates)
        }, index=dates)
        historical_data.columns = pd.MultiIndex.from_tuples(
            historical_data.columns, names=['Ticker', 'Field']
        )
        
        # Call generate_signals
        current_date = pd.Timestamp("2023-01-15")
        mock_strategy.generate_signals(
            all_historical_data=historical_data,
            benchmark_historical_data=pd.DataFrame(),
            non_universe_historical_data=pd.DataFrame(),
            current_date=current_date
        )
        
        # Should only have trade for MSFT (AAPL skipped due to no price)
        assert trade_callback.call_count == 1
        trade = trade_callback.call_args[0][0]
        assert trade.asset == 'MSFT'
    
    