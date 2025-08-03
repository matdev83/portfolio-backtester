"""
Unit tests for PositionTracker class.

Tests the position tracking system for daily strategy evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from src.portfolio_backtester.backtesting.position_tracker import PositionTracker, Position, Trade


class TestPositionTracker:
    """Test cases for PositionTracker class."""
    
    def test_position_tracker_initialization(self):
        """Test basic position tracker initialization."""
        tracker = PositionTracker()
        
        assert len(tracker.current_positions) == 0
        assert len(tracker.completed_trades) == 0
        assert len(tracker.daily_weights) == 0
    
    def test_position_opening(self):
        """Test opening new positions."""
        tracker = PositionTracker()
        
        # Create test signals - open position
        dates = pd.DatetimeIndex(['2025-01-01'])
        signals = pd.DataFrame({'TLT': [1.0]}, index=dates)
        
        # Update positions
        weights = tracker.update_positions(signals, pd.Timestamp('2025-01-01'))
        
        # Check position was opened
        assert len(tracker.current_positions) == 1
        assert 'TLT' in tracker.current_positions
        assert tracker.current_positions['TLT'].weight == 1.0
        assert tracker.current_positions['TLT'].entry_date == pd.Timestamp('2025-01-01')
        
        # Check returned weights
        assert weights['TLT'] == 1.0
    
    def test_position_closing(self):
        """Test closing existing positions."""
        tracker = PositionTracker()
        
        # Open position first
        dates = pd.DatetimeIndex(['2025-01-01', '2025-01-10'])
        signals_open = pd.DataFrame({'TLT': [1.0]}, index=[dates[0]])
        tracker.update_positions(signals_open, dates[0])
        
        # Close position
        signals_close = pd.DataFrame({'TLT': [0.0]}, index=[dates[1]])
        weights = tracker.update_positions(signals_close, dates[1])
        
        # Check position was closed
        assert len(tracker.current_positions) == 0
        assert len(tracker.completed_trades) == 1
        
        # Check trade record
        trade = tracker.completed_trades[0]
        assert trade.ticker == 'TLT'
        assert trade.entry_date == dates[0]
        assert trade.exit_date == dates[1]
        assert trade.entry_weight == 1.0
        assert trade.exit_weight == 0.0
        
        # Check returned weights
        assert weights['TLT'] == 0.0
    
    def test_trade_duration_calculation(self):
        """Test that trade duration is calculated correctly in business days."""
        tracker = PositionTracker()
        
        # Open position on Jan 1 (Wednesday), close on Jan 10 (Friday)
        entry_date = pd.Timestamp('2025-01-01')  # Wednesday
        exit_date = pd.Timestamp('2025-01-10')   # Friday
        
        # Open position
        signals_open = pd.DataFrame({'TLT': [1.0]}, index=[entry_date])
        tracker.update_positions(signals_open, entry_date)
        
        # Close position
        signals_close = pd.DataFrame({'TLT': [0.0]}, index=[exit_date])
        tracker.update_positions(signals_close, exit_date)
        
        # Check trade duration
        trade = tracker.completed_trades[0]
        expected_duration = len(pd.bdate_range(entry_date, exit_date)) - 1
        assert trade.duration_days == expected_duration
        assert trade.duration_days == 7  # 7 business days between Jan 1 and Jan 10
    
    def test_position_adjustment(self):
        """Test adjusting existing position weights."""
        tracker = PositionTracker()
        
        # Open position with weight 1.0
        dates = pd.DatetimeIndex(['2025-01-01', '2025-01-02'])
        signals_open = pd.DataFrame({'TLT': [1.0]}, index=[dates[0]])
        tracker.update_positions(signals_open, dates[0])
        
        # Adjust position to weight 0.5
        signals_adjust = pd.DataFrame({'TLT': [0.5]}, index=[dates[1]])
        weights = tracker.update_positions(signals_adjust, dates[1])
        
        # Check position was adjusted
        assert len(tracker.current_positions) == 1
        assert tracker.current_positions['TLT'].weight == 0.5
        assert len(tracker.completed_trades) == 0  # No trade completed
        
        # Check returned weights
        assert weights['TLT'] == 0.5
    
    def test_multiple_positions(self):
        """Test handling multiple positions simultaneously."""
        tracker = PositionTracker()
        
        # Open multiple positions
        date = pd.Timestamp('2025-01-01')
        signals = pd.DataFrame({
            'TLT': [1.0],
            'SPY': [-0.5],
            'GLD': [0.0]
        }, index=[date])
        
        weights = tracker.update_positions(signals, date)
        
        # Check positions
        assert len(tracker.current_positions) == 2  # TLT and SPY (GLD has 0 weight)
        assert tracker.current_positions['TLT'].weight == 1.0
        assert tracker.current_positions['SPY'].weight == -0.5
        assert 'GLD' not in tracker.current_positions
        
        # Check returned weights
        assert weights['TLT'] == 1.0
        assert weights['SPY'] == -0.5
        assert weights['GLD'] == 0.0
    
    def test_price_based_pnl_calculation(self):
        """Test P&L calculation when price data is available."""
        tracker = PositionTracker()
        
        # Create price data
        dates = pd.DatetimeIndex(['2025-01-01', '2025-01-10'])
        prices = pd.DataFrame({
            'TLT': [100.0, 105.0]  # 5% price increase
        }, index=dates)
        
        # Open position
        signals_open = pd.DataFrame({'TLT': [1.0]}, index=[dates[0]])
        tracker.update_positions(signals_open, dates[0], prices.loc[[dates[0]]])
        
        # Close position
        signals_close = pd.DataFrame({'TLT': [0.0]}, index=[dates[1]])
        tracker.update_positions(signals_close, dates[1], prices.loc[[dates[1]]])
        
        # Check P&L calculation
        trade = tracker.completed_trades[0]
        expected_pnl = (105.0 - 100.0) / 100.0 * 1.0  # 5% return * 1.0 weight
        assert abs(trade.pnl - expected_pnl) < 1e-6
    
    def test_multiindex_price_data(self):
        """Test handling of MultiIndex price data."""
        tracker = PositionTracker()
        
        # Create MultiIndex price data
        dates = pd.DatetimeIndex(['2025-01-01', '2025-01-10'])
        columns = pd.MultiIndex.from_product([['TLT'], ['Close']], names=['Ticker', 'Field'])
        prices = pd.DataFrame({
            ('TLT', 'Close'): [100.0, 105.0]
        }, index=dates, columns=columns)
        
        # Open position
        signals_open = pd.DataFrame({'TLT': [1.0]}, index=[dates[0]])
        tracker.update_positions(signals_open, dates[0], prices.loc[[dates[0]]])
        
        # Close position
        signals_close = pd.DataFrame({'TLT': [0.0]}, index=[dates[1]])
        tracker.update_positions(signals_close, dates[1], prices.loc[[dates[1]]])
        
        # Check P&L calculation
        trade = tracker.completed_trades[0]
        expected_pnl = (105.0 - 100.0) / 100.0 * 1.0  # 5% return * 1.0 weight
        assert abs(trade.pnl - expected_pnl) < 1e-6
    
    def test_get_daily_weights_df(self):
        """Test getting daily weights as DataFrame."""
        tracker = PositionTracker()
        
        # Create multiple days of signals
        dates = pd.DatetimeIndex(['2025-01-01', '2025-01-02', '2025-01-03'])
        
        # Day 1: Open TLT position
        signals1 = pd.DataFrame({'TLT': [1.0], 'SPY': [0.0]}, index=[dates[0]])
        tracker.update_positions(signals1, dates[0])
        
        # Day 2: Open SPY position, keep TLT
        signals2 = pd.DataFrame({'TLT': [1.0], 'SPY': [0.5]}, index=[dates[1]])
        tracker.update_positions(signals2, dates[1])
        
        # Day 3: Close TLT, keep SPY
        signals3 = pd.DataFrame({'TLT': [0.0], 'SPY': [0.5]}, index=[dates[2]])
        tracker.update_positions(signals3, dates[2])
        
        # Get daily weights DataFrame
        weights_df = tracker.get_daily_weights_df()
        
        # Check structure
        assert len(weights_df) == 3
        assert list(weights_df.columns) == ['SPY', 'TLT']  # Sorted alphabetically
        
        # Check values
        assert weights_df.loc[dates[0], 'TLT'] == 1.0
        assert weights_df.loc[dates[0], 'SPY'] == 0.0
        assert weights_df.loc[dates[1], 'TLT'] == 1.0
        assert weights_df.loc[dates[1], 'SPY'] == 0.5
        assert weights_df.loc[dates[2], 'TLT'] == 0.0
        assert weights_df.loc[dates[2], 'SPY'] == 0.5
    
    def test_get_trade_summary(self):
        """Test getting trade summary statistics."""
        tracker = PositionTracker()
        
        # Create multiple trades with different durations
        dates = pd.DatetimeIndex(['2025-01-01', '2025-01-05', '2025-01-10', '2025-01-15'])
        
        # Trade 1: 4 business days (Jan 1 to Jan 5)
        signals1 = pd.DataFrame({'TLT': [1.0]}, index=[dates[0]])
        tracker.update_positions(signals1, dates[0])
        signals1_close = pd.DataFrame({'TLT': [0.0]}, index=[dates[1]])
        tracker.update_positions(signals1_close, dates[1])
        
        # Trade 2: 5 business days (Jan 10 to Jan 15)
        signals2 = pd.DataFrame({'SPY': [1.0]}, index=[dates[2]])
        tracker.update_positions(signals2, dates[2])
        signals2_close = pd.DataFrame({'SPY': [0.0]}, index=[dates[3]])
        tracker.update_positions(signals2_close, dates[3])
        
        # Get trade summary
        summary = tracker.get_trade_summary()
        
        # Check summary statistics
        assert summary['total_trades'] == 2
        assert summary['min_duration'] == 2  # Minimum duration
        assert summary['max_duration'] == 3  # Maximum duration
        assert abs(summary['avg_duration'] - 2.5) < 1e-6  # Average duration
    
    def test_empty_signals_handling(self):
        """Test handling of empty signals."""
        tracker = PositionTracker()
        
        # Empty signals DataFrame
        empty_signals = pd.DataFrame()
        date = pd.Timestamp('2025-01-01')
        
        # Should return empty series without error
        weights = tracker.update_positions(empty_signals, date)
        assert len(weights) == 0
        assert isinstance(weights, pd.Series)
    
    def test_missing_date_in_signals(self):
        """Test handling when current date is not in signals index."""
        tracker = PositionTracker()
        
        # Signals for different date
        signals = pd.DataFrame({'TLT': [1.0]}, index=[pd.Timestamp('2025-01-01')])
        current_date = pd.Timestamp('2025-01-02')
        
        # Should return empty series without error
        weights = tracker.update_positions(signals, current_date)
        assert len(weights) == 0
        assert isinstance(weights, pd.Series)
    
    def test_reset_functionality(self):
        """Test resetting the tracker state."""
        tracker = PositionTracker()
        
        # Add some positions and trades
        date = pd.Timestamp('2025-01-01')
        signals = pd.DataFrame({'TLT': [1.0]}, index=[date])
        tracker.update_positions(signals, date)
        
        # Reset
        tracker.reset()
        
        # Check everything is cleared
        assert len(tracker.current_positions) == 0
        assert len(tracker.completed_trades) == 0
        assert len(tracker.daily_weights) == 0