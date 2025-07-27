"""
Tests for TimingState dataclass.
"""

import pytest
import pandas as pd
from src.portfolio_backtester.timing.timing_state import TimingState


class TestTimingState:
    """Test cases for TimingState."""
    
    def test_initialization(self):
        """Test TimingState initialization."""
        state = TimingState()
        
        assert state.last_signal_date is None
        assert state.last_weights is None
        assert len(state.position_entry_dates) == 0
        assert len(state.position_entry_prices) == 0
        assert len(state.scheduled_dates) == 0
        assert len(state.consecutive_periods) == 0
    
    def test_reset(self):
        """Test state reset functionality."""
        state = TimingState()
        
        # Set some state
        state.last_signal_date = pd.Timestamp('2023-01-01')
        state.last_weights = pd.Series([0.5, 0.5], index=['A', 'B'])
        state.position_entry_dates['A'] = pd.Timestamp('2023-01-01')
        state.position_entry_prices['A'] = 100.0
        state.scheduled_dates.add(pd.Timestamp('2023-01-01'))
        state.consecutive_periods['test'] = 5
        
        # Reset
        state.reset()
        
        # Verify reset
        assert state.last_signal_date is None
        assert state.last_weights is None
        assert len(state.position_entry_dates) == 0
        assert len(state.position_entry_prices) == 0
        assert len(state.scheduled_dates) == 0
        assert len(state.consecutive_periods) == 0
    
    def test_update_signal(self):
        """Test signal update functionality."""
        state = TimingState()
        date = pd.Timestamp('2023-01-01')
        weights = pd.Series([0.6, 0.4], index=['A', 'B'])
        
        state.update_signal(date, weights)
        
        assert state.last_signal_date == date
        assert state.last_weights.equals(weights)
        assert state.last_weights is not weights  # Should be a copy
    
    def test_update_signal_with_none(self):
        """Test signal update with None weights."""
        state = TimingState()
        date = pd.Timestamp('2023-01-01')
        
        state.update_signal(date, None)
        
        assert state.last_signal_date == date
        assert state.last_weights is None
    
    def test_update_positions_new_positions(self):
        """Test position updates for new positions."""
        state = TimingState()
        date = pd.Timestamp('2023-01-01')
        new_weights = pd.Series([0.5, 0.5], index=['A', 'B'])
        prices = pd.Series([100.0, 200.0], index=['A', 'B'])
        
        state.update_positions(date, new_weights, prices)
        
        # Both positions should be tracked as new
        assert state.position_entry_dates['A'] == date
        assert state.position_entry_dates['B'] == date
        assert state.position_entry_prices['A'] == 100.0
        assert state.position_entry_prices['B'] == 200.0
    
    def test_update_positions_close_positions(self):
        """Test position updates for closing positions."""
        state = TimingState()
        date1 = pd.Timestamp('2023-01-01')
        date2 = pd.Timestamp('2023-01-02')
        
        # Set up initial positions
        state.last_weights = pd.Series([0.5, 0.5], index=['A', 'B'])
        state.position_entry_dates['A'] = date1
        state.position_entry_dates['B'] = date1
        state.position_entry_prices['A'] = 100.0
        state.position_entry_prices['B'] = 200.0
        
        # Close position A, keep position B
        new_weights = pd.Series([0.0, 1.0], index=['A', 'B'])
        prices = pd.Series([105.0, 210.0], index=['A', 'B'])
        
        state.update_positions(date2, new_weights, prices)
        
        # Position A should be closed
        assert 'A' not in state.position_entry_dates
        assert 'A' not in state.position_entry_prices
        
        # Position B should remain
        assert state.position_entry_dates['B'] == date1
        assert state.position_entry_prices['B'] == 200.0
    
    def test_update_positions_mixed_changes(self):
        """Test position updates with mixed changes."""
        state = TimingState()
        date1 = pd.Timestamp('2023-01-01')
        date2 = pd.Timestamp('2023-01-02')
        
        # Set up initial positions
        state.last_weights = pd.Series([0.5, 0.5, 0.0], index=['A', 'B', 'C'])
        state.position_entry_dates['A'] = date1
        state.position_entry_dates['B'] = date1
        state.position_entry_prices['A'] = 100.0
        state.position_entry_prices['B'] = 200.0
        
        # Close A, keep B, add C
        new_weights = pd.Series([0.0, 0.3, 0.7], index=['A', 'B', 'C'])
        prices = pd.Series([105.0, 210.0, 50.0], index=['A', 'B', 'C'])
        
        state.update_positions(date2, new_weights, prices)
        
        # Position A should be closed
        assert 'A' not in state.position_entry_dates
        assert 'A' not in state.position_entry_prices
        
        # Position B should remain unchanged
        assert state.position_entry_dates['B'] == date1
        assert state.position_entry_prices['B'] == 200.0
        
        # Position C should be new
        assert state.position_entry_dates['C'] == date2
        assert state.position_entry_prices['C'] == 50.0
    
    def test_get_position_holding_days(self):
        """Test position holding days calculation."""
        state = TimingState()
        entry_date = pd.Timestamp('2023-01-01')
        current_date = pd.Timestamp('2023-01-11')
        
        state.position_entry_dates['A'] = entry_date
        
        holding_days = state.get_position_holding_days('A', current_date)
        assert holding_days == 10
        
        # Test non-existent position
        holding_days = state.get_position_holding_days('B', current_date)
        assert holding_days is None
    
    def test_is_position_held(self):
        """Test position held check."""
        state = TimingState()
        
        assert not state.is_position_held('A')
        
        state.position_entry_dates['A'] = pd.Timestamp('2023-01-01')
        assert state.is_position_held('A')
        assert not state.is_position_held('B')
    
    def test_get_held_assets(self):
        """Test getting held assets."""
        state = TimingState()
        
        assert state.get_held_assets() == set()
        
        state.position_entry_dates['A'] = pd.Timestamp('2023-01-01')
        state.position_entry_dates['B'] = pd.Timestamp('2023-01-02')
        
        held_assets = state.get_held_assets()
        assert held_assets == {'A', 'B'}
    
    def test_update_positions_with_missing_prices(self):
        """Test position updates when prices are missing."""
        state = TimingState()
        date = pd.Timestamp('2023-01-01')
        new_weights = pd.Series([0.5, 0.5], index=['A', 'B'])
        prices = pd.Series([100.0, float('nan')], index=['A', 'B'])
        
        state.update_positions(date, new_weights, prices)
        
        # Position A should have price recorded
        assert state.position_entry_dates['A'] == date
        assert state.position_entry_prices['A'] == 100.0
        
        # Position B should have entry date but no price
        assert state.position_entry_dates['B'] == date
        assert 'B' not in state.position_entry_prices