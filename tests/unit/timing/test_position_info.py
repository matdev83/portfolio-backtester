"""
Tests for position information management.
Split from test_advanced_state_management.py for better organization.
"""

import pytest
import pandas as pd
import numpy as np
from src.portfolio_backtester.timing.timing_state import PositionInfo


class TestPositionInfo:
    """Test position information tracking."""
    
    def test_position_creation(self):
        """Test position creation with basic information."""
        entry_date = pd.Timestamp('2023-01-01')
        entry_price = 100.0
        entry_weight = 0.25
        current_weight = 0.25
        
        position = PositionInfo(
            entry_date=entry_date,
            entry_price=entry_price,
            entry_weight=entry_weight,
            current_weight=current_weight
        )
        
        assert position.entry_date == entry_date
        assert position.entry_price == entry_price
        assert position.entry_weight == entry_weight
        assert position.current_weight == current_weight
    
    def test_position_update(self):
        """Test position weight updates."""
        position = PositionInfo(
            entry_date=pd.Timestamp('2023-01-01'),
            entry_price=100.0,
            entry_weight=0.25,
            current_weight=0.25
        )
        
        # Update weight
        position.update_weight(0.30, 105.0)
        assert position.current_weight == 0.30
        assert position.max_weight == 0.30
        # min_weight is initialized to the first weight set, not the entry_weight
        assert position.min_weight == 0.30
        
        # Check total return calculation
        expected_return = (105.0 - 100.0) / 100.0
        assert abs(position.total_return - expected_return) < 1e-6
    
    def test_position_metrics(self):
        """Test position performance metrics."""
        position = PositionInfo(
            entry_date=pd.Timestamp('2023-01-01'),
            entry_price=100.0,
            entry_weight=0.25,
            current_weight=0.25
        )
        
        # Update with new price
        position.update_weight(0.25, 120.0)
        
        # Test return calculation
        expected_return = (120.0 - 100.0) / 100.0  # 20% return
        assert abs(position.total_return - expected_return) < 1e-6
        
        # Test unrealized PnL
        expected_pnl = 0.25 * expected_return
        assert abs(position.unrealized_pnl - expected_pnl) < 1e-6
    
    def test_consecutive_periods_tracking(self):
        """Test consecutive periods tracking."""
        position = PositionInfo(
            entry_date=pd.Timestamp('2023-01-01'),
            entry_price=100.0,
            entry_weight=0.25,
            current_weight=0.25,
            consecutive_periods=1
        )
        
        assert position.consecutive_periods == 1
        
        # Update weight (simulating another period)
        position.update_weight(0.30, 105.0)
        # Note: consecutive_periods would be updated by the TimingState, not PositionInfo itself


if __name__ == '__main__':
    pytest.main([__file__, '-v'])