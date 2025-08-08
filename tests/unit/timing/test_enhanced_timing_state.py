"""
Tests for enhanced timing state management features.
Split from test_advanced_state_management.py for better organization.
"""

import pytest
import pandas as pd
from portfolio_backtester.interfaces.timing_state_interface import create_timing_state


class TestEnhancedTimingState:
    """Test enhanced timing state management features."""
    
    def setup_method(self):
        """Set up test data."""
        self.state = create_timing_state()
        self.test_date = pd.Timestamp('2023-01-01')
        self.assets = ['AAPL', 'MSFT', 'GOOGL']
        self.weights = pd.Series([0.4, 0.3, 0.3], index=self.assets)
        self.prices = pd.Series([150.0, 250.0, 2500.0], index=self.assets)
    
    def test_enhanced_position_tracking(self):
        """Test enhanced position tracking with PositionInfo."""
        # Initial position entry
        self.state.update_positions(self.test_date, self.weights, self.prices)
        
        # Verify enhanced tracking
        assert len(self.state.positions) == 3
        assert 'AAPL' in self.state.positions
        
        aapl_position = self.state.positions['AAPL']
        assert aapl_position.entry_date == self.test_date
        assert aapl_position.entry_price == 150.0
        assert aapl_position.entry_weight == 0.4
        assert aapl_position.current_weight == 0.4
        assert aapl_position.consecutive_periods == 1
        
        # Verify legacy compatibility
        assert len(self.state.position_entry_dates) == 3
        assert self.state.position_entry_dates['AAPL'] == self.test_date
        assert self.state.position_entry_prices['AAPL'] == 150.0
        assert self.state.consecutive_periods['AAPL'] == 1
    
    def test_position_weight_updates(self):
        """Test position weight updates and tracking."""
        # Initial positions
        self.state.update_positions(self.test_date, self.weights, self.prices)
        
        # Update weights
        next_date = self.test_date + pd.Timedelta(days=1)
        new_weights = pd.Series([0.5, 0.3, 0.2], index=self.assets)
        new_prices = pd.Series([155.0, 245.0, 2600.0], index=self.assets)
        
        self.state.update_positions(next_date, new_weights, new_prices)
        
        # Check AAPL position updates
        aapl_position = self.state.positions['AAPL']
        assert aapl_position.current_weight == 0.5
        assert aapl_position.max_weight == 0.5
        assert aapl_position.min_weight == 0.4
        assert aapl_position.consecutive_periods == 2
        assert abs(aapl_position.total_return - (155.0 - 150.0) / 150.0) < 1e-10
        
        # Check GOOGL position (weight decreased)
        googl_position = self.state.positions['GOOGL']
        assert googl_position.current_weight == 0.2
        assert googl_position.max_weight == 0.3
        assert googl_position.min_weight == 0.2
    
    def test_position_exit_and_history(self):
        """Test position exit and history tracking."""
        # Initial positions
        self.state.update_positions(self.test_date, self.weights, self.prices)
        
        # Exit one position
        exit_date = self.test_date + pd.Timedelta(days=5)
        exit_weights = pd.Series([0.5, 0.5, 0.0], index=self.assets)  # Exit GOOGL
        exit_prices = pd.Series([160.0, 260.0, 2700.0], index=self.assets)
        
        self.state.update_positions(exit_date, exit_weights, exit_prices)
        
        # Verify position was removed from active positions
        assert 'GOOGL' not in self.state.positions
        assert len(self.state.positions) == 2
        
        # Verify position history was recorded
        assert len(self.state.position_history) == 1
        googl_history = self.state.position_history[0]
        
        assert googl_history['asset'] == 'GOOGL'
        assert googl_history['entry_date'] == self.test_date
        assert googl_history['exit_date'] == exit_date
        assert googl_history['holding_days'] == 5
        assert googl_history['entry_price'] == 2500.0
        assert googl_history['exit_price'] == 2700.0
        assert googl_history['entry_weight'] == 0.3
        
        # Verify legacy compatibility
        assert 'GOOGL' not in self.state.position_entry_dates
        assert 'GOOGL' not in self.state.consecutive_periods
    
    def test_consecutive_periods_tracking(self):
        """Test consecutive periods tracking."""
        dates = [self.test_date + pd.Timedelta(days=i) for i in range(5)]
        
        for i, date in enumerate(dates):
            self.state.update_positions(date, self.weights, self.prices)
            
            # Check consecutive periods
            for asset in self.assets:
                assert self.state.get_consecutive_periods(asset) == i + 1
                assert self.state.positions[asset].consecutive_periods == i + 1
    
    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        self.state.update_positions(self.test_date, self.weights, self.prices)
        
        summary = self.state.get_portfolio_summary()
        
        assert summary['total_positions'] == 3
        assert abs(summary['total_weight'] - 1.0) < 1e-10
        assert summary['avg_holding_days'] == 0  # Same day
        assert summary['last_updated'] == self.test_date
        assert summary['total_historical_positions'] == 0
        assert set(summary['assets']) == set(self.assets)
    
    def test_position_statistics(self):
        """Test position statistics calculation."""
        # No history initially
        stats = self.state.get_position_statistics()
        assert stats['total_trades'] == 0
        
        # Add some position history manually for testing
        self.state.add_test_position_history([
            {
                'asset': 'AAPL',
                'holding_days': 5,
                'total_return': 0.1,
                'entry_price': 100.0,
                'exit_price': 110.0
            },
            {
                'asset': 'MSFT',
                'holding_days': 10,
                'total_return': -0.05,
                'entry_price': 200.0,
                'exit_price': 190.0
            },
            {
                'asset': 'GOOGL',
                'holding_days': 3,
                'total_return': 0.15,
                'entry_price': 2000.0,
                'exit_price': 2300.0
            }
        ])
        
        stats = self.state.get_position_statistics()
        
        assert stats['total_trades'] == 3
        assert stats['avg_holding_days'] == (5 + 10 + 3) / 3
        assert stats['min_holding_days'] == 3
        assert stats['max_holding_days'] == 10
        assert stats['positive_trades'] == 2
        assert stats['negative_trades'] == 1
        assert abs(stats['win_rate'] - 2/3) < 1e-10
        assert abs(stats['avg_return'] - (0.1 - 0.05 + 0.15) / 3) < 1e-10
    
    def test_position_return_calculation(self):
        """Test position return calculation."""
        self.state.update_positions(self.test_date, self.weights, self.prices)
        
        # Test with current price
        current_return = self.state.get_position_return('AAPL', 165.0)
        expected_return = (165.0 - 150.0) / 150.0
        assert abs(current_return - expected_return) < 1e-10
        
        # Test without current price (should use stored return)
        stored_return = self.state.get_position_return('AAPL')
        assert stored_return == 0.0  # No price update yet
        
        # Test non-existent position
        assert self.state.get_position_return('TSLA') is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])