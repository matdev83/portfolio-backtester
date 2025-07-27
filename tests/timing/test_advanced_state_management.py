"""
Tests for advanced state management features (Task 9).
Tests enhanced position tracking, state persistence, and debugging utilities.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.portfolio_backtester.timing.timing_state import TimingState, PositionInfo


class TestPositionInfo:
    """Test the PositionInfo dataclass."""
    
    def test_position_info_initialization(self):
        """Test PositionInfo initialization."""
        entry_date = pd.Timestamp('2023-01-01')
        position = PositionInfo(
            entry_date=entry_date,
            entry_price=100.0,
            entry_weight=0.1,
            current_weight=0.1
        )
        
        assert position.entry_date == entry_date
        assert position.entry_price == 100.0
        assert position.entry_weight == 0.1
        assert position.current_weight == 0.1
        assert position.consecutive_periods == 0
        assert position.max_weight == 0.0
        assert position.min_weight == 0.0
        assert position.total_return == 0.0
        assert position.unrealized_pnl == 0.0
    
    def test_position_info_update_weight(self):
        """Test position weight updates."""
        position = PositionInfo(
            entry_date=pd.Timestamp('2023-01-01'),
            entry_price=100.0,
            entry_weight=0.1,
            current_weight=0.1
        )
        
        # First update
        position.update_weight(0.15, 110.0)
        assert position.current_weight == 0.15
        assert position.max_weight == 0.15
        assert position.min_weight == 0.15
        assert position.total_return == 0.1  # (110-100)/100
        assert position.unrealized_pnl == 0.15 * 0.1
        
        # Second update with lower weight
        position.update_weight(0.08, 105.0)
        assert position.current_weight == 0.08
        assert position.max_weight == 0.15  # Unchanged
        assert position.min_weight == 0.08  # Updated
        assert position.total_return == 0.05  # (105-100)/100
        assert position.unrealized_pnl == 0.08 * 0.05


class TestEnhancedTimingState:
    """Test enhanced timing state management features."""
    
    def setup_method(self):
        """Set up test data."""
        self.state = TimingState()
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
        self.state.position_history = [
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
        ]
        
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


class TestStateDebugUtilities:
    """Test debugging utilities."""
    
    def setup_method(self):
        """Set up test data."""
        self.state = TimingState()
        self.state.enable_debug(True)
    
    def test_debug_logging(self):
        """Test debug logging functionality."""
        assert self.state.debug_enabled
        assert len(self.state.debug_log) == 1  # Debug enabled message
        
        # Test manual debug logging
        self.state._log_debug("Test message", {"key": "value"})
        
        log_entries = self.state.get_debug_log()
        assert len(log_entries) == 2
        
        test_entry = log_entries[-1]
        assert test_entry['message'] == "Test message"
        assert test_entry['data']['key'] == "value"
        assert 'timestamp' in test_entry
    
    def test_debug_log_size_management(self):
        """Test debug log size management."""
        # Add many entries to test size management
        for i in range(1200):
            self.state._log_debug(f"Message {i}", {"index": i})
        
        # Should be limited to around 500 entries (with some buffer for the initial debug message)
        assert len(self.state.debug_log) <= 700  # Allow some buffer
        assert len(self.state.debug_log) >= 500  # Should have at least 500
        
        # Should contain the most recent entries
        last_entry = self.state.debug_log[-1]
        assert "Message 1199" in last_entry['message']
    
    def test_debug_log_operations(self):
        """Test debug log operations."""
        # Add some entries
        for i in range(10):
            self.state._log_debug(f"Message {i}", {"index": i})
        
        # Test getting last N entries
        last_3 = self.state.get_debug_log(last_n=3)
        assert len(last_3) == 3
        assert "Message 9" in last_3[-1]['message']
        
        # Test clearing log
        self.state.clear_debug_log()
        assert len(self.state.debug_log) == 1  # Only the "cleared" message
    
    def test_state_summary_printing(self):
        """Test state summary printing."""
        # Add some test data
        test_date = pd.Timestamp('2023-01-01')
        weights = pd.Series([0.5, 0.5], index=['AAPL', 'MSFT'])
        prices = pd.Series([150.0, 250.0], index=['AAPL', 'MSFT'])
        
        self.state.update_positions(test_date, weights, prices)
        
        # This should not raise an exception
        self.state.print_state_summary()


class TestStatePersistence:
    """Test state persistence functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.state = TimingState()
        self.test_date = pd.Timestamp('2023-01-01')
        
        # Add some test data
        weights = pd.Series([0.4, 0.3, 0.3], index=['AAPL', 'MSFT', 'GOOGL'])
        prices = pd.Series([150.0, 250.0, 2500.0], index=['AAPL', 'MSFT', 'GOOGL'])
        
        self.state.update_signal(self.test_date, weights)
        self.state.update_positions(self.test_date, weights, prices)
        self.state.enable_debug(True)
        
        # Add some history
        self.state.position_history.append({
            'asset': 'TSLA',
            'entry_date': self.test_date - pd.Timedelta(days=5),
            'exit_date': self.test_date - pd.Timedelta(days=1),
            'holding_days': 4,
            'total_return': 0.1
        })
    
    def test_state_serialization(self):
        """Test state serialization to dictionary."""
        state_dict = self.state.to_dict()
        
        assert state_dict['state_version'] == "1.0"
        assert state_dict['last_signal_date'] == self.test_date.isoformat()
        assert state_dict['debug_enabled'] == True
        assert len(state_dict['positions']) == 3
        assert len(state_dict['position_history']) == 1
        
        # Check position data
        aapl_data = state_dict['positions']['AAPL']
        assert aapl_data['entry_date'] == self.test_date.isoformat()
        assert aapl_data['entry_price'] == 150.0
        assert aapl_data['entry_weight'] == 0.4
    
    def test_state_deserialization(self):
        """Test state deserialization from dictionary."""
        # Serialize first
        state_dict = self.state.to_dict()
        
        # Deserialize
        restored_state = TimingState.from_dict(state_dict)
        
        # Verify restoration
        assert restored_state.state_version == "1.0"
        assert restored_state.last_signal_date == self.test_date
        assert restored_state.debug_enabled == True
        assert len(restored_state.positions) == 3
        assert len(restored_state.position_history) == 1
        
        # Check position restoration
        aapl_position = restored_state.positions['AAPL']
        assert aapl_position.entry_date == self.test_date
        assert aapl_position.entry_price == 150.0
        assert aapl_position.entry_weight == 0.4
        
        # Check legacy compatibility restoration
        assert len(restored_state.position_entry_dates) == 3
        assert restored_state.position_entry_dates['AAPL'] == self.test_date
    
    def test_file_persistence(self):
        """Test saving and loading state from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save to file
            self.state.save_to_file(temp_file)
            
            # Load from file
            restored_state = TimingState.load_from_file(temp_file)
            
            # Verify restoration
            assert restored_state.last_signal_date == self.test_date
            assert len(restored_state.positions) == 3
            assert len(restored_state.position_history) == 1
            
            # Verify position data
            aapl_position = restored_state.positions['AAPL']
            assert aapl_position.entry_price == 150.0
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestBackwardCompatibility:
    """Test backward compatibility with legacy methods."""
    
    def setup_method(self):
        """Set up test data."""
        self.state = TimingState()
        self.test_date = pd.Timestamp('2023-01-01')
        self.assets = ['AAPL', 'MSFT']
        self.weights = pd.Series([0.6, 0.4], index=self.assets)
        self.prices = pd.Series([150.0, 250.0], index=self.assets)
    
    def test_legacy_methods_still_work(self):
        """Test that legacy methods still work correctly."""
        # Update positions using new method
        self.state.update_positions(self.test_date, self.weights, self.prices)
        
        # Test legacy methods
        assert self.state.is_position_held('AAPL')
        assert self.state.is_position_held('MSFT')
        assert not self.state.is_position_held('GOOGL')
        
        assert self.state.get_position_holding_days('AAPL', self.test_date) == 0
        
        held_assets = self.state.get_held_assets()
        assert 'AAPL' in held_assets
        assert 'MSFT' in held_assets
        
        # Test legacy dictionaries are populated
        assert len(self.state.position_entry_dates) == 2
        assert len(self.state.position_entry_prices) == 2
        assert len(self.state.consecutive_periods) == 2
    
    def test_enhanced_methods_work_with_legacy_data(self):
        """Test enhanced methods work when only legacy data exists."""
        # Manually populate legacy data (simulating old state)
        self.state.position_entry_dates['AAPL'] = self.test_date
        self.state.position_entry_prices['AAPL'] = 150.0
        self.state.consecutive_periods['AAPL'] = 3
        
        # Enhanced methods should still work
        assert self.state.is_position_held('AAPL')
        assert self.state.get_position_holding_days('AAPL', self.test_date + pd.Timedelta(days=5)) == 5
        assert self.state.get_consecutive_periods('AAPL') == 3
        
        held_assets = self.state.get_held_assets()
        assert 'AAPL' in held_assets


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_state_operations(self):
        """Test operations on empty state."""
        state = TimingState()
        
        # Should not raise exceptions
        summary = state.get_portfolio_summary()
        assert summary['total_positions'] == 0
        
        stats = state.get_position_statistics()
        assert stats['total_trades'] == 0
        
        assert state.get_position_return('AAPL') is None
        assert state.get_position_info('AAPL') is None
        assert state.get_consecutive_periods('AAPL') == 0
    
    def test_invalid_position_operations(self):
        """Test operations on non-existent positions."""
        state = TimingState()
        current_date = pd.Timestamp('2023-01-01')
        
        assert state.get_position_holding_days('NONEXISTENT', current_date) is None
        assert not state.is_position_held('NONEXISTENT')
        assert state.get_position_return('NONEXISTENT') is None
        assert state.get_position_info('NONEXISTENT') is None
    
    def test_zero_price_handling(self):
        """Test handling of zero or invalid prices."""
        state = TimingState()
        test_date = pd.Timestamp('2023-01-01')
        weights = pd.Series([0.5], index=['AAPL'])
        prices = pd.Series([0.0], index=['AAPL'])  # Zero price
        
        state.update_positions(test_date, weights, prices)
        
        # Should handle gracefully
        position = state.positions['AAPL']
        assert position.entry_price == 0.0
        
        # Return calculation should handle zero price
        assert state.get_position_return('AAPL', 100.0) is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])