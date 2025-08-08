"""
Tests for state debugging utilities.
Split from test_advanced_state_management.py for better organization.
"""

import pytest
import pandas as pd
from portfolio_backtester.interfaces.timing_state_interface import create_timing_state


class TestStateDebugUtilities:
    """Test debugging utilities."""
    
    def setup_method(self):
        """Set up test data."""
        self.state = create_timing_state(debug_enabled=True)
    
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])