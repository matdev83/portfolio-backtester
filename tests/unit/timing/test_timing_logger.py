"""
Tests for enhanced timing logger.
Split from test_configuration_extensibility.py for better organization.
"""

import pytest
import pandas as pd
import tempfile
import os
import json
from src.portfolio_backtester.timing.timing_logger import TimingLogger


class TestTimingLogger:
    """Test enhanced timing logger."""
    
    def setup_method(self):
        """Set up test environment."""
        self.logger = TimingLogger('test', enable_detailed_logging=True, log_level='DEBUG')
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        assert self.logger.enable_detailed_logging == True
        assert len(self.logger.log_entries) == 0
    
    def test_signal_generation_logging(self):
        """Test signal generation logging."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        self.logger.log_signal_generation(
            strategy_name, current_date, True, 'RSI below threshold',
            rsi_value=25.0
        )
        
        entries = self.logger.get_log_entries()
        assert len(entries) == 1
        
        entry = entries[0]
        assert entry.event_type == 'signal_generation'
        assert entry.strategy_name == strategy_name
        assert entry.current_date == current_date
        assert 'YES' in entry.message
        assert entry.data['should_generate'] == True
        assert entry.data['rsi_value'] == 25.0
    
    def test_position_update_logging(self):
        """Test position update logging."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        self.logger.log_position_update(
            strategy_name, current_date, 'AAPL', 'entry', 0.25, 150.0
        )
        
        entries = self.logger.get_log_entries()
        assert len(entries) == 1
        
        entry = entries[0]
        assert entry.event_type == 'position_update'
        assert 'AAPL' in entry.message
        assert 'entry' in entry.message
        assert entry.data['asset'] == 'AAPL'
        assert entry.data['weight'] == 0.25
        assert entry.data['price'] == 150.0
    
    def test_rebalance_event_logging(self):
        """Test rebalance event logging."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        self.logger.log_rebalance_event(
            strategy_name, current_date, 5, 1.0
        )
        
        entries = self.logger.get_log_entries()
        assert len(entries) == 1
        
        entry = entries[0]
        assert entry.event_type == 'rebalance'
        assert '5 positions' in entry.message
        assert entry.data['num_positions'] == 5
        assert entry.data['total_weight'] == 1.0
    
    def test_error_logging(self):
        """Test error logging."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        self.logger.log_error(
            strategy_name, current_date, 'ValidationError', 'Invalid configuration'
        )
        
        entries = self.logger.get_log_entries()
        assert len(entries) == 1
        
        entry = entries[0]
        assert entry.event_type == 'error'
        assert entry.level == 'ERROR'
        assert 'ValidationError' in entry.message
    
    def test_log_filtering(self):
        """Test log entry filtering."""
        # Add multiple entries
        dates = [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')]
        strategies = ['Strategy1', 'Strategy2']
        
        for i, (date, strategy) in enumerate(zip(dates, strategies)):
            self.logger.log_signal_generation(strategy, date, True, f'Reason {i}')
            self.logger.log_position_update(strategy, date, 'AAPL', 'entry', 0.1, 100.0)
        
        # Test strategy filtering
        strategy1_entries = self.logger.get_log_entries(strategy_name='Strategy1')
        assert len(strategy1_entries) == 2
        assert all(e.strategy_name == 'Strategy1' for e in strategy1_entries)
        
        # Test event type filtering
        signal_entries = self.logger.get_log_entries(event_type='signal_generation')
        assert len(signal_entries) == 2
        assert all(e.event_type == 'signal_generation' for e in signal_entries)
        
        # Test date filtering
        date1_entries = self.logger.get_log_entries(start_date=dates[0], end_date=dates[0])
        assert len(date1_entries) == 2
        assert all(e.current_date == dates[0] for e in date1_entries)
        
        # Test last N filtering
        last_2_entries = self.logger.get_log_entries(last_n=2)
        assert len(last_2_entries) == 2
    
    def test_log_summary(self):
        """Test log summary generation."""
        # Add some entries
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        self.logger.log_signal_generation(strategy_name, current_date, True, 'Test')
        self.logger.log_position_update(strategy_name, current_date, 'AAPL', 'entry', 0.1, 100.0)
        self.logger.log_error(strategy_name, current_date, 'TestError', 'Test error')
        
        summary = self.logger.get_log_summary()
        
        assert summary['total_entries'] == 3
        assert summary['event_counts']['signal_generation'] == 1
        assert summary['event_counts']['position_update'] == 1
        assert summary['event_counts']['error'] == 1
        assert summary['level_counts']['INFO'] == 2
        assert summary['level_counts']['ERROR'] == 1
        assert 'TestStrategy' in summary['strategies']
    
    def test_log_export(self):
        """Test log export functionality."""
        # Add some entries
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        self.logger.log_signal_generation(strategy_name, current_date, True, 'Test')
        
        # Test JSON export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
        
        try:
            self.logger.export_logs(json_file, format='json')
            assert os.path.exists(json_file)
            
            # Verify content
            with open(json_file, 'r') as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]['event_type'] == 'signal_generation'
        finally:
            os.unlink(json_file)
        
        # Test CSV export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_file = f.name
        
        try:
            self.logger.export_logs(csv_file, format='csv')
            assert os.path.exists(csv_file)
        finally:
            os.unlink(csv_file)
    
    def test_global_logger_functions(self):
        """Test global logger convenience functions."""
        from src.portfolio_backtester.timing.timing_logger import (
            get_timing_logger, log_signal_generation, configure_timing_logging
        )
        
        # Configure global logging
        configure_timing_logging(enable_detailed_logging=True, log_level='DEBUG')
        
        # Test convenience functions
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        log_signal_generation(strategy_name, current_date, True, 'Test signal')
        
        # Get global logger and check entries
        global_logger = get_timing_logger()
        entries = global_logger.get_log_entries()
        assert len(entries) >= 1
        
        # Find our entry
        our_entries = [e for e in entries if e.strategy_name == strategy_name]
        assert len(our_entries) == 1
        assert our_entries[0].event_type == 'signal_generation'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])