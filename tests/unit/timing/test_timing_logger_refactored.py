"""
Tests for refactored TimingLogger with SOLID principles.
Ensures backward compatibility while testing new architecture.
"""

import pandas as pd
import tempfile
import os
from portfolio_backtester.timing.timing_logger_refactored import (
    TimingLogger,
    get_timing_logger,
    configure_timing_logging
)
from portfolio_backtester.timing.logging import LogEntryManager, LogExporter, LogAnalyzer


class TestTimingLoggerRefactored:
    """Test refactored TimingLogger class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.logger = TimingLogger('test_refactored', enable_detailed_logging=True, log_level='DEBUG')
    
    def test_logger_initialization(self):
        """Test logger initialization with specialized components."""
        assert self.logger.enable_detailed_logging
        assert isinstance(self.logger.entry_manager, LogEntryManager)
        assert isinstance(self.logger.exporter, LogExporter)
        assert isinstance(self.logger.analyzer, LogAnalyzer)
    
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
        assert entry.data['should_generate']
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
        assert entry.data['asset'] == 'AAPL'
        assert entry.data['action'] == 'entry'
        assert entry.data['weight'] == 0.25
        assert entry.data['price'] == 150.0
    
    def test_error_logging(self):
        """Test error logging."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        self.logger.log_error(
            strategy_name, current_date, 'ValidationError', 'Invalid parameter'
        )
        
        entries = self.logger.get_log_entries()
        assert len(entries) == 1
        
        entry = entries[0]
        assert entry.event_type == 'error'
        assert entry.level == 'ERROR'
        assert entry.data['error_type'] == 'ValidationError'
        assert entry.data['error_message'] == 'Invalid parameter'
    
    def test_log_filtering(self):
        """Test log entry filtering capabilities."""
        strategy1 = 'Strategy1'
        strategy2 = 'Strategy2'
        current_date = pd.Timestamp('2023-01-01')
        
        # Add multiple entries
        self.logger.log_signal_generation(strategy1, current_date, True, 'Reason1')
        self.logger.log_position_update(strategy1, current_date, 'AAPL', 'entry', 0.25)
        self.logger.log_signal_generation(strategy2, current_date, False, 'Reason2')
        self.logger.log_error(strategy2, current_date, 'Error', 'Test error')
        
        # Test filtering by strategy
        strategy1_entries = self.logger.get_log_entries(strategy_name=strategy1)
        assert len(strategy1_entries) == 2
        assert all(e.strategy_name == strategy1 for e in strategy1_entries)
        
        # Test filtering by event type
        signal_entries = self.logger.get_log_entries(event_type='signal_generation')
        assert len(signal_entries) == 2
        assert all(e.event_type == 'signal_generation' for e in signal_entries)
        
        # Test last_n filtering
        last_2_entries = self.logger.get_log_entries(last_n=2)
        assert len(last_2_entries) == 2
    
    def test_log_summary_statistics(self):
        """Test log summary and statistics."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        # Add various types of entries
        self.logger.log_signal_generation(strategy_name, current_date, True, 'Reason1')
        self.logger.log_position_update(strategy_name, current_date, 'AAPL', 'entry', 0.25)
        self.logger.log_rebalance_event(strategy_name, current_date, 5, 1.0)
        self.logger.log_error(strategy_name, current_date, 'TestError', 'Test message')
        
        summary = self.logger.get_log_summary()
        assert summary['total_entries'] == 4
        assert 'event_counts' in summary
        assert 'level_counts' in summary
        assert summary['event_counts']['signal_generation'] == 1
        assert summary['event_counts']['position_update'] == 1
        assert summary['level_counts']['ERROR'] == 1
    
    def test_error_analysis(self):
        """Test error analysis functionality."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        # Add some errors
        self.logger.log_error(strategy_name, current_date, 'ValidationError', 'Error 1')
        self.logger.log_error(strategy_name, current_date, 'RuntimeError', 'Error 2')
        self.logger.log_signal_generation(strategy_name, current_date, True, 'Normal log')
        
        error_analysis = self.logger.get_error_analysis()
        assert error_analysis['total_errors'] == 2
        assert error_analysis['error_rate'] == 2/3  # 2 errors out of 3 total entries
        assert 'ValidationError' in error_analysis['error_types']
        assert 'RuntimeError' in error_analysis['error_types']
    
    def test_performance_insights(self):
        """Test performance insights functionality."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        # Add performance-related entries
        self.logger.log_signal_generation(strategy_name, current_date, True, 'Signal 1')
        self.logger.log_signal_generation(strategy_name, current_date, False, 'Signal 2')
        self.logger.log_rebalance_event(strategy_name, current_date, 5, 1.0)
        self.logger.log_position_update(strategy_name, current_date, 'AAPL', 'entry', 0.25)
        
        insights = self.logger.get_performance_insights()
        assert insights['signal_generation']['total_signals'] == 2
        assert insights['rebalancing']['total_rebalances'] == 1
        assert insights['position_updates']['total_updates'] == 1
        assert strategy_name in insights['strategy_activity']
    
    def test_comprehensive_report(self):
        """Test comprehensive analysis report generation."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        # Add diverse entries
        self.logger.log_signal_generation(strategy_name, current_date, True, 'Signal')
        self.logger.log_position_update(strategy_name, current_date, 'AAPL', 'entry', 0.25)
        self.logger.log_error(strategy_name, current_date, 'TestError', 'Test error')
        
        report = self.logger.generate_comprehensive_report()
        assert 'summary' in report
        assert 'errors' in report
        assert 'performance' in report
        assert 'daily_activity' in report
        
        # Verify report structure
        assert report['summary']['total_entries'] == 3
        assert report['errors']['total_errors'] == 1
        assert report['daily_activity']['days_analyzed'] == 1
    
    def test_log_export_functionality(self):
        """Test log export to different formats."""
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        # Add some entries
        self.logger.log_signal_generation(strategy_name, current_date, True, 'Test log')
        self.logger.log_position_update(strategy_name, current_date, 'AAPL', 'entry', 0.25)
        
        # Test JSON export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
        
        try:
            self.logger.export_logs(json_file, format='json')
            assert os.path.exists(json_file)
            assert os.path.getsize(json_file) > 0
        finally:
            os.unlink(json_file)
        
        # Test CSV export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_file = f.name
        
        try:
            self.logger.export_logs(csv_file, format='csv')
            assert os.path.exists(csv_file)
            assert os.path.getsize(csv_file) > 0
        finally:
            os.unlink(csv_file)
    
    def test_global_logger_functions(self):
        """Test global logger functions."""
        # Test getting global logger
        global_logger = get_timing_logger()
        assert isinstance(global_logger, TimingLogger)
        
        # Test getting same instance
        same_logger = get_timing_logger()
        assert global_logger is same_logger
        
        # Test getting different named logger
        named_logger = get_timing_logger('test_named')
        assert named_logger is not global_logger
    
    def test_configure_timing_logging(self):
        """Test global timing logging configuration."""
        configure_timing_logging(
            enable_detailed_logging=True,
            log_level='DEBUG'
        )
        
        logger = get_timing_logger()
        assert logger.enable_detailed_logging is True
    
    def test_convenience_functions(self):
        """Test convenience logging functions."""
        from portfolio_backtester.timing.timing_logger_refactored import (
            log_signal_generation, log_position_update, log_rebalance_event
        )
        
        strategy_name = 'TestStrategy'
        current_date = pd.Timestamp('2023-01-01')
        
        # Test convenience functions
        log_signal_generation(strategy_name, current_date, True, 'Test signal')
        log_position_update(strategy_name, current_date, 'AAPL', 'entry', 0.25)
        log_rebalance_event(strategy_name, current_date, 5, 1.0)
        
        # Verify entries were logged to global logger
        global_logger = get_timing_logger()
        entries = global_logger.get_log_entries()
        assert len(entries) >= 3  # May have entries from other tests