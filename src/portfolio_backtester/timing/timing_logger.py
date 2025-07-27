"""
Enhanced logging system for timing framework.
Provides detailed logging of timing decisions, state changes, and performance metrics.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
import json


@dataclass
class TimingLogEntry:
    """Represents a single timing log entry."""
    timestamp: datetime
    event_type: str  # 'signal_generation', 'state_change', 'rebalance', 'position_update', 'error'
    strategy_name: str
    current_date: pd.Timestamp
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    level: str = 'INFO'


class TimingLogger:
    """Enhanced logger for timing framework operations."""
    
    def __init__(self, name: str = 'timing', enable_detailed_logging: bool = False, log_level: str = 'INFO'):
        """
        Initialize timing logger.
        
        Args:
            name: Logger name
            enable_detailed_logging: Enable detailed operation logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = logging.getLogger(f'portfolio_backtester.timing.{name}')
        self.enable_detailed_logging = enable_detailed_logging
        self.log_entries: List[TimingLogEntry] = []
        
        # Set log level
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Create formatter for timing logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(strategy)s] %(message)s',
            defaults={'strategy': 'Unknown'}
        )
        
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log_signal_generation(self, strategy_name: str, current_date: pd.Timestamp, 
                            should_generate: bool, reason: str, **kwargs):
        """
        Log signal generation decision.
        
        Args:
            strategy_name: Name of the strategy
            current_date: Current date
            should_generate: Whether signal should be generated
            reason: Reason for the decision
            **kwargs: Additional data to log
        """
        message = f"Signal generation: {'YES' if should_generate else 'NO'} - {reason}"
        
        data = {
            'should_generate': should_generate,
            'reason': reason,
            'date': current_date.isoformat(),
            **kwargs
        }
        
        self._log_entry('signal_generation', strategy_name, current_date, message, data, 'INFO')
        
        if self.enable_detailed_logging:
            extra = {'strategy': strategy_name}
            self.logger.info(message, extra=extra)
    
    def log_state_change(self, strategy_name: str, current_date: pd.Timestamp,
                        change_type: str, old_value: Any, new_value: Any, **kwargs):
        """
        Log timing state changes.
        
        Args:
            strategy_name: Name of the strategy
            current_date: Current date
            change_type: Type of state change
            old_value: Previous value
            new_value: New value
            **kwargs: Additional data to log
        """
        message = f"State change [{change_type}]: {old_value} -> {new_value}"
        
        data = {
            'change_type': change_type,
            'old_value': str(old_value),
            'new_value': str(new_value),
            'date': current_date.isoformat(),
            **kwargs
        }
        
        self._log_entry('state_change', strategy_name, current_date, message, data, 'DEBUG')
        
        if self.enable_detailed_logging:
            extra = {'strategy': strategy_name}
            self.logger.debug(message, extra=extra)
    
    def log_position_update(self, strategy_name: str, current_date: pd.Timestamp,
                          asset: str, action: str, weight: float, price: Optional[float] = None, **kwargs):
        """
        Log position updates.
        
        Args:
            strategy_name: Name of the strategy
            current_date: Current date
            asset: Asset symbol
            action: Action taken ('entry', 'exit', 'weight_change')
            weight: Position weight
            price: Asset price (if available)
            **kwargs: Additional data to log
        """
        price_str = f" @ ${price:.2f}" if price is not None else ""
        message = f"Position {action}: {asset} weight={weight:.4f}{price_str}"
        
        data = {
            'asset': asset,
            'action': action,
            'weight': weight,
            'price': price,
            'date': current_date.isoformat(),
            **kwargs
        }
        
        self._log_entry('position_update', strategy_name, current_date, message, data, 'INFO')
        
        if self.enable_detailed_logging:
            extra = {'strategy': strategy_name}
            self.logger.info(message, extra=extra)
    
    def log_rebalance_event(self, strategy_name: str, current_date: pd.Timestamp,
                          num_positions: int, total_weight: float, **kwargs):
        """
        Log rebalancing events.
        
        Args:
            strategy_name: Name of the strategy
            current_date: Current date
            num_positions: Number of positions
            total_weight: Total portfolio weight
            **kwargs: Additional data to log
        """
        message = f"Rebalance: {num_positions} positions, total weight={total_weight:.4f}"
        
        data = {
            'num_positions': num_positions,
            'total_weight': total_weight,
            'date': current_date.isoformat(),
            **kwargs
        }
        
        self._log_entry('rebalance', strategy_name, current_date, message, data, 'INFO')
        
        if self.enable_detailed_logging:
            extra = {'strategy': strategy_name}
            self.logger.info(message, extra=extra)
    
    def log_timing_decision(self, strategy_name: str, current_date: pd.Timestamp,
                          controller_type: str, decision: str, details: Dict[str, Any]):
        """
        Log timing controller decisions.
        
        Args:
            strategy_name: Name of the strategy
            current_date: Current date
            controller_type: Type of timing controller
            decision: Decision made
            details: Additional decision details
        """
        message = f"Timing decision [{controller_type}]: {decision}"
        
        data = {
            'controller_type': controller_type,
            'decision': decision,
            'details': details,
            'date': current_date.isoformat()
        }
        
        self._log_entry('timing_decision', strategy_name, current_date, message, data, 'DEBUG')
        
        if self.enable_detailed_logging:
            extra = {'strategy': strategy_name}
            self.logger.debug(message, extra=extra)
    
    def log_performance_metric(self, strategy_name: str, current_date: pd.Timestamp,
                             metric_name: str, metric_value: float, **kwargs):
        """
        Log performance metrics.
        
        Args:
            strategy_name: Name of the strategy
            current_date: Current date
            metric_name: Name of the metric
            metric_value: Metric value
            **kwargs: Additional data to log
        """
        message = f"Performance metric [{metric_name}]: {metric_value:.4f}"
        
        data = {
            'metric_name': metric_name,
            'metric_value': metric_value,
            'date': current_date.isoformat(),
            **kwargs
        }
        
        self._log_entry('performance_metric', strategy_name, current_date, message, data, 'INFO')
        
        if self.enable_detailed_logging:
            extra = {'strategy': strategy_name}
            self.logger.info(message, extra=extra)
    
    def log_error(self, strategy_name: str, current_date: pd.Timestamp,
                  error_type: str, error_message: str, **kwargs):
        """
        Log timing-related errors.
        
        Args:
            strategy_name: Name of the strategy
            current_date: Current date
            error_type: Type of error
            error_message: Error message
            **kwargs: Additional data to log
        """
        message = f"Timing error [{error_type}]: {error_message}"
        
        data = {
            'error_type': error_type,
            'error_message': error_message,
            'date': current_date.isoformat(),
            **kwargs
        }
        
        self._log_entry('error', strategy_name, current_date, message, data, 'ERROR')
        
        extra = {'strategy': strategy_name}
        self.logger.error(message, extra=extra)
    
    def _log_entry(self, event_type: str, strategy_name: str, current_date: pd.Timestamp,
                   message: str, data: Dict[str, Any], level: str):
        """Internal method to create and store log entry."""
        entry = TimingLogEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            strategy_name=strategy_name,
            current_date=current_date,
            message=message,
            data=data,
            level=level
        )
        
        self.log_entries.append(entry)
        
        # Keep log entries manageable (last 1000 entries)
        if len(self.log_entries) > 1000:
            self.log_entries = self.log_entries[-500:]
    
    def get_log_entries(self, strategy_name: Optional[str] = None,
                       event_type: Optional[str] = None,
                       start_date: Optional[pd.Timestamp] = None,
                       end_date: Optional[pd.Timestamp] = None,
                       last_n: Optional[int] = None) -> List[TimingLogEntry]:
        """
        Get filtered log entries.
        
        Args:
            strategy_name: Filter by strategy name
            event_type: Filter by event type
            start_date: Filter by start date
            end_date: Filter by end date
            last_n: Return last N entries
            
        Returns:
            List of filtered log entries
        """
        entries = self.log_entries
        
        # Apply filters
        if strategy_name:
            entries = [e for e in entries if e.strategy_name == strategy_name]
        
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        
        if start_date:
            entries = [e for e in entries if e.current_date >= start_date]
        
        if end_date:
            entries = [e for e in entries if e.current_date <= end_date]
        
        # Sort by timestamp
        entries = sorted(entries, key=lambda x: x.timestamp)
        
        # Return last N if specified
        if last_n:
            entries = entries[-last_n:]
        
        return entries
    
    def get_log_summary(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of log entries.
        
        Args:
            strategy_name: Filter by strategy name
            
        Returns:
            Summary statistics
        """
        entries = self.get_log_entries(strategy_name=strategy_name)
        
        if not entries:
            return {'total_entries': 0}
        
        # Count by event type
        event_counts = {}
        level_counts = {}
        
        for entry in entries:
            event_counts[entry.event_type] = event_counts.get(entry.event_type, 0) + 1
            level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
        
        # Get date range
        dates = [entry.current_date for entry in entries]
        date_range = {
            'start_date': min(dates).isoformat(),
            'end_date': max(dates).isoformat()
        }
        
        return {
            'total_entries': len(entries),
            'event_counts': event_counts,
            'level_counts': level_counts,
            'date_range': date_range,
            'strategies': list(set(entry.strategy_name for entry in entries))
        }
    
    def export_logs(self, file_path: str, strategy_name: Optional[str] = None,
                   format: str = 'json'):
        """
        Export log entries to file.
        
        Args:
            file_path: Output file path
            strategy_name: Filter by strategy name
            format: Export format ('json', 'csv')
        """
        entries = self.get_log_entries(strategy_name=strategy_name)
        
        if format.lower() == 'json':
            self._export_json(entries, file_path)
        elif format.lower() == 'csv':
            self._export_csv(entries, file_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, entries: List[TimingLogEntry], file_path: str):
        """Export entries to JSON format."""
        data = []
        for entry in entries:
            data.append({
                'timestamp': entry.timestamp.isoformat(),
                'event_type': entry.event_type,
                'strategy_name': entry.strategy_name,
                'current_date': entry.current_date.isoformat(),
                'message': entry.message,
                'data': entry.data,
                'level': entry.level
            })
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_csv(self, entries: List[TimingLogEntry], file_path: str):
        """Export entries to CSV format."""
        import csv
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['timestamp', 'event_type', 'strategy_name', 'current_date', 
                           'message', 'level', 'data'])
            
            # Write entries
            for entry in entries:
                writer.writerow([
                    entry.timestamp.isoformat(),
                    entry.event_type,
                    entry.strategy_name,
                    entry.current_date.isoformat(),
                    entry.message,
                    entry.level,
                    json.dumps(entry.data)
                ])
    
    def clear_logs(self):
        """Clear all log entries."""
        self.log_entries.clear()
        self.logger.info("Cleared timing log entries")
    
    def print_recent_logs(self, last_n: int = 10, strategy_name: Optional[str] = None):
        """
        Print recent log entries in a readable format.
        
        Args:
            last_n: Number of recent entries to print
            strategy_name: Filter by strategy name
        """
        entries = self.get_log_entries(strategy_name=strategy_name, last_n=last_n)
        
        if not entries:
            print("No log entries found")
            return
        
        print(f"\n{'='*80}")
        print(f"RECENT TIMING LOG ENTRIES (Last {len(entries)})")
        print(f"{'='*80}")
        
        for i, entry in enumerate(entries, 1):
            print(f"\n{i}. [{entry.level}] {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Strategy: {entry.strategy_name}")
            print(f"   Date: {entry.current_date.date()}")
            print(f"   Event: {entry.event_type}")
            print(f"   Message: {entry.message}")
            
            if entry.data and self.enable_detailed_logging:
                print(f"   Data: {entry.data}")
        
        print(f"\n{'='*80}")


# Global timing logger instance
_global_timing_logger: Optional[TimingLogger] = None


def get_timing_logger(name: str = 'global', **kwargs) -> TimingLogger:
    """
    Get or create a timing logger instance.
    
    Args:
        name: Logger name
        **kwargs: Logger configuration options
        
    Returns:
        TimingLogger instance
    """
    global _global_timing_logger
    
    if name == 'global':
        if _global_timing_logger is None:
            _global_timing_logger = TimingLogger(name='global', **kwargs)
        return _global_timing_logger
    else:
        return TimingLogger(name=name, **kwargs)


def configure_timing_logging(enable_detailed_logging: bool = False, 
                           log_level: str = 'INFO',
                           log_file: Optional[str] = None):
    """
    Configure global timing logging settings.
    
    Args:
        enable_detailed_logging: Enable detailed operation logging
        log_level: Logging level
        log_file: Optional log file path
    """
    global _global_timing_logger
    
    _global_timing_logger = TimingLogger(
        name='global',
        enable_detailed_logging=enable_detailed_logging,
        log_level=log_level
    )
    
    if log_file:
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(strategy)s] %(message)s',
            defaults={'strategy': 'Unknown'}
        )
        file_handler.setFormatter(formatter)
        _global_timing_logger.logger.addHandler(file_handler)


# Convenience functions for common logging operations
def log_signal_generation(strategy_name: str, current_date: pd.Timestamp, 
                        should_generate: bool, reason: str, **kwargs):
    """Convenience function for logging signal generation."""
    logger = get_timing_logger()
    logger.log_signal_generation(strategy_name, current_date, should_generate, reason, **kwargs)


def log_position_update(strategy_name: str, current_date: pd.Timestamp,
                      asset: str, action: str, weight: float, price: Optional[float] = None, **kwargs):
    """Convenience function for logging position updates."""
    logger = get_timing_logger()
    logger.log_position_update(strategy_name, current_date, asset, action, weight, price, **kwargs)


def log_rebalance_event(strategy_name: str, current_date: pd.Timestamp,
                      num_positions: int, total_weight: float, **kwargs):
    """Convenience function for logging rebalance events."""
    logger = get_timing_logger()
    logger.log_rebalance_event(strategy_name, current_date, num_positions, total_weight, **kwargs)