"""
Enhanced logging system for timing framework.
Refactored version using SOLID principles with facade pattern for backward compatibility.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, List

from .logging import LogEntryManager, LogExporter, LogAnalyzer, TimingLogEntry
from ..interfaces.attribute_accessor_interface import (
    IModuleAttributeAccessor,
    create_module_attribute_accessor,
)


class TimingLogger:
    """
    Enhanced logger for timing framework operations.

    Refactored facade that delegates to specialized classes following SOLID principles:
    - LogEntryManager: Handles log entry creation, storage, and filtering
    - LogExporter: Handles exporting logs to different formats
    - LogAnalyzer: Provides analysis and summary of log data
    """

    def __init__(
        self,
        name: str = "timing",
        enable_detailed_logging: bool = False,
        log_level: str = "INFO",
        max_entries: int = 1000,
        module_attribute_accessor: Optional[IModuleAttributeAccessor] = None,
    ):
        """
        Initialize timing logger.

        Args:
            name: Logger name
            enable_detailed_logging: Enable detailed operation logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            max_entries: Maximum number of entries to keep in memory
            module_attribute_accessor: Injected accessor for module attribute access (DIP)
        """
        self.logger = logging.getLogger(f"portfolio_backtester.timing.{name}")
        self.enable_detailed_logging = enable_detailed_logging

        # Dependency injection for attribute access (DIP)
        self._module_accessor = module_attribute_accessor or create_module_attribute_accessor()

        # Initialize specialized components with dependency injection
        self.entry_manager = LogEntryManager(max_entries=max_entries)
        self.exporter = LogExporter()
        self.analyzer = LogAnalyzer()

        # Set log level using injected dependency instead of direct getattr
        level = self._module_accessor.get_module_attribute(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)

        # Create formatter for timing logs
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(strategy)s] %(message)s",
            defaults={"strategy": "Unknown"},
        )

        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def log_signal_generation(
        self,
        strategy_name: str,
        current_date: pd.Timestamp,
        should_generate: bool,
        reason: str,
        **kwargs,
    ) -> None:
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
            "should_generate": should_generate,
            "reason": reason,
            "date": current_date.isoformat(),
            **kwargs,
        }

        self.entry_manager.add_entry(
            "signal_generation", strategy_name, current_date, message, data, "INFO"
        )

        if self.enable_detailed_logging:
            extra = {"strategy": strategy_name}
            self.logger.info(message, extra=extra)

    def log_state_change(
        self,
        strategy_name: str,
        current_date: pd.Timestamp,
        change_type: str,
        old_value: Any,
        new_value: Any,
        **kwargs,
    ) -> None:
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
            "change_type": change_type,
            "old_value": str(old_value),
            "new_value": str(new_value),
            "date": current_date.isoformat(),
            **kwargs,
        }

        self.entry_manager.add_entry(
            "state_change", strategy_name, current_date, message, data, "DEBUG"
        )

        if self.enable_detailed_logging:
            extra = {"strategy": strategy_name}
            self.logger.debug(message, extra=extra)

    def log_position_update(
        self,
        strategy_name: str,
        current_date: pd.Timestamp,
        asset: str,
        action: str,
        weight: float,
        price: Optional[float] = None,
        **kwargs,
    ) -> None:
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
            "asset": asset,
            "action": action,
            "weight": weight,
            "price": price,
            "date": current_date.isoformat(),
            **kwargs,
        }

        self.entry_manager.add_entry(
            "position_update", strategy_name, current_date, message, data, "INFO"
        )

        if self.enable_detailed_logging:
            extra = {"strategy": strategy_name}
            self.logger.info(message, extra=extra)

    def log_rebalance_event(
        self,
        strategy_name: str,
        current_date: pd.Timestamp,
        num_positions: int,
        total_weight: float,
        **kwargs,
    ) -> None:
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
            "num_positions": num_positions,
            "total_weight": total_weight,
            "date": current_date.isoformat(),
            **kwargs,
        }

        self.entry_manager.add_entry(
            "rebalance", strategy_name, current_date, message, data, "INFO"
        )

        if self.enable_detailed_logging:
            extra = {"strategy": strategy_name}
            self.logger.info(message, extra=extra)

    def log_timing_decision(
        self,
        strategy_name: str,
        current_date: pd.Timestamp,
        controller_type: str,
        decision: str,
        details: Dict[str, Any],
    ) -> None:
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
            "controller_type": controller_type,
            "decision": decision,
            "details": details,
            "date": current_date.isoformat(),
        }

        self.entry_manager.add_entry(
            "timing_decision", strategy_name, current_date, message, data, "DEBUG"
        )

        if self.enable_detailed_logging:
            extra = {"strategy": strategy_name}
            self.logger.debug(message, extra=extra)

    def log_performance_metric(
        self,
        strategy_name: str,
        current_date: pd.Timestamp,
        metric_name: str,
        metric_value: float,
        **kwargs,
    ) -> None:
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
            "metric_name": metric_name,
            "metric_value": metric_value,
            "date": current_date.isoformat(),
            **kwargs,
        }

        self.entry_manager.add_entry(
            "performance_metric", strategy_name, current_date, message, data, "INFO"
        )

        if self.enable_detailed_logging:
            extra = {"strategy": strategy_name}
            self.logger.info(message, extra=extra)

    def log_error(
        self,
        strategy_name: str,
        current_date: pd.Timestamp,
        error_type: str,
        error_message: str,
        **kwargs,
    ) -> None:
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
            "error_type": error_type,
            "error_message": error_message,
            "date": current_date.isoformat(),
            **kwargs,
        }

        self.entry_manager.add_entry("error", strategy_name, current_date, message, data, "ERROR")

        extra = {"strategy": strategy_name}
        self.logger.error(message, extra=extra)

    # Delegate to LogEntryManager
    def get_log_entries(
        self,
        strategy_name: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        last_n: Optional[int] = None,
    ) -> List[TimingLogEntry]:
        """Get filtered log entries - delegates to LogEntryManager."""
        return self.entry_manager.get_entries(
            strategy_name=strategy_name,
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
            last_n=last_n,
        )

    def clear_logs(self) -> None:
        """Clear all log entries - delegates to LogEntryManager."""
        self.entry_manager.clear_entries()
        self.logger.info("Cleared timing log entries")

    # Delegate to LogAnalyzer
    def get_log_summary(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of log entries - delegates to LogAnalyzer."""
        entries = self.entry_manager.get_entries(strategy_name=strategy_name)
        return self.analyzer.get_summary_statistics(entries)

    def get_error_analysis(self) -> Dict[str, Any]:
        """Get error analysis - delegates to LogAnalyzer."""
        entries = self.entry_manager.get_entries()
        return self.analyzer.get_error_analysis(entries)

    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights - delegates to LogAnalyzer."""
        entries = self.entry_manager.get_entries()
        return self.analyzer.get_performance_insights(entries)

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report - delegates to LogAnalyzer."""
        entries = self.entry_manager.get_entries()
        return self.analyzer.generate_comprehensive_report(entries)

    # Delegate to LogExporter
    def export_logs(
        self, file_path: str, strategy_name: Optional[str] = None, format: str = "json", **kwargs
    ) -> None:
        """Export log entries to file - delegates to LogExporter."""
        entries = self.entry_manager.get_entries(strategy_name=strategy_name)
        self.exporter.export(entries, file_path, format, **kwargs)

    def print_recent_logs(self, last_n: int = 10, strategy_name: Optional[str] = None) -> None:
        """
        Print recent log entries in a readable format.

        Args:
            last_n: Number of recent entries to print
            strategy_name: Filter by strategy name
        """
        entries = self.entry_manager.get_entries(strategy_name=strategy_name, last_n=last_n)

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


def get_timing_logger(name: str = "global", **kwargs) -> TimingLogger:
    """
    Get or create a timing logger instance.

    Args:
        name: Logger name
        **kwargs: Logger configuration options

    Returns:
        TimingLogger instance
    """
    global _global_timing_logger

    if name == "global":
        if _global_timing_logger is None:
            _global_timing_logger = TimingLogger(name="global", **kwargs)
        return _global_timing_logger
    else:
        return TimingLogger(name=name, **kwargs)


def configure_timing_logging(
    enable_detailed_logging: bool = False, log_level: str = "INFO", log_file: Optional[str] = None
) -> None:
    """
    Configure global timing logging settings.

    Args:
        enable_detailed_logging: Enable detailed operation logging
        log_level: Logging level
        log_file: Optional log file path
    """
    global _global_timing_logger

    _global_timing_logger = TimingLogger(
        name="global", enable_detailed_logging=enable_detailed_logging, log_level=log_level
    )

    if log_file:
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(strategy)s] %(message)s",
            defaults={"strategy": "Unknown"},
        )
        file_handler.setFormatter(formatter)
        _global_timing_logger.logger.addHandler(file_handler)


# Convenience functions for common logging operations - delegate to global logger
def log_signal_generation(
    strategy_name: str, current_date: pd.Timestamp, should_generate: bool, reason: str, **kwargs
) -> None:
    """Convenience function for logging signal generation."""
    logger = get_timing_logger()
    logger.log_signal_generation(strategy_name, current_date, should_generate, reason, **kwargs)


def log_position_update(
    strategy_name: str,
    current_date: pd.Timestamp,
    asset: str,
    action: str,
    weight: float,
    price: Optional[float] = None,
    **kwargs,
) -> None:
    """Convenience function for logging position updates."""
    logger = get_timing_logger()
    logger.log_position_update(strategy_name, current_date, asset, action, weight, price, **kwargs)


def log_rebalance_event(
    strategy_name: str,
    current_date: pd.Timestamp,
    num_positions: int,
    total_weight: float,
    **kwargs,
) -> None:
    """Convenience function for logging rebalance events."""
    logger = get_timing_logger()
    logger.log_rebalance_event(strategy_name, current_date, num_positions, total_weight, **kwargs)
