"""
Log analysis functionality for timing framework.
Provides statistical analysis and insights from log entries.
"""

from typing import Dict, Any, List
from collections import Counter
from .log_entry_manager import TimingLogEntry


class LogAnalyzer:
    """Analyzes timing log entries to provide insights and statistics."""

    @staticmethod
    def get_summary_statistics(entries: List[TimingLogEntry]) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics of log entries.

        Args:
            entries: List of log entries to analyze

        Returns:
            Dictionary with summary statistics
        """
        if not entries:
            return {"total_entries": 0}

        # Count by event type
        event_counts = Counter(entry.event_type for entry in entries)

        # Count by log level
        level_counts = Counter(entry.level for entry in entries)

        # Count by strategy
        strategy_counts = Counter(entry.strategy_name for entry in entries)

        # Get date range
        dates = [entry.current_date for entry in entries]
        date_range = {
            "start_date": min(dates).isoformat(),
            "end_date": max(dates).isoformat(),
            "days_covered": (max(dates) - min(dates)).days,
        }

        # Time distribution
        timestamps = [entry.timestamp for entry in entries]
        time_range = {
            "first_log": min(timestamps).isoformat(),
            "last_log": max(timestamps).isoformat(),
            "duration_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600,
        }

        return {
            "total_entries": len(entries),
            "event_counts": dict(event_counts),
            "level_counts": dict(level_counts),
            "strategy_counts": dict(strategy_counts),
            "date_range": date_range,
            "time_range": time_range,
            "unique_strategies": len(strategy_counts),
            "unique_event_types": len(event_counts),
            "strategies": list(strategy_counts.keys()),  # Backward compatibility
        }

    @staticmethod
    def get_error_analysis(entries: List[TimingLogEntry]) -> Dict[str, Any]:
        """
        Analyze error entries for patterns and insights.

        Args:
            entries: List of log entries to analyze

        Returns:
            Dictionary with error analysis
        """
        error_entries = [e for e in entries if e.level == "ERROR"]

        if not error_entries:
            return {"total_errors": 0, "error_rate": 0.0}

        # Error patterns
        error_types: Counter[str] = Counter()
        error_strategies: Counter[str] = Counter()

        for entry in error_entries:
            error_type = entry.data.get("error_type", entry.event_type)
            error_types[error_type] += 1
            error_strategies[entry.strategy_name] += 1

        # Error frequency over time
        error_dates = [entry.current_date.date() for entry in error_entries]
        error_by_date = Counter(error_dates)

        return {
            "total_errors": len(error_entries),
            "error_rate": len(error_entries) / len(entries) if entries else 0,
            "error_types": dict(error_types),
            "error_strategies": dict(error_strategies),
            "error_by_date": {str(k): v for k, v in error_by_date.items()},
            "most_common_error_type": error_types.most_common(1)[0] if error_types else None,
            "most_error_prone_strategy": (
                error_strategies.most_common(1)[0] if error_strategies else None
            ),
        }

    @staticmethod
    def get_performance_insights(entries: List[TimingLogEntry]) -> Dict[str, Any]:
        """
        Get performance-related insights from log entries.

        Args:
            entries: List of log entries to analyze

        Returns:
            Dictionary with performance insights
        """
        # Signal generation analysis
        signal_entries = [e for e in entries if e.event_type == "signal_generation"]
        rebalance_entries = [e for e in entries if e.event_type == "rebalance"]

        insights: Dict[str, Any] = {
            "signal_generation": {
                "total_signals": len(signal_entries),
                "signals_per_strategy": Counter(e.strategy_name for e in signal_entries),
            },
            "rebalancing": {
                "total_rebalances": len(rebalance_entries),
                "rebalances_per_strategy": Counter(e.strategy_name for e in rebalance_entries),
            },
        }

        # Position update patterns
        position_entries = [e for e in entries if e.event_type == "position_update"]
        if position_entries:
            position_actions: Counter[str] = Counter()
            for entry in position_entries:
                action = entry.data.get("action", "unknown")
                position_actions[action] += 1

            insights["position_updates"] = {
                "total_updates": len(position_entries),
                "actions": dict(position_actions),
            }

        # Activity patterns by strategy
        strategy_activity: Dict[str, Dict[str, Any]] = {}
        for strategy in set(e.strategy_name for e in entries):
            strategy_entries = [e for e in entries if e.strategy_name == strategy]
            strategy_activity[strategy] = {
                "total_logs": len(strategy_entries),
                "event_distribution": dict(Counter(e.event_type for e in strategy_entries)),
                "level_distribution": dict(Counter(e.level for e in strategy_entries)),
            }

        insights["strategy_activity"] = strategy_activity

        return insights

    @staticmethod
    def get_daily_activity_report(entries: List[TimingLogEntry]) -> Dict[str, Any]:
        """
        Generate daily activity report.

        Args:
            entries: List of log entries to analyze

        Returns:
            Dictionary with daily activity statistics
        """
        if not entries:
            return {"days_analyzed": 0}

        # Group by date
        daily_activity: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            date_str = entry.current_date.date().isoformat()

            if date_str not in daily_activity:
                daily_activity[date_str] = {
                    "total_entries": 0,
                    "event_counts": Counter(),
                    "level_counts": Counter(),
                    "strategy_counts": Counter(),
                }

            daily_activity[date_str]["total_entries"] += 1
            daily_activity[date_str]["event_counts"][entry.event_type] += 1
            daily_activity[date_str]["level_counts"][entry.level] += 1
            daily_activity[date_str]["strategy_counts"][entry.strategy_name] += 1

        # Convert counters to regular dicts for JSON serialization
        for date_data in daily_activity.values():
            date_data["event_counts"] = dict(date_data["event_counts"])
            date_data["level_counts"] = dict(date_data["level_counts"])
            date_data["strategy_counts"] = dict(date_data["strategy_counts"])

        # Calculate daily averages
        total_days = len(daily_activity)
        total_entries = sum(int(day["total_entries"]) for day in daily_activity.values())

        return {
            "days_analyzed": total_days,
            "daily_activity": daily_activity,
            "averages": {
                "entries_per_day": total_entries / total_days if total_days > 0 else 0,
                "most_active_day": (
                    max(daily_activity.items(), key=lambda x: int(x[1]["total_entries"]))[0]
                    if daily_activity
                    else None
                ),
                "least_active_day": (
                    min(daily_activity.items(), key=lambda x: int(x[1]["total_entries"]))[0]
                    if daily_activity
                    else None
                ),
            },
        }

    @classmethod
    def generate_comprehensive_report(cls, entries: List[TimingLogEntry]) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.

        Args:
            entries: List of log entries to analyze

        Returns:
            Dictionary with comprehensive analysis results
        """
        return {
            "summary": cls.get_summary_statistics(entries),
            "errors": cls.get_error_analysis(entries),
            "performance": cls.get_performance_insights(entries),
            "daily_activity": cls.get_daily_activity_report(entries),
        }
