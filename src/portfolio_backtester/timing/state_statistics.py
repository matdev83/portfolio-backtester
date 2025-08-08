"""
State statistics functionality for timing framework.
Handles calculation of portfolio summaries, position statistics, and performance metrics.
"""

import pandas as pd
from typing import Dict, Any, Optional
from .position_tracker import PositionTracker


class StateStatistics:
    """Calculates statistics and summaries from timing state data."""

    def __init__(self, position_tracker: PositionTracker):
        """
        Initialize state statistics.

        Args:
            position_tracker: Reference to the position tracker instance
        """
        self.position_tracker = position_tracker

    def get_portfolio_summary(self, last_updated: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Get summary of current portfolio state.

        Args:
            last_updated: Last update timestamp for calculations

        Returns:
            Dictionary with portfolio summary statistics
        """
        positions = self.position_tracker.get_active_positions()
        total_positions = len(positions)
        total_weight = sum(pos.current_weight for pos in positions.values())
        avg_holding_days = 0

        if last_updated and total_positions > 0:
            total_days = sum((last_updated - pos.entry_date).days for pos in positions.values())
            avg_holding_days = int(total_days / total_positions)

        return {
            "total_positions": total_positions,
            "total_weight": total_weight,
            "avg_holding_days": avg_holding_days,
            "last_updated": last_updated,
            "total_historical_positions": len(self.position_tracker.get_position_history()),
            "assets": list(positions.keys()),
        }

    def get_position_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about position history.

        Returns:
            Dictionary with position trading statistics
        """
        position_history = self.position_tracker.get_position_history()

        if not position_history:
            return {"total_trades": 0}

        holding_days = [pos["holding_days"] for pos in position_history]
        returns = [
            pos["total_return"] for pos in position_history if pos["total_return"] is not None
        ]

        stats = {
            "total_trades": len(position_history),
            "avg_holding_days": sum(holding_days) / len(holding_days) if holding_days else 0,
            "min_holding_days": min(holding_days) if holding_days else 0,
            "max_holding_days": max(holding_days) if holding_days else 0,
        }

        if returns:
            stats.update(
                {
                    "avg_return": sum(returns) / len(returns),
                    "min_return": min(returns),
                    "max_return": max(returns),
                    "positive_trades": sum(1 for r in returns if r > 0),
                    "negative_trades": sum(1 for r in returns if r < 0),
                    "win_rate": sum(1 for r in returns if r > 0) / len(returns),
                }
            )

        return stats

    def get_asset_statistics(self, asset: str) -> Dict[str, Any]:
        """
        Get statistics for a specific asset.

        Args:
            asset: Asset symbol to analyze

        Returns:
            Dictionary with asset-specific statistics
        """
        position_history = self.position_tracker.get_position_history()
        asset_history = [pos for pos in position_history if pos["asset"] == asset]

        if not asset_history:
            return {"asset": asset, "total_trades": 0}

        holding_days = [pos["holding_days"] for pos in asset_history]
        returns = [pos["total_return"] for pos in asset_history if pos["total_return"] is not None]
        weights = [pos["entry_weight"] for pos in asset_history]

        stats = {
            "asset": asset,
            "total_trades": len(asset_history),
            "avg_holding_days": sum(holding_days) / len(holding_days) if holding_days else 0,
            "avg_entry_weight": sum(weights) / len(weights) if weights else 0,
            "first_trade_date": min(pos["entry_date"] for pos in asset_history),
            "last_trade_date": max(pos["exit_date"] for pos in asset_history),
        }

        if returns:
            stats.update(
                {
                    "avg_return": sum(returns) / len(returns),
                    "best_return": max(returns),
                    "worst_return": min(returns),
                    "positive_trades": sum(1 for r in returns if r > 0),
                    "win_rate": sum(1 for r in returns if r > 0) / len(returns),
                }
            )

        # Current position info if held
        current_position = self.position_tracker.get_position_info(asset)
        if current_position:
            stats["current_position"] = {
                "entry_date": current_position.entry_date,
                "current_weight": current_position.current_weight,
                "consecutive_periods": current_position.consecutive_periods,
                "unrealized_pnl": current_position.unrealized_pnl,
            }

        return stats

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        position_history = self.position_tracker.get_position_history()
        active_positions = self.position_tracker.get_active_positions()

        if not position_history and not active_positions:
            return {"no_data": True}

        metrics = {}

        # Historical performance
        if position_history:
            returns = [
                pos["total_return"] for pos in position_history if pos["total_return"] is not None
            ]
            holding_periods = [pos["consecutive_periods"] for pos in position_history]

            if returns:
                metrics["historical"] = {
                    "total_realized_trades": len(returns),
                    "average_return": sum(returns) / len(returns),
                    "total_return": sum(returns),
                    "best_trade": max(returns),
                    "worst_trade": min(returns),
                    "win_rate": sum(1 for r in returns if r > 0) / len(returns),
                    "average_periods_held": (
                        sum(holding_periods) / len(holding_periods) if holding_periods else 0
                    ),
                }

        # Current position performance
        if active_positions:
            current_returns = [
                pos.total_return for pos in active_positions.values() if pos.total_return != 0
            ]
            current_weights = [pos.current_weight for pos in active_positions.values()]

            metrics["current"] = {
                "active_positions": len(active_positions),
                "total_weight": sum(current_weights),
                "positions_with_gains": sum(1 for r in current_returns if r > 0),
                "positions_with_losses": sum(1 for r in current_returns if r < 0),
                "unrealized_pnl": sum(pos.unrealized_pnl for pos in active_positions.values()),
            }

            if current_returns:
                metrics["current"]["average_unrealized_return"] = sum(current_returns) / len(
                    current_returns
                )

        return metrics

    def get_turnover_statistics(self) -> Dict[str, Any]:
        """
        Get portfolio turnover statistics.

        Returns:
            Dictionary with turnover metrics
        """
        position_history = self.position_tracker.get_position_history()

        if not position_history:
            return {"total_entries": 0, "total_exits": 0}

        # Group by dates to analyze turnover patterns
        entry_dates = [pos["entry_date"] for pos in position_history]
        exit_dates = [pos["exit_date"] for pos in position_history]

        # Calculate unique trading dates
        all_dates = set(entry_dates + exit_dates)

        return {
            "total_entries": len(entry_dates),
            "total_exits": len(exit_dates),
            "unique_trading_dates": len(all_dates),
            "first_trade_date": min(entry_dates) if entry_dates else None,
            "last_trade_date": max(exit_dates) if exit_dates else None,
            "average_trades_per_day": len(position_history) / len(all_dates) if all_dates else 0,
        }

    def print_statistics_summary(self, last_updated: Optional[pd.Timestamp] = None):
        """
        Print a comprehensive statistics summary.

        Args:
            last_updated: Last update timestamp for calculations
        """
        portfolio_summary = self.get_portfolio_summary(last_updated)
        position_stats = self.get_position_statistics()
        performance_metrics = self.get_performance_metrics()

        print("=" * 60)
        print("TIMING STATE STATISTICS SUMMARY")
        print("=" * 60)

        # Portfolio Summary
        print(f"Last Updated: {portfolio_summary['last_updated']}")
        print(f"Active Positions: {portfolio_summary['total_positions']}")
        print(f"Total Weight: {portfolio_summary['total_weight']:.4f}")
        print(f"Average Holding Days: {portfolio_summary['avg_holding_days']:.1f}")

        if portfolio_summary["assets"]:
            print(f"Assets: {', '.join(portfolio_summary['assets'])}")

        # Historical Statistics
        print("\nHistorical Statistics:")
        print(f"Total Trades: {position_stats['total_trades']}")
        if position_stats["total_trades"] > 0:
            print(f"Average Holding Days: {position_stats.get('avg_holding_days', 0):.1f}")
            if "win_rate" in position_stats:
                print(f"Win Rate: {position_stats['win_rate']:.2%}")
                print(f"Average Return: {position_stats['avg_return']:.2%}")

        # Performance Metrics
        if "historical" in performance_metrics:
            hist = performance_metrics["historical"]
            print("\nPerformance Metrics:")
            print(f"Total Return: {hist['total_return']:.2%}")
            print(f"Best Trade: {hist['best_trade']:.2%}")
            print(f"Worst Trade: {hist['worst_trade']:.2%}")

        if "current" in performance_metrics:
            curr = performance_metrics["current"]
            print("\nCurrent Positions:")
            print(f"Unrealized P&L: {curr['unrealized_pnl']:.4f}")
            print(f"Positions with Gains: {curr['positions_with_gains']}")
            print(f"Positions with Losses: {curr['positions_with_losses']}")

        print("=" * 60)
