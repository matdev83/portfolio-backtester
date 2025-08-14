"""
State statistics functionality for timing framework.
Provides statistical analysis and portfolio summary calculations.
"""

import pandas as pd
from typing import Dict, Optional, Any, List
from .position_tracker import PositionTracker


class StateStatistics:
    """Provides statistical analysis of timing state and positions."""

    def __init__(self, position_tracker: PositionTracker):
        """
        Initialize state statistics.

        Args:
            position_tracker: Position tracker instance to analyze
        """
        self.position_tracker = position_tracker

    def get_portfolio_summary(self, last_updated: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Get summary of current portfolio state.

        Args:
            last_updated: Last update timestamp

        Returns:
            Dictionary with portfolio summary
        """
        total_positions = self.position_tracker.get_active_positions_count()
        total_weight = sum(pos.current_weight for pos in self.position_tracker.positions.values())
        avg_holding_days = 0

        if last_updated and total_positions > 0:
            total_days = sum(
                (last_updated - pos.entry_date).days
                for pos in self.position_tracker.positions.values()
            )
            avg_holding_days = int(total_days / total_positions)

        return {
            "total_positions": total_positions,
            "total_weight": total_weight,
            "avg_holding_days": avg_holding_days,
            "last_updated": last_updated,
            "total_historical_positions": self.position_tracker.get_historical_positions_count(),
            "assets": list(self.position_tracker.positions.keys()),
        }

    def get_position_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about position history.

        Returns:
            Dictionary with position statistics
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

    def get_position_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary of positions.

        Returns:
            Dictionary with performance metrics
        """
        active_positions = self.position_tracker.positions
        position_history = self.position_tracker.get_position_history()

        # Active positions summary
        active_summary = {
            "count": len(active_positions),
            "total_weight": sum(pos.current_weight for pos in active_positions.values()),
            "avg_weight": (
                sum(pos.current_weight for pos in active_positions.values()) / len(active_positions)
                if active_positions
                else 0
            ),
            "max_weight": (
                max(pos.current_weight for pos in active_positions.values())
                if active_positions
                else 0
            ),
            "min_weight": (
                min(pos.current_weight for pos in active_positions.values())
                if active_positions
                else 0
            ),
        }

        # Historical positions summary
        returns_history = [
            pos["total_return"] for pos in position_history if pos["total_return"] is not None
        ]

        historical_summary = {
            "count": len(position_history),
            "avg_holding_days": (
                sum(pos["holding_days"] for pos in position_history) / len(position_history)
                if position_history
                else 0
            ),
            "total_return_history": returns_history,
        }

        if returns_history:
            historical_summary.update(
                {
                    "avg_return": sum(returns_history) / len(returns_history),
                    "best_return": max(returns_history),
                    "worst_return": min(returns_history),
                    "win_rate": sum(1 for r in returns_history if r > 0) / len(returns_history),
                }
            )

        return {
            "active_positions": active_summary,
            "historical_positions": historical_summary,
        }

    def get_asset_analysis(self) -> Dict[str, Any]:
        """
        Get analysis by asset across active and historical positions.

        Returns:
            Dictionary with per-asset analysis
        """
        asset_analysis: Dict[str, Any] = {}

        # Analyze active positions
        for asset, position in self.position_tracker.positions.items():
            asset_analysis[asset] = {
                "status": "active",
                "current_weight": position.current_weight,
                "entry_date": position.entry_date,
                "entry_weight": position.entry_weight,
                "max_weight": position.max_weight,
                "min_weight": position.min_weight,
                "consecutive_periods": position.consecutive_periods,
                "total_return": position.total_return,
                "unrealized_pnl": position.unrealized_pnl,
            }

        # Analyze historical positions
        position_history = self.position_tracker.get_position_history()
        for historical_pos in position_history:
            asset = historical_pos["asset"]
            if asset not in asset_analysis:
                asset_analysis[asset] = {"status": "historical", "trades": []}
            elif asset_analysis[asset]["status"] == "active":
                # Asset has both active and historical positions
                asset_analysis[asset]["historical_trades"] = []

            trade_info = {
                "entry_date": historical_pos["entry_date"],
                "exit_date": historical_pos["exit_date"],
                "holding_days": historical_pos["holding_days"],
                "entry_weight": historical_pos["entry_weight"],
                "max_weight": historical_pos["max_weight"],
                "total_return": historical_pos["total_return"],
                "final_pnl": historical_pos["final_pnl"],
            }

            if asset_analysis[asset]["status"] == "historical":
                if "trades" not in asset_analysis[asset]:
                    asset_analysis[asset]["trades"] = []
                asset_analysis[asset]["trades"].append(trade_info)
            else:
                if "historical_trades" not in asset_analysis[asset]:
                    asset_analysis[asset]["historical_trades"] = []
                asset_analysis[asset]["historical_trades"].append(trade_info)

        return asset_analysis

    def get_weight_distribution_analysis(self) -> Dict[str, Any]:
        """
        Analyze weight distribution across positions.

        Returns:
            Dictionary with weight distribution metrics
        """
        if not self.position_tracker.positions:
            return {"total_positions": 0, "total_weight": 0.0}

        weights = [pos.current_weight for pos in self.position_tracker.positions.values()]

        return {
            "total_positions": len(weights),
            "total_weight": sum(weights),
            "avg_weight": sum(weights) / len(weights),
            "max_weight": max(weights),
            "min_weight": min(weights),
            "weight_std": self._calculate_std(weights),
            "weight_distribution": {
                "heavy_positions": sum(1 for w in weights if w > 0.1),  # > 10%
                "medium_positions": sum(1 for w in weights if 0.05 <= w <= 0.1),  # 5-10%
                "light_positions": sum(1 for w in weights if w < 0.05),  # < 5%
            },
        }

    def get_holding_period_analysis(
        self, current_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, Any]:
        """
        Analyze holding periods for active and historical positions.

        Args:
            current_date: Current date for calculating active position holding periods

        Returns:
            Dictionary with holding period analysis
        """
        analysis: Dict[str, Dict[str, Any]] = {
            "active_positions": {},
            "historical_positions": {},
        }

        # Analyze active positions
        if current_date and self.position_tracker.positions:
            active_holding_days = []
            for asset, position in self.position_tracker.positions.items():
                days_held = (current_date - position.entry_date).days
                active_holding_days.append(days_held)

            if active_holding_days:
                analysis["active_positions"] = {
                    "count": len(active_holding_days),
                    "avg_days": sum(active_holding_days) / len(active_holding_days),
                    "max_days": max(active_holding_days),
                    "min_days": min(active_holding_days),
                    "std_days": self._calculate_std([float(x) for x in active_holding_days]),
                }

        # Analyze historical positions
        position_history = self.position_tracker.get_position_history()
        if position_history:
            historical_holding_days = [pos["holding_days"] for pos in position_history]

            analysis["historical_positions"] = {
                "count": len(historical_holding_days),
                "avg_days": sum(historical_holding_days) / len(historical_holding_days),
                "max_days": max(historical_holding_days),
                "min_days": min(historical_holding_days),
                "std_days": self._calculate_std([float(x) for x in historical_holding_days]),
            }

        return analysis

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return float(variance**0.5)
