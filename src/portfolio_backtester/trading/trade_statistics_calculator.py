"""
Trade statistics calculator.

This module provides functionality tailored to calculating the performance
and risk metrics of trades in a portfolio over time.
"""

from typing import List, Dict, Any
from .trade_lifecycle_manager import Trade
import numpy as np


class TradeStatisticsCalculator:
    """
    Calculates detailed performance metrics for trades.

    The class operates on a list of completed trades and provides key
    performance indicators and summary metrics for investment analysis.
    """

    def calculate_statistics(
        self, completed_trades: List[Trade], initial_portfolio_value: float, allocation_mode: str
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive trade statistics.

        Args:
            completed_trades: A list of completed trades
            initial_portfolio_value: Initial capital amount
            allocation_mode: The allocation mode (e.g., reinvestment or fixed)

        Returns:
            A dictionary containing detailed trade statistics.
        """
        if not completed_trades:
            return self._get_empty_trade_stats(initial_portfolio_value, allocation_mode)

        # Calculate statistics based on direction
        completed_trades = [t for t in completed_trades if t.exit_date is not None]

        if not completed_trades:
            return self._get_empty_trade_stats(initial_portfolio_value, allocation_mode)

        # Split trades by direction
        all_trades = completed_trades
        long_trades = [t for t in completed_trades if t.quantity > 0]
        short_trades = [t for t in completed_trades if t.quantity < 0]

        # Calculate stats for each direction
        all_stats = self._calculate_direction_stats(all_trades, "all")
        long_stats = self._calculate_direction_stats(long_trades, "long")
        short_stats = self._calculate_direction_stats(short_trades, "short")

        # Combine stats
        combined_stats = {}

        # Add directional statistics with prefixes
        for direction, stats in [("all", all_stats), ("long", long_stats), ("short", short_stats)]:
            for key, value in stats.items():
                combined_stats[f"{direction}_{key}"] = value

        return combined_stats

    def _calculate_direction_stats(self, trades: List[Trade], direction: str) -> Dict[str, Any]:
        """Calculate statistics for a specific trade direction."""
        if not trades:
            return self._get_empty_direction_stats()

        # Basic trade counts
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]
        num_winners = len(winning_trades)
        num_losers = len(losing_trades)
        win_rate = (num_winners / total_trades * 100) if total_trades > 0 else 0.0

        # P&L statistics
        pnl_values = [t.pnl_net for t in trades if t.pnl_net is not None]
        total_pnl_net = sum(pnl_values)
        total_commissions = sum(t.commission_entry + t.commission_exit for t in trades)

        # New metrics: Largest single trade profit/loss
        winning_pnls = [t.pnl_net for t in winning_trades if t.pnl_net is not None]
        losing_pnls = [t.pnl_net for t in losing_trades if t.pnl_net is not None]

        largest_profit = max(winning_pnls) if winning_pnls else 0.0
        largest_loss = min(losing_pnls) if losing_pnls else 0.0  # Most negative

        # Mean single trade profit/loss
        mean_profit = np.mean(winning_pnls) if winning_pnls else 0.0
        mean_loss = np.mean(losing_pnls) if losing_pnls else 0.0
        mean_trade_pnl = np.mean(pnl_values) if pnl_values else 0.0

        # Reward/Risk ratio (average win / average loss)
        reward_risk_ratio = (
            abs(mean_profit / mean_loss) if mean_loss != 0 else np.inf if mean_profit > 0 else 0.0
        )

        # Duration statistics
        durations = [t.duration_days for t in trades if t.duration_days is not None]
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        mean_duration = np.mean(durations) if durations else 0

        # MFE/MAE statistics
        mfe_values = [t.mfe for t in trades]
        mae_values = [t.mae for t in trades]
        avg_mfe = np.mean(mfe_values) if mfe_values else 0.0
        avg_mae = np.mean(mae_values) if mae_values else 0.0

        # Calculate trades per month
        if trades:
            start_date = min(t.entry_date for t in trades)
            end_date = max(t.exit_date for t in trades if t.exit_date)
            total_months = ((end_date - start_date).days / 30.44) if end_date else 1
            trades_per_month = total_trades / total_months if total_months > 0 else 0
        else:
            trades_per_month = 0

        # Information Score (simplified)
        returns = [
            (t.pnl_net or 0.0) / t.entry_value
            for t in trades
            if (t.entry_value is not None and t.entry_value > 0)
        ]
        information_score = (
            (np.mean(returns) / np.std(returns))
            if len(returns) > 1 and np.std(returns) > 0
            else 0.0
        )

        return {
            "num_trades": total_trades,
            "num_winners": num_winners,
            "num_losers": num_losers,
            "win_rate_pct": win_rate,
            "total_commissions_paid": total_commissions,
            "total_pnl_net": total_pnl_net,
            "largest_profit": largest_profit,
            "largest_loss": largest_loss,
            "mean_profit": mean_profit,
            "mean_loss": mean_loss,
            "mean_trade_pnl": mean_trade_pnl,
            "reward_risk_ratio": reward_risk_ratio,
            "avg_mfe": avg_mfe,
            "avg_mae": avg_mae,
            "min_trade_duration_days": min_duration,
            "max_trade_duration_days": max_duration,
            "mean_trade_duration_days": mean_duration,
            "information_score": information_score,
            "trades_per_month": trades_per_month,
        }

    def _get_empty_direction_stats(self) -> Dict[str, Any]:
        """Return empty trade statistics for a specific direction."""
        return {
            "num_trades": 0,
            "num_winners": 0,
            "num_losers": 0,
            "win_rate_pct": 0.0,
            "total_commissions_paid": 0.0,
            "total_pnl_net": 0.0,
            "largest_profit": 0.0,
            "largest_loss": 0.0,
            "mean_profit": 0.0,
            "mean_loss": 0.0,
            "mean_trade_pnl": 0.0,
            "reward_risk_ratio": 0.0,
            "avg_mfe": 0.0,
            "avg_mae": 0.0,
            "min_trade_duration_days": 0,
            "max_trade_duration_days": 0,
            "mean_trade_duration_days": 0.0,
            "information_score": 0.0,
            "trades_per_month": 0.0,
        }

    def _get_empty_trade_stats(self, initial_portfolio_value, allocation_mode) -> Dict[str, Any]:
        """Return empty trade statistics for cases with no trades."""
        empty_direction = self._get_empty_direction_stats()

        # Create empty stats for all directions
        combined_stats = {}
        for direction in ["all", "long", "short"]:
            for key, value in empty_direction.items():
                combined_stats[f"{direction}_{key}"] = value

        # Add portfolio-level statistics
        combined_stats.update(
            {
                "max_margin_load": 0.0,
                "mean_margin_load": 0.0,
                "allocation_mode": allocation_mode,
                "initial_capital": initial_portfolio_value,
                "final_capital": initial_portfolio_value,
                "total_return_pct": 0.0,
                "capital_growth_factor": 1.0,
            }
        )

        return combined_stats
