"""
Trade statistics table formatter.

This module provides functionality to format trade statistics
into readable tables and summary views.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class TradeTableFormatter:
    """
    Formats trade statistics into tables and summary views.

    This class handles the presentation layer of trade statistics,
    converting raw statistics into formatted tables and summaries.
    """

    def format_statistics_table(self, stats: Dict[str, Any]) -> pd.DataFrame:
        """
        Get trade statistics formatted as a table with All/Long/Short columns.

        Args:
            stats: Dictionary of trade statistics

        Returns:
            DataFrame with formatted trade statistics
        """
        if not stats or stats.get("all_num_trades", 0) == 0:
            return pd.DataFrame()

        # Define the metrics we want to display
        metrics_config = [
            ("num_trades", "Number of Trades", "int"),
            ("num_winners", "Number of Winners", "int"),
            ("num_losers", "Number of Losers", "int"),
            ("win_rate_pct", "Win Rate (%)", "pct"),
            ("total_pnl_net", "Total P&L Net", "currency"),
            ("largest_profit", "Largest Single Profit", "currency"),
            ("largest_loss", "Largest Single Loss", "currency"),
            ("mean_profit", "Mean Profit", "currency"),
            ("mean_loss", "Mean Loss", "currency"),
            ("mean_trade_pnl", "Mean Trade P&L", "currency"),
            ("reward_risk_ratio", "Reward/Risk Ratio", "ratio"),
            ("total_commissions_paid", "Commissions Paid", "currency"),
            ("avg_mfe", "Avg MFE", "currency"),
            ("avg_mae", "Avg MAE", "currency"),
            ("min_trade_duration_days", "Min Duration (days)", "int"),
            ("max_trade_duration_days", "Max Duration (days)", "int"),
            ("mean_trade_duration_days", "Mean Duration (days)", "float"),
            ("information_score", "Information Score", "ratio"),
            ("trades_per_month", "Trades per Month", "float"),
        ]

        # Build the table data
        table_data = []

        for metric_key, metric_name, format_type in metrics_config:
            row = {"Metric": metric_name}

            for direction in ["all", "long", "short"]:
                direction_title = direction.title()
                value = stats.get(f"{direction}_{metric_key}", 0)

                # Format the value based on type
                formatted_value = self._format_value(value, format_type)
                row[direction_title] = formatted_value

            table_data.append(row)

        # Add portfolio-level metrics
        portfolio_metrics = [
            ("max_margin_load", "Max Margin Load", "pct"),
            ("mean_margin_load", "Mean Margin Load", "pct"),
            ("allocation_mode", "Allocation Mode", "string"),
            ("initial_capital", "Initial Capital", "currency"),
            ("final_capital", "Final Capital", "currency"),
            ("total_return_pct", "Total Return (%)", "pct"),
            ("capital_growth_factor", "Capital Growth Factor", "ratio"),
        ]

        for metric_key, metric_name, format_type in portfolio_metrics:
            row = {"Metric": metric_name}
            value = stats.get(metric_key, 0)

            if format_type == "pct" and metric_key not in ["total_return_pct"]:
                formatted_value = f"{value * 100:.2f}%"
            else:
                formatted_value = self._format_value(value, format_type)

            # Portfolio metrics apply to all directions
            for direction in ["All", "Long", "Short"]:
                row[direction] = formatted_value

            table_data.append(row)

        return pd.DataFrame(table_data)

    def format_directional_summary(self, stats: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of key metrics by direction for easy comparison.

        Args:
            stats: Dictionary of trade statistics

        Returns:
            Dictionary with directional summary data
        """
        summary = {}
        for direction in ["all", "long", "short"]:
            prefix = f"{direction}_"
            summary[direction.title()] = {
                "trades": stats.get(f"{prefix}num_trades", 0),
                "win_rate": stats.get(f"{prefix}win_rate_pct", 0.0),
                "total_pnl": stats.get(f"{prefix}total_pnl_net", 0.0),
                "avg_profit": stats.get(f"{prefix}mean_profit", 0.0),
                "avg_loss": stats.get(f"{prefix}mean_loss", 0.0),
                "reward_risk": stats.get(f"{prefix}reward_risk_ratio", 0.0),
                "largest_win": stats.get(f"{prefix}largest_profit", 0.0),
                "largest_loss": stats.get(f"{prefix}largest_loss", 0.0),
            }

        return summary

    def _format_value(self, value: Any, format_type: str) -> str:
        """
        Format a value according to the specified format type.

        Args:
            value: The value to format
            format_type: The formatting type ('int', 'pct', 'currency', etc.)

        Returns:
            Formatted string representation of the value
        """
        if format_type == "int":
            return f"{int(value)}"
        elif format_type == "pct":
            return f"{value:.2f}%"
        elif format_type == "currency":
            return f"${value:,.2f}"
        elif format_type == "ratio":
            if value == np.inf:
                return "∞"
            else:
                return f"{value:.3f}"
        elif format_type == "float":
            return f"{value:.2f}"
        elif format_type == "string":
            return str(value)
        else:
            return str(value)
