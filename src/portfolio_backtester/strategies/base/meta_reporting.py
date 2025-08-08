"""Meta strategy specific reporting and analytics."""

from __future__ import annotations

import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

from .trade_aggregator import TradeAggregator

logger = logging.getLogger(__name__)


class MetaStrategyReporter:
    """
    Comprehensive reporting system for meta strategies.

    This class generates detailed reports showing:
    - Trade statistics as if meta strategy executed all trades
    - Attribution analysis by sub-strategy
    - Capital utilization metrics
    - Performance breakdown
    """

    def __init__(self, trade_aggregator: TradeAggregator):
        """
        Initialize the meta strategy reporter.

        Args:
            trade_aggregator: TradeAggregator instance with trade data
        """
        self.trade_aggregator = trade_aggregator

    def generate_trade_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive trade statistics for the meta strategy.

        Returns:
            Dictionary containing detailed trade statistics
        """
        all_trades = self.trade_aggregator.get_aggregated_trades()

        if not all_trades:
            return {
                "total_trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_trade_size": 0.0,
                "total_transaction_costs": 0.0,
                "trading_period": None,
            }

        # Basic counts
        total_trades = len(all_trades)
        buy_trades = len([t for t in all_trades if t.is_buy])
        sell_trades = len([t for t in all_trades if t.is_sell])

        # Trade sizes and costs
        trade_values = [t.trade_value for t in all_trades if t.trade_value is not None]
        transaction_costs = [t.transaction_cost for t in all_trades]

        avg_trade_size = np.mean(trade_values) if trade_values else 0.0
        total_transaction_costs = sum(transaction_costs)

        # Trading period
        dates = [t.date for t in all_trades]
        trading_period = {
            "start": min(dates),
            "end": max(dates),
            "days": (max(dates) - min(dates)).days + 1,
        }

        # Performance metrics
        performance = self.trade_aggregator.calculate_weighted_performance()

        return {
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "avg_trade_size": avg_trade_size,
            "total_transaction_costs": total_transaction_costs,
            "trading_period": trading_period,
            "performance_summary": performance,
        }

    def generate_attribution_analysis(self) -> Dict[str, Any]:
        """
        Generate detailed attribution analysis by sub-strategy.

        Returns:
            Dictionary containing attribution analysis
        """
        attribution = self.trade_aggregator.get_strategy_attribution()
        all_trades = self.trade_aggregator.get_aggregated_trades()

        if not all_trades:
            return {"strategies": {}, "summary": {}}

        # Enhanced attribution with additional metrics
        enhanced_attribution = {}

        for strategy_id, basic_stats in attribution.items():
            strategy_trades = self.trade_aggregator.get_trades_by_strategy(strategy_id)

            if not strategy_trades:
                continue

            # Calculate additional metrics
            trade_values = [t.trade_value for t in strategy_trades if t.trade_value is not None]
            avg_trade_size = np.mean(trade_values) if trade_values else 0.0
            # Asset diversification
            assets_traded = set(t.asset for t in strategy_trades)

            # Trading frequency
            dates = [t.date for t in strategy_trades]
            trading_days = len(set(dates))

            enhanced_attribution[strategy_id] = {
                **basic_stats,
                "avg_trade_size": avg_trade_size,
                "assets_traded": list(assets_traded),
                "num_assets": len(assets_traded),
                "trading_days": trading_days,
                "trades_per_day": len(strategy_trades) / max(trading_days, 1),
            }

        # Summary statistics
        total_trades = len(all_trades)
        summary = {
            "total_strategies": len(enhanced_attribution),
            "total_trades": total_trades,
            "strategy_trade_distribution": {
                strategy_id: len(self.trade_aggregator.get_trades_by_strategy(strategy_id))
                for strategy_id in enhanced_attribution.keys()
            },
        }

        return {"strategies": enhanced_attribution, "summary": summary}

    def generate_capital_utilization_report(self) -> Dict[str, Any]:
        """
        Generate capital utilization metrics.

        Returns:
            Dictionary containing capital utilization analysis
        """
        all_trades = self.trade_aggregator.get_aggregated_trades()

        if not all_trades:
            return {
                "total_capital_deployed": 0.0,
                "avg_capital_utilization": 0.0,
                "peak_capital_usage": 0.0,
                "strategy_capital_breakdown": {},
            }

        # Calculate capital deployment by strategy
        strategy_capital = {}
        for trade in all_trades:
            if trade.strategy_id not in strategy_capital:
                strategy_capital[trade.strategy_id] = {
                    "allocated_capital": trade.allocated_capital,
                    "total_deployed": 0.0,
                    "trades": 0,
                }

            if trade.trade_value is not None:
                strategy_capital[trade.strategy_id]["total_deployed"] += trade.trade_value
            strategy_capital[trade.strategy_id]["trades"] += 1

        # Calculate utilization metrics
        total_allocated = sum(info["allocated_capital"] for info in strategy_capital.values())
        total_deployed = sum(info["total_deployed"] for info in strategy_capital.values())

        avg_utilization = (total_deployed / total_allocated) if total_allocated > 0 else 0.0

        # Strategy-level utilization
        strategy_utilization = {}
        for strategy_id, info in strategy_capital.items():
            utilization = (
                (info["total_deployed"] / info["allocated_capital"])
                if info["allocated_capital"] > 0
                else 0.0
            )
            strategy_utilization[strategy_id] = {
                "allocated_capital": info["allocated_capital"],
                "total_deployed": info["total_deployed"],
                "utilization_ratio": utilization,
                "trades": info["trades"],
            }

        return {
            "total_allocated_capital": total_allocated,
            "total_capital_deployed": total_deployed,
            "avg_capital_utilization": avg_utilization,
            "strategy_capital_breakdown": strategy_utilization,
        }

    def generate_performance_breakdown(self) -> Dict[str, Any]:
        """
        Generate detailed performance breakdown.

        Returns:
            Dictionary containing performance breakdown
        """
        performance = self.trade_aggregator.calculate_weighted_performance()
        timeline = self.trade_aggregator.get_portfolio_timeline()

        breakdown = {"overall_performance": performance, "timeline_available": not timeline.empty}

        if not timeline.empty:
            # Calculate additional performance metrics
            returns = timeline["returns"].dropna()

            if not returns.empty:
                # Risk metrics
                volatility = returns.std() * np.sqrt(252)  # Annualized
                downside_returns = returns[returns < 0]
                downside_volatility = (
                    downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
                )

                # Return distribution
                return_percentiles = returns.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()

                breakdown.update(
                    {
                        "risk_metrics": {
                            "volatility": volatility,
                            "downside_volatility": downside_volatility,
                            "positive_days": len(returns[returns > 0]),
                            "negative_days": len(returns[returns < 0]),
                            "flat_days": len(returns[returns == 0]),
                        },
                        "return_distribution": {
                            f"percentile_{int(p*100)}": v for p, v in return_percentiles.items()
                        },
                    }
                )

        return breakdown

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive meta strategy report.

        Returns:
            Dictionary containing all report sections
        """
        return {
            "trade_statistics": self.generate_trade_statistics(),
            "attribution_analysis": self.generate_attribution_analysis(),
            "capital_utilization": self.generate_capital_utilization_report(),
            "performance_breakdown": self.generate_performance_breakdown(),
            "report_metadata": {
                "generated_at": pd.Timestamp.now(),
                "total_trades_analyzed": len(self.trade_aggregator.get_aggregated_trades()),
                "strategies_analyzed": len(self.trade_aggregator.get_strategy_attribution()),
            },
        }

    def export_trades_for_framework(self) -> pd.DataFrame:
        """
        Export trades in a format compatible with the backtesting framework.

        This method formats the aggregated trades as if they were executed
        by a single strategy (the meta strategy) for framework compatibility.

        Returns:
            DataFrame formatted for framework consumption
        """
        trades_df = self.trade_aggregator.export_trades_to_dataframe()

        if trades_df.empty:
            return pd.DataFrame()

        # Rename columns to match framework expectations
        framework_df = trades_df.copy()

        # Add framework-specific columns if needed
        framework_df["meta_strategy_trade"] = True
        framework_df["original_strategy"] = framework_df["strategy_id"]

        # Sort by date for chronological order
        framework_df = framework_df.sort_values("date").reset_index(drop=True)

        return framework_df

    def generate_trade_summary_for_framework(self) -> Dict[str, Any]:
        """
        Generate trade summary in framework-expected format.

        Returns:
            Dictionary with trade summary for framework reporting
        """
        trade_stats = self.generate_trade_statistics()
        performance = self.trade_aggregator.calculate_weighted_performance()

        # Format for framework compatibility
        framework_summary = {
            "Number of Trades": trade_stats["total_trades"],
            "Number of Winners": trade_stats["buy_trades"],  # Simplified
            "Number of Losers": trade_stats["sell_trades"],  # Simplified
            "Win Rate (%)": 0.0,  # Would need P&L calculation per trade
            "Total P&L Net": performance["total_pnl"],
            "Mean Trade P&L": performance["total_pnl"] / max(trade_stats["total_trades"], 1),
            "Commissions Paid": trade_stats["total_transaction_costs"],
            "Current Portfolio Value": performance["current_value"],
            "Cash Balance": performance["cash_balance"],
            "Position Value": performance["position_value"],
        }

        return framework_summary

    def get_strategy_comparison(self) -> pd.DataFrame:
        """
        Generate comparison table of sub-strategies.

        Returns:
            DataFrame comparing sub-strategy performance
        """
        attribution = self.generate_attribution_analysis()

        if not attribution["strategies"]:
            return pd.DataFrame()

        comparison_data = []

        for strategy_id, stats in attribution["strategies"].items():
            comparison_data.append(
                {
                    "Strategy": strategy_id,
                    "Total Trades": stats["total_trades"],
                    "Buy Trades": stats["buy_trades"],
                    "Sell Trades": stats["sell_trades"],
                    "Total Trade Value": stats["total_trade_value"],
                    "Avg Trade Size": stats["avg_trade_size"],
                    "Assets Traded": stats["num_assets"],
                    "Trading Days": stats["trading_days"],
                    "Transaction Costs": stats["total_transaction_costs"],
                }
            )

        df = pd.DataFrame(comparison_data)
        df = df.set_index("Strategy")

        return df
