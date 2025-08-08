"""
Portfolio commission calculator for portfolio-level commission calculations.

This module provides portfolio-level commission calculations that handle
weight changes, turnover calculations, and portfolio-wide cost aggregation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable

from .trade_commission_info import TradeCommissionInfo
from ..interfaces.price_data_extractor_interface import PriceDataExtractorFactory
from ..interfaces.turnover_normalizer_interface import TurnoverNormalizerFactory


class PortfolioCommissionCalculator:
    """
    Calculator for portfolio-level commission calculations.

    This class handles complex portfolio commission calculations including
    weight changes, turnover processing, and portfolio-wide cost aggregation.
    """

    def __init__(self, global_config: Dict[str, Any]):
        """
        Initialize the portfolio commission calculator.

        Args:
            global_config: Global configuration containing commission parameters
        """
        self.global_config = global_config
        # Slippage parameters for detailed calculations
        self.slippage_bps = global_config.get("slippage_bps", 2.5)

    def calculate_portfolio_commissions(
        self,
        turnover: pd.Series,
        weights_daily: pd.DataFrame,
        price_data: pd.DataFrame,
        portfolio_value: float,
        trade_commission_calculator: Callable[..., TradeCommissionInfo],
        transaction_costs_bps: Optional[float] = None,
    ) -> Tuple[pd.Series, Dict[str, Any], Dict[pd.Timestamp, Dict[str, TradeCommissionInfo]]]:
        """
        Calculate commissions for portfolio-based strategies.

        Args:
            turnover: Daily turnover series
            weights_daily: Daily portfolio weights DataFrame
            price_data: Price data DataFrame
            portfolio_value: Portfolio value in dollars
            trade_commission_calculator: Function to calculate individual trade commissions
            transaction_costs_bps: Optional override for transaction costs

        Returns:
            Tuple of (total_costs_series, breakdown_dict, detailed_trade_info)
        """
        # Extract close prices using polymorphic interface
        price_extractor = PriceDataExtractorFactory.create_extractor(price_data)
        daily_closes = price_extractor.extract_close_prices(price_data)

        # Calculate weight changes; first day should not incur costs
        weight_changes = weights_daily.diff().abs()
        if not weight_changes.empty:
            weight_changes.iloc[0] = 0.0
        weight_changes = weight_changes.fillna(0.0)

        # Align columns
        aligned_weights, aligned_closes = weight_changes.align(
            daily_closes, join="left", axis=1, fill_value=0.0
        )

        # Initialize results
        total_costs = pd.Series(0.0, index=turnover.index)
        commission_costs = pd.Series(0.0, index=turnover.index)
        slippage_costs = pd.Series(0.0, index=turnover.index)
        detailed_trade_info: Dict[pd.Timestamp, Dict[str, TradeCommissionInfo]] = {}

        # Calculate costs for each date
        first_date = aligned_weights.index[0] if not aligned_weights.empty else None
        for date in turnover.index:
            if date not in aligned_weights.index or date not in aligned_closes.index:
                continue

            date_weights = aligned_weights.loc[date]
            date_prices = aligned_closes.loc[date]
            date_trade_info: Dict[str, TradeCommissionInfo] = {}

            date_commission_total = 0.0
            date_slippage_total = 0.0

            # Calculate costs for each asset with non-zero weight change
            for asset in date_weights.index:
                weight_change = date_weights[asset]
                if abs(weight_change) < 1e-8:
                    continue
                # Skip first day entirely (no prior position)
                if first_date is not None and date == first_date:
                    continue

                price = date_prices[asset] if asset in date_prices.index else 0.0
                if price <= 0 or pd.isna(price):
                    continue

                # Calculate trade details
                trade_value = weight_change * portfolio_value
                quantity = trade_value / price

                # Calculate commission for this trade
                commission_info = trade_commission_calculator(
                    asset=asset,
                    date=date,
                    quantity=quantity,
                    price=price,
                    transaction_costs_bps=transaction_costs_bps,
                )

                date_trade_info[asset] = commission_info
                date_commission_total += commission_info.commission_amount

                # For detailed calculation, add slippage per trade
                if transaction_costs_bps is None:
                    date_slippage_total += commission_info.slippage_amount

            # For detailed calculation, if no individual trades but turnover exists, apply slippage
            if transaction_costs_bps is None:
                turnover_value = turnover.loc[date] if date in turnover.index else 0.0
                # Use polymorphic turnover normalization
                turnover_normalizer = TurnoverNormalizerFactory.create_normalizer(turnover_value)
                turnover_value = turnover_normalizer.normalize_turnover_value(turnover_value)
                if (first_date is not None and date == first_date) or (
                    aligned_weights.loc[date].abs().sum() == 0.0
                ):
                    turnover_value = 0.0
                if turnover_value > 0 and date_slippage_total == 0:
                    date_slippage_total = (
                        portfolio_value * turnover_value * (self.slippage_bps / 10000.0)
                    )

            # Store results
            detailed_trade_info[date] = date_trade_info
            commission_costs.loc[date] = date_commission_total / portfolio_value
            slippage_costs.loc[date] = date_slippage_total / portfolio_value
            total_costs.loc[date] = commission_costs.loc[date] + slippage_costs.loc[date]

        # Create breakdown dictionary for backward compatibility
        breakdown = {
            "commission_costs": commission_costs.fillna(0),
            "slippage_costs": slippage_costs.fillna(0),
            "total_costs": total_costs.fillna(0),
        }

        return total_costs.fillna(0), breakdown, detailed_trade_info

    def get_commission_summary(
        self, detailed_trade_info: Dict[pd.Timestamp, Dict[str, TradeCommissionInfo]]
    ) -> Dict[str, Any]:
        """
        Generate a summary of commission costs from detailed trade information.

        Args:
            detailed_trade_info: Detailed trade commission information

        Returns:
            Dictionary with commission summary statistics
        """
        if not detailed_trade_info:
            return {
                "total_trades": 0,
                "total_commission": 0.0,
                "total_slippage": 0.0,
                "total_costs": 0.0,
                "avg_commission_per_trade": 0.0,
                "avg_slippage_per_trade": 0.0,
                "avg_cost_per_trade": 0.0,
                "commission_rate_bps_avg": 0.0,
                "slippage_rate_bps_avg": 0.0,
            }

        all_trades: list[TradeCommissionInfo] = []
        for date_trades in detailed_trade_info.values():
            all_trades.extend(date_trades.values())

        if not all_trades:
            return {
                "total_trades": 0,
                "total_commission": 0.0,
                "total_slippage": 0.0,
                "total_costs": 0.0,
                "avg_commission_per_trade": 0.0,
                "avg_slippage_per_trade": 0.0,
                "avg_cost_per_trade": 0.0,
                "commission_rate_bps_avg": 0.0,
                "slippage_rate_bps_avg": 0.0,
            }

        total_commission = sum(trade.commission_amount for trade in all_trades)
        total_slippage = sum(trade.slippage_amount for trade in all_trades)
        total_costs = sum(trade.total_cost for trade in all_trades)

        return {
            "total_trades": len(all_trades),
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "total_costs": total_costs,
            "avg_commission_per_trade": total_commission / len(all_trades),
            "avg_slippage_per_trade": total_slippage / len(all_trades),
            "avg_cost_per_trade": total_costs / len(all_trades),
            "commission_rate_bps_avg": np.mean([trade.commission_rate_bps for trade in all_trades]),
            "slippage_rate_bps_avg": np.mean([trade.slippage_rate_bps for trade in all_trades]),
        }
