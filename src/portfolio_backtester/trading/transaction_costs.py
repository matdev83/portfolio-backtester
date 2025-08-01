import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from .unified_commission_calculator import get_unified_commission_calculator

logger = logging.getLogger(__name__)

class TransactionCostModel(ABC):
    """
    Abstract base class for all transaction cost models.
    """

    @abstractmethod
    def calculate(
        self,
        turnover: pd.Series,
        weights_daily: pd.DataFrame,
        price_data: pd.DataFrame,
        portfolio_value: float
    ) -> tuple[pd.Series, dict]:
        """
        Calculate transaction costs.

        Returns:
            A tuple containing the total costs as a pandas Series and a
            dictionary with a breakdown of the costs.
        """
        pass


class RealisticTransactionCostModel(TransactionCostModel):
    """
    A realistic transaction cost model based on the IBKR commission
    structure and slippage estimates.
    
    This model now uses the unified commission calculator to ensure
    consistency across all strategy types.
    """

    def __init__(self, global_config: dict):
        self.global_config = global_config
        self.unified_calculator = get_unified_commission_calculator(global_config)

    def calculate(
        self,
        turnover: pd.Series,
        weights_daily: pd.DataFrame,
        price_data: pd.DataFrame,
        portfolio_value: float = 100000.0,
        transaction_costs_bps: float = None
    ) -> tuple[pd.Series, dict]:
        """
        Calculate transaction costs using the unified commission calculator.
        
        This method maintains backward compatibility while using the new
        unified calculation system internally.
        """
        # Use the unified calculator for consistent results
        total_costs, breakdown, detailed_trade_info = self.unified_calculator.calculate_portfolio_commissions(
            turnover=turnover,
            weights_daily=weights_daily,
            price_data=price_data,
            portfolio_value=portfolio_value,
            transaction_costs_bps=transaction_costs_bps
        )
        
        # Store detailed trade info for potential client access
        self._last_detailed_trade_info = detailed_trade_info
        
        return total_costs, breakdown
    
    def get_last_detailed_trade_info(self):
        """
        Get detailed trade information from the last calculation.
        
        Returns:
            Dictionary with detailed per-trade commission information
        """
        return getattr(self, '_last_detailed_trade_info', {})


def get_transaction_cost_model(config: dict) -> TransactionCostModel:
    """
    Factory function to get the specified transaction cost model.
    """
    model_name = config.get("transaction_cost_model", "realistic").lower()
    if model_name == "realistic":
        return RealisticTransactionCostModel(config)
    else:
        raise ValueError(f"Unsupported transaction cost model: {model_name}")