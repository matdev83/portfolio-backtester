import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

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
    """

    def __init__(self, global_config: dict):
        self.slippage_bps = global_config.get("slippage_bps", 2.5)
        self.commission_min_per_order = global_config.get("commission_min_per_order", 1.0)
        self.commission_per_share = global_config.get("commission_per_share", 0.005)
        self.commission_max_percent = global_config.get("commission_max_percent_of_trade", 0.005)

    def calculate(
        self,
        turnover: pd.Series,
        weights_daily: pd.DataFrame,
        price_data: pd.DataFrame,
        portfolio_value: float = 100000.0
    ) -> tuple[pd.Series, dict]:
        if isinstance(price_data.columns, pd.MultiIndex):
            daily_closes = price_data.xs('Close', level='Field', axis=1)
        else:
            daily_closes = price_data

        trade_value = turnover * portfolio_value
        weight_changes = weights_daily.diff().abs()
        shares_traded = (weight_changes * portfolio_value) / daily_closes
        commission_per_trade = shares_traded * self.commission_per_share
        commission_per_trade = np.maximum(commission_per_trade, self.commission_min_per_order)
        commission_per_trade = np.minimum(commission_per_trade, (trade_value * self.commission_max_percent).values.reshape(-1, 1))
        total_commission = commission_per_trade.sum(axis=1)
        commission_costs = total_commission / portfolio_value

        slippage_costs = turnover * (self.slippage_bps / 10000.0)

        total_costs = commission_costs + slippage_costs
        total_costs = total_costs.fillna(0)

        breakdown = {
            'commission_costs': commission_costs.fillna(0),
            'slippage_costs': slippage_costs.fillna(0),
            'total_costs': total_costs
        }

        return total_costs, breakdown


def get_transaction_cost_model(config: dict) -> TransactionCostModel:
    """
    Factory function to get the specified transaction cost model.
    """
    model_name = config.get("transaction_cost_model", "realistic").lower()
    if model_name == "realistic":
        return RealisticTransactionCostModel(config)
    else:
        raise ValueError(f"Unsupported transaction cost model: {model_name}")