import pandas as pd
import logging
from typing import Optional
from abc import ABC, abstractmethod
from .unified_commission_calculator import get_unified_commission_calculator
from ..interfaces.validator_interface import IModelValidator, create_model_validator

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
        portfolio_value: float,
        transaction_costs_bps: Optional[float] = None,
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
        transaction_costs_bps: Optional[float] = None,
    ) -> tuple[pd.Series, dict]:
        """
        Calculate transaction costs using the unified commission calculator.

        This method maintains backward compatibility while using the new
        unified calculation system internally.
        """
        # Use the unified calculator for consistent results
        total_costs, breakdown, detailed_trade_info = self.unified_calculator.calculate(
            turnover=turnover,
            weights_daily=weights_daily,
            price_data=price_data,
            portfolio_value=portfolio_value,
            transaction_costs_bps=transaction_costs_bps,
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
        return getattr(self, "_last_detailed_trade_info", {})


def get_transaction_cost_model(
    config: dict, model_validator: Optional[IModelValidator] = None
) -> TransactionCostModel:
    """
    Factory function to get the specified transaction cost model.

    Args:
        config: Configuration dictionary containing model settings
        model_validator: Injected validator for model name validation (DIP)

    Returns:
        TransactionCostModel instance

    Raises:
        ValueError: If model name is invalid (via validator interface)
    """
    # Dependency injection for model validation (DIP)
    validator = model_validator or create_model_validator()

    model_name = config.get("transaction_cost_model", "realistic").lower()

    # Validate model name using injected validator
    validation_result = validator.validate_model_name(model_name)
    if not validation_result.is_valid:
        raise ValueError(validation_result.message)

    if model_name == "realistic":
        return RealisticTransactionCostModel(config)
    else:
        # This should not happen if validator is working correctly, but keep as defensive programming
        raise ValueError(f"Unsupported transaction cost model: {model_name}")
