"""
Simple Buy and Hold Example Strategy

This is a basic example strategy that demonstrates:
- How to inherit from PortfolioStrategy
- Simple parameter handling
- Basic signal generation
- Equal weight allocation

This strategy simply holds all assets in the universe with equal weights.
"""

from typing import Optional, Dict, Any
import pandas as pd

from ..base.portfolio_strategy import PortfolioStrategy


class SimpleBuyHoldStrategy(PortfolioStrategy):
    """
    Example strategy that holds all assets with equal weights.
    
    This is the simplest possible portfolio strategy - it allocates
    equal weights to all assets in the universe and holds them.
    """

    def __init__(self, strategy_config: dict):
        """
        Initialize the buy and hold strategy.
        
        Args:
            strategy_config: Strategy configuration dictionary
        """
        super().__init__(strategy_config)
        
        # Example of accessing strategy parameters
        self.rebalance_threshold = self.strategy_params.get(
            "simple_buy_hold.rebalance_threshold", 0.05
        )

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define tunable parameters for optimization.
        
        Returns:
            Dictionary defining parameter ranges and types
        """
        return {
            "rebalance_threshold": {
                "type": "float",
                "min": 0.01,
                "max": 0.20,
                "default": 0.05,
                "description": "Threshold for rebalancing when weights drift"
            }
        }

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate equal weight signals for all assets.
        
        Args:
            all_historical_data: Historical price data for universe
            benchmark_historical_data: Benchmark price data
            non_universe_historical_data: Additional data (not used)
            current_date: Current evaluation date
            start_date: Strategy start date
            end_date: Strategy end date
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with equal weights for all assets
        """
        if current_date is None:
            current_date = all_historical_data.index[-1]

        # Get asset names from the data
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            # MultiIndex columns: (asset, field)
            assets = all_historical_data.columns.get_level_values(0).unique()
        else:
            # Simple columns
            assets = all_historical_data.columns

        # Calculate equal weights
        num_assets = len(assets)
        if num_assets == 0:
            return pd.DataFrame(index=[current_date])
        
        equal_weight = 1.0 / num_assets
        weights = pd.Series(equal_weight, index=assets)

        return pd.DataFrame([weights], index=[current_date])

    def __str__(self):
        return f"SimpleBuyHoldStrategy(threshold={self.rebalance_threshold})"