from typing import Dict, Any
import pandas as pd
import numpy as np
from .fixed_weight_portfolio_strategy import FixedWeightPortfolioStrategy


class VolatilityTargetedFixedWeightPortfolioStrategy(FixedWeightPortfolioStrategy):
    """
    Extends the FixedWeightPortfolioStrategy to include volatility targeting.
    This strategy first determines the base fixed weights and then scales the entire
    portfolio to meet a specified volatility target.
    """

    def __init__(self, strategy_params: Dict[str, Any]):
        super().__init__(strategy_params)
        self.target_volatility = self.strategy_params.get("target_volatility", 0.15)
        self.vol_window = self.strategy_params.get(
            "vol_window", 126
        )  # Default to 6 months (approx. 126 trading days)

    def generate_signals(self, all_historical_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generates weights by first getting the base fixed allocation and then applying
        a volatility scaling factor to the entire portfolio.
        """
        # Get the base fixed weights from the parent class
        base_weights_df = super().generate_signals(all_historical_data, **kwargs)
        if base_weights_df.empty:
            return base_weights_df

        base_weights = base_weights_df.iloc[0]

        # We need daily returns to calculate volatility
        # The framework provides daily OHLC data in `all_historical_data`
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            daily_prices = all_historical_data.xs("Close", level="Field", axis=1)
        else:
            daily_prices = all_historical_data

        daily_returns = daily_prices.pct_change().fillna(0)

        # Calculate the historical returns of the base fixed-weight portfolio
        portfolio_returns = (daily_returns[base_weights.index] * base_weights).sum()

        # Calculate the rolling annualized volatility of the portfolio
        rolling_vol = portfolio_returns.rolling(window=self.vol_window).std() * np.sqrt(252)

        # Determine the scaling factor
        # Use the most recent volatility calculation
        latest_vol = rolling_vol.iloc[-1]
        if latest_vol > 0:
            scaling_factor = self.target_volatility / latest_vol
        else:
            scaling_factor = 1.0  # Default to 1.0 if volatility is zero

        # Scale the base weights by the leverage factor
        final_weights = base_weights * scaling_factor

        return pd.DataFrame([final_weights])

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Defines the set of parameters that can be optimized for this strategy.
        """
        return {
            "target_volatility": {
                "type": "float",
                "default": 0.15,
                "min": 0.05,
                "max": 0.30,
                "step": 0.01,
            },
            "vol_window": {
                "type": "int",
                "default": 126,
                "min": 21,
                "max": 252,
                "step": 21,
            },
        }
