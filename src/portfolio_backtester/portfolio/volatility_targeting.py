from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class VolatilityTargetingBase(ABC):
    """
    Abstract base class for volatility targeting mechanisms.
    """

    @abstractmethod
    def calculate_leverage_factor(
        self,
        current_raw_weights: pd.Series,
        portfolio_returns_history: pd.Series,
        current_date: pd.Timestamp,
        daily_prices: pd.DataFrame, # Added for potential future use, e.g., calculating portfolio value
        lookback_period_days: int
    ) -> float:
        """
        Calculates the desired leverage factor for the portfolio.

        Args:
            current_raw_weights (pd.Series): The raw target weights of assets before this leverage adjustment.
                                             Index is asset tickers, values are weights.
            portfolio_returns_history (pd.Series): Historical daily returns of the portfolio up to the day
                                                   *before* current_date. Index is pd.Timestamp.
            current_date (pd.Timestamp): The current date for which the leverage factor is being calculated.
                                         The factor will apply to trading based on signals/weights for this day.
            daily_prices (pd.DataFrame): DataFrame of daily prices for all assets in the universe.
                                         Index is pd.Timestamp, columns are asset tickers.
            lookback_period_days (int): The number of trading days to use for volatility calculation.

        Returns:
            float: The calculated leverage factor. E.g., 1.0 means no change, 0.5 means half exposure.
        """
        pass

class NoVolatilityTargeting(VolatilityTargetingBase):
    """
    A dummy implementation that performs no volatility targeting.
    The leverage factor will always be 1.0.
    """

    def calculate_leverage_factor(
        self,
        current_raw_weights: pd.Series,
        portfolio_returns_history: pd.Series,
        current_date: pd.Timestamp,
        daily_prices: pd.DataFrame,
        lookback_period_days: int
    ) -> float:
        """
        Returns a leverage factor of 1.0, indicating no adjustment.
        """
        return 1.0

class AnnualizedVolatilityTargeting(VolatilityTargetingBase):
    """
    Adjusts portfolio leverage to target a specific annualized volatility level.
    """
    MIN_LOOKBACK_DATA_POINTS_RATIO = 0.5 # Minimum data points relative to lookback_period_days

    def __init__(
        self,
        target_annual_volatility: float,
        lookback_period_days: int,
        max_leverage: float = 2.0,
        min_leverage: float = 0.1,
        annualization_factor: float = 252.0 # Trading days in a year
    ):
        if target_annual_volatility <= 0:
            raise ValueError("Target annual volatility must be positive.")
        if lookback_period_days <= 0:
            raise ValueError("Lookback period must be positive.")
        if max_leverage <= 0:
            raise ValueError("Max leverage must be positive.")
        if min_leverage < 0: # min_leverage can be 0
            raise ValueError("Min leverage cannot be negative.")
        if min_leverage >= max_leverage:
            raise ValueError("Min leverage must be less than max leverage.")

        self.target_annual_volatility = target_annual_volatility
        self.lookback_period_days = lookback_period_days
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.annualization_factor = annualization_factor
        self.min_data_points = int(self.lookback_period_days * self.MIN_LOOKBACK_DATA_POINTS_RATIO)
        if self.min_data_points < 5: # Ensure at least a very small number of points
            self.min_data_points = 5


    def _calculate_annualized_volatility(self, returns_series: pd.Series) -> float | None:
        """Helper to calculate annualized volatility from a returns series."""
        if returns_series is None or len(returns_series) < self.min_data_points :
            return None

        # Ensure we only use the required lookback window from the end of the series
        relevant_returns = returns_series.iloc[-self.lookback_period_days:]
        if len(relevant_returns) < self.min_data_points: # Double check after slicing
             return None

        std_dev = relevant_returns.std(ddof=1) # ddof=1 for sample standard deviation
        if pd.isna(std_dev) or std_dev < 1e-9: # Effectively zero volatility
            return 0.0

        annualized_vol = std_dev * np.sqrt(self.annualization_factor)
        return annualized_vol

    def calculate_leverage_factor(
        self,
        current_raw_weights: pd.Series,
        portfolio_returns_history: pd.Series,
        current_date: pd.Timestamp,
        daily_prices: pd.DataFrame,
        lookback_period_days: int # This argument is now part of the class state
    ) -> float:
        """
        Calculates the leverage factor to achieve target annualized volatility.
        """
        # Ensure lookback_period_days from constructor is used, not the one passed.
        # This is a bit redundant if the caller always passes self.lookback_period_days,
        # but good for clarity. The method signature requires it from VolatilityTargetingBase.

        if portfolio_returns_history.empty or len(portfolio_returns_history) < self.min_data_points:
            # Not enough data, default to 1.0 (or potentially a configured initial leverage)
            return 1.0

        # Ensure history is actually before current_date (it should be by convention)
        # And take the most recent `lookback_period_days` of data from that history
        relevant_history = portfolio_returns_history[portfolio_returns_history.index < current_date].tail(self.lookback_period_days)

        if len(relevant_history) < self.min_data_points:
            return 1.0 # Not enough data in the relevant window

        current_annual_vol = self._calculate_annualized_volatility(relevant_history)

        if current_annual_vol is None:
            # Could not calculate volatility (e.g. all NaNs, though unlikely with daily returns)
            return 1.0 # Default leverage

        if current_annual_vol < 1e-8: # Effectively zero or negligible volatility
            # If current volatility is zero, applying target_vol / current_vol would be infinity.
            # In this case, we might want to apply max_leverage if target_vol > 0,
            # or 1.0 if we are trying to de-risk. For now, let's cap at max_leverage.
            # A common practice is to hold previous leverage or revert to 1x.
            # Let's use max_leverage as an aggressive stance, assuming we want exposure.
            # Or, more conservatively, return 1.0 or previous leverage.
            # For now, returning 1.0 to be safe when vol is zero.
            return 1.0

        leverage_factor = self.target_annual_volatility / current_annual_vol

        # Apply caps
        capped_leverage_factor = max(self.min_leverage, min(leverage_factor, self.max_leverage))

        return capped_leverage_factor

    def __repr__(self):
        return (f"AnnualizedVolatilityTargeting(target_annual_volatility={self.target_annual_volatility}, "
                f"lookback_period_days={self.lookback_period_days}, max_leverage={self.max_leverage}, "
                f"min_leverage={self.min_leverage})")

# Example of how it might be configured/used (for thought process, not for this file)
# config = {
# 'name': 'annualized',
# 'target_annual_vol': 0.15, # 15%
# 'lookback_days': 60, # approx 3 months
# 'max_leverage': 2.0,
# 'min_leverage': 0.5
# }
# if config['name'] == 'annualized':
#   mechanism = AnnualizedVolatilityTargeting(
#       target_annual_volatility=config['target_annual_vol'],
#       lookback_period_days=config['lookback_days'],
#       max_leverage=config.get('max_leverage', 2.0),
#       min_leverage=config.get('min_leverage', 0.1)
#   )
# elif config['name'] == 'none':
#   mechanism = NoVolatilityTargeting()
