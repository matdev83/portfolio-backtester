"""
Simplified Synthetic Data Generator for Financial Time Series

This module implements a basic GARCH(1,1) model for generating synthetic data.
It is designed to be robust and predictable, avoiding the instabilities of more
complex libraries.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class GARCHParameters:
    """Container for GARCH model parameters."""
    omega: float
    alpha: float
    beta: float

@dataclass
class AssetStatistics:
    """Container for asset statistical properties."""
    mean_return: float
    volatility: float
    garch_params: GARCHParameters

class SyntheticDataGenerator:
    """
    Simplified synthetic data generator using a manual GARCH(1,1) implementation.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.random_seed = config.get('random_seed', 42)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def analyze_asset_statistics(self, data) -> AssetStatistics:
        if hasattr(data, 'empty') and data.empty or len(data) == 0:
            logger.warning("Empty data provided. Using default statistics.")
            return AssetStatistics(
                mean_return=0.0,
                volatility=1.0,
                garch_params=GARCHParameters(omega=0.0, alpha=0.0, beta=0.0)
            )
        
        if isinstance(data, pd.DataFrame):
            if 'Close' not in data.columns:
                logger.error("No 'Close' column found in OHLC data. Using default statistics.")
                return AssetStatistics(
                    mean_return=0.0,
                    volatility=1.0,
                    garch_params=GARCHParameters(omega=0.0, alpha=0.0, beta=0.0)
                )
            returns = data['Close'].pct_change(fill_method=None).dropna()
        else:
            returns = data.dropna() if hasattr(data, 'dropna') else pd.Series(data).dropna()
        
        if len(returns) < 2:
            logger.warning("Not enough data to calculate statistics. Using default statistics.")
            return AssetStatistics(
                mean_return=0.0,
                volatility=1.0,
                garch_params=GARCHParameters(omega=0.0, alpha=0.0, beta=0.0)
            )

        mean_return = returns.mean()
        volatility = returns.std()
        
        # Simplified GARCH parameter estimation
        garch_params = self._fit_garch_model(returns)

        return AssetStatistics(
            mean_return=mean_return,
            volatility=volatility,
            garch_params=garch_params
        )

    def _fit_garch_model(self, returns: pd.Series) -> GARCHParameters:
        # Simplified GARCH(1,1) parameter estimation using method of moments
        alpha = 0.1
        beta = 0.8
        omega = returns.var() * (1 - alpha - beta)
        return GARCHParameters(omega=omega, alpha=alpha, beta=beta)

    def generate_synthetic_returns(self, asset_stats: AssetStatistics, length: int, asset_name: str = "Unknown") -> np.ndarray:
        garch_params = asset_stats.garch_params
        mean_return = asset_stats.mean_return
        volatility = asset_stats.volatility

        returns = np.zeros(length)
        variances = np.zeros(length)

        variances[0] = volatility ** 2

        for t in range(1, length):
            variances[t] = garch_params.omega + garch_params.alpha * (returns[t-1] - mean_return)**2 + garch_params.beta * variances[t-1]
            returns[t] = np.random.normal(mean_return, np.sqrt(variances[t]))

        return returns

    def generate_synthetic_prices(self, ohlc_data: pd.DataFrame, length: int, asset_name: str = "Unknown") -> pd.DataFrame:
        asset_stats = self.analyze_asset_statistics(ohlc_data)
        synthetic_returns = self.generate_synthetic_returns(asset_stats, length, asset_name)
        
        initial_price = ohlc_data['Close'].iloc[-1]
        synthetic_prices = initial_price * (1 + synthetic_returns).cumprod()
        
        synthetic_ohlc = self._generate_ohlc_from_prices(synthetic_prices)
        
        synthetic_df = pd.DataFrame(
            synthetic_ohlc,
            columns=['Open', 'High', 'Low', 'Close'],
            index=pd.date_range(
                start=ohlc_data.index[-1] + pd.Timedelta(days=1),
                periods=length,
                freq='D'
            )
        )
        
        return synthetic_df

    def _generate_ohlc_from_prices(self, prices: np.ndarray) -> np.ndarray:
        ohlc = np.zeros((len(prices), 4))
        
        for i in range(len(prices)):
            close = prices[i]
            
            if i == 0:
                open_price = close
            else:
                open_price = prices[i-1]
            
            high = max(open_price, close)
            low = min(open_price, close)
            
            ohlc[i] = [open_price, high, low, close]
        
        return ohlc