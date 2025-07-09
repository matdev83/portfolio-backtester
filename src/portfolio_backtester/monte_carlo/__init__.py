"""
Monte-Carlo Synthetic Data Generation Module

This module provides advanced Monte-Carlo capabilities for portfolio backtesting,
including synthetic data generation using GARCH models with heavy-tailed distributions.
"""

from .synthetic_data_generator import (
    SyntheticDataGenerator,
    GARCHParameters,
    AssetStatistics,
    DistributionType
)

from .asset_replacement import (
    AssetReplacementManager,
    ReplacementInfo
)

# Re-export existing monte_carlo functionality
from .monte_carlo import MonteCarloSimulator

__all__ = [
    'SyntheticDataGenerator',
    'GARCHParameters', 
    'AssetStatistics',
    'DistributionType',
    'AssetReplacementManager',
    'ReplacementInfo',
    'MonteCarloSimulator'
] 