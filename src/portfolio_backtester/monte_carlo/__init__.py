"""
Monte-Carlo Synthetic Data Generation Module

This module provides advanced Monte-Carlo capabilities for portfolio backtesting,
including synthetic data generation using GARCH models with heavy-tailed distributions.
"""

from .asset_replacement import AssetReplacementManager
from .synthetic_data_generator import (
    SyntheticDataGenerator,
    GARCHParameters as SyntheticGARCHParameters,  # Alias to avoid conflicts
    AssetStatistics as SyntheticAssetStatistics,
)

__all__ = [
    "AssetReplacementManager",
    "SyntheticDataGenerator",
    "SyntheticGARCHParameters",
    "SyntheticAssetStatistics",
]
