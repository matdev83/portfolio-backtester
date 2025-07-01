from typing import Set
import pandas as pd
from .feature import Feature

def precompute_features(
    data: pd.DataFrame,
    required_features: Set[Feature],
    benchmark_data: pd.Series = None
) -> dict:
    """
    Pre-computes all necessary features and indicators for all strategies.
    
    Args:
        data: DataFrame with prices for all assets over the entire backtest period.
        required_features: A set of all unique features required by the strategies.
        benchmark_data: Series with benchmark prices.

    Returns:
        A dictionary where keys are feature names and values are DataFrames of the
        pre-computed features.
    """
    features = {}
    
    # --- Perform Calculations ---
    for feature in required_features:
        features[feature.name] = feature.compute(data, benchmark_data)

    return features
