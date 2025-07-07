from typing import Set
import pandas as pd
from .features.base import Feature

def precompute_features(
    data: pd.DataFrame,
    required_features: Set[Feature],
    benchmark_data: pd.Series | None = None,
    legacy_monthly_closes: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Pre-computes all necessary features and indicators for all strategies.
    
    Args:
        data: DataFrame with prices for all assets over the entire backtest period.
               This is typically monthly OHLC data for features like ATR.
        required_features: A set of all unique features required by the strategies.
        benchmark_data: Series with benchmark prices.
        legacy_monthly_closes: DataFrame with monthly close prices for features that
                              expect only close prices (backward compatibility).

    Returns:
        A dictionary where keys are feature names and values are DataFrames of the
        pre-computed features.
    """
    features = {}
    
    # --- Perform Calculations ---
    for feature in required_features:
        # For backward compatibility, some features might need close prices only
        # If legacy_monthly_closes is provided and the feature needs it, use it
        # Otherwise, use the main data (OHLC)
        if legacy_monthly_closes is not None and hasattr(feature, 'needs_close_prices_only') and feature.needs_close_prices_only:
            features[feature.name] = feature.compute(legacy_monthly_closes, benchmark_data)
        else:
            features[feature.name] = feature.compute(data, benchmark_data)

    return features
