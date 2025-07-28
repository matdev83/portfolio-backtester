import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

def apply_leverage_and_smoothing(candidate_weights: pd.Series, prev_weights: Optional[pd.Series], params: Optional[Dict[str, Any]] = None) -> pd.Series:
    """
    Applies leverage and exponential smoothing to candidate weights.
    params: dict with keys 'leverage', 'smoothing_lambda'
    """
    if params is None:
        params = {}
    leverage = params.get("leverage", 1.0)
    smoothing_lambda = params.get("smoothing_lambda", 0.5)

    # Apply leverage
    weights = candidate_weights * leverage

    # Apply exponential smoothing if previous weights are provided
    if prev_weights is not None:
        # Align indices
        prev_weights = prev_weights.reindex(weights.index).fillna(0.0)
        weights = smoothing_lambda * weights + (1 - smoothing_lambda) * prev_weights

    # Re-normalize if needed (e.g., for long-only)
    if params.get("long_only", True):
        total = weights.sum()
        if total != 0:
            weights = weights / total
    return weights
