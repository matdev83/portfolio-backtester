
import pandas as pd

def rebalance(weights: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Rebalances the portfolio at a given frequency."""
    # Ensure 'M' is mapped to 'ME' for month-end frequency to avoid deprecation warning
    if frequency.upper() == 'M':
        frequency = 'ME'
    return weights.resample(frequency).first().ffill()
