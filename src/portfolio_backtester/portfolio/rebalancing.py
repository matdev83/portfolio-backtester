import pandas as pd


def rebalance(weights: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Rebalances the portfolio at a given frequency."""
    # Map deprecated frequencies to their end-of-period equivalents to avoid deprecation warnings
    if frequency.upper() == "M":
        frequency = "ME"
    elif frequency.upper() == "Q":
        frequency = "QE"
    elif frequency.upper() == "Y":
        frequency = "YE"
    return weights.resample(frequency).first().ffill()
