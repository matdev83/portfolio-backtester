
import pandas as pd

def equal_weight_sizer(signals: pd.DataFrame) -> pd.DataFrame:
    """Applies equal weighting to the signals."""
    return signals.div(signals.abs().sum(axis=1), axis=0)
