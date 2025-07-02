
import pandas as pd

def equal_weight_sizer(signals: pd.DataFrame) -> pd.DataFrame:
    """Applies equal weighting to the signals."""
    sized_signals = signals.div(signals.abs().sum(axis=1), axis=0)
    return sized_signals
