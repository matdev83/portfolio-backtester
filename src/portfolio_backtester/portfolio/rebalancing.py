import pandas as pd


def _normalize_resample_freq(frequency: str) -> str:
    u = frequency.upper()
    if u == "M":
        return "ME"
    if u == "Q":
        return "QE"
    if u == "Y":
        return "YE"
    return frequency


def rebalance(weights: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Rebalances the portfolio at a given frequency."""
    frequency = _normalize_resample_freq(frequency)
    return weights.resample(frequency).first().ffill()


def rebalance_to_first_event_per_period(weights: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """First observation per resample period, preserving each row's actual event timestamp.

    Unlike :func:`rebalance`, this does not rewrite row labels to period-end bins. That keeps
    sparse targets on real decision dates so downstream execution timing and shifted-return
    weighting stay aligned with the original calendar.
    """
    if weights.empty:
        return weights.copy()
    frequency = _normalize_resample_freq(frequency)
    sorted_w = weights.sort_index()
    rows: list[pd.Series] = []
    index_list: list[pd.Timestamp] = []
    for _, grp in sorted_w.groupby(pd.Grouper(freq=frequency)):
        if grp.empty:
            continue
        rows.append(grp.iloc[0])
        index_list.append(pd.Timestamp(grp.index[0]))
    if not rows:
        tz = getattr(weights.index, "tz", None)
        return pd.DataFrame(columns=list(weights.columns), index=pd.DatetimeIndex([], tz=tz))
    out = pd.DataFrame(
        rows, index=pd.DatetimeIndex(index_list, tz=getattr(weights.index, "tz", None))
    )
    return out.sort_index().ffill()
