
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import statsmodels.api as sm

CAL_FACTOR = 12  # monthly -> annual

def calculate_metrics(rets, bench_rets, bench_ticker_name, name="Strategy"):
    """Calculates performance metrics for a given returns series."""

    def sortino_ratio(r, target=0):
        target_returns = r.fillna(0) - target
        downside_risk = np.sqrt(np.mean(np.minimum(0, target_returns)**2))
        if downside_risk == 0: return np.inf if r.mean() > 0 else 0
        return (r.mean() * CAL_FACTOR) / (downside_risk * np.sqrt(CAL_FACTOR))

    def total_ret(x): return (1 + x).prod() - 1 if len(x) > 0 else 0
    def ann(x):
        if len(x) == 0:
            return 0
        x = x.fillna(0)
        prod = (1 + x).prod()
        if prod < 0:
            return -1.0  # Total loss, annualized return is -100%
        return prod**(CAL_FACTOR / len(x)) - 1
    def ann_vol(x): return x.std() * np.sqrt(CAL_FACTOR)
    def sharpe(x): return (ann(x) / ann_vol(x)) if ann_vol(x) != 0 else 0
    def mdd(series): return (series / series.cummax() - 1).min() if not series.empty else 0

    common_index = rets.index.intersection(bench_rets.index)
    rets_aligned, bench_aligned = rets.loc[common_index], bench_rets.loc[common_index]

    X = sm.add_constant(bench_aligned)
    capm = sm.OLS(rets_aligned, X).fit()
    alpha = capm.params.get('const', 0) * CAL_FACTOR
    beta = capm.params.get(bench_ticker_name, 0)

    metrics = pd.Series({
        "Total Return": total_ret(rets),
        "Ann. Return": ann(rets),
        "Ann. Vol": ann_vol(rets),
        "Sharpe": sharpe(rets),
        "Sortino": sortino_ratio(rets),
        "Alpha (ann)": alpha,
        "Beta": beta,
        "Max DD": mdd((1 + rets).cumprod()),
        "Skew": skew(rets),
        "Kurtosis": kurtosis(rets)
    }, name=name)
    return metrics
