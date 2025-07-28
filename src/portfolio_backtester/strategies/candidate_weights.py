import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

def default_candidate_weights(scores: pd.Series, params: Optional[Dict[str, Any]] = None) -> pd.Series:
    """
    Default candidate weight calculation: equal weight for top N assets by score.
    params: dict with keys 'num_holdings', 'top_decile_fraction', 'long_only'
    """
    if params is None:
        params = {}
    num_holdings = params.get("num_holdings")
    top_decile_fraction = params.get("top_decile_fraction", 0.1)
    long_only = params.get("long_only", True)
    n_assets = scores.count()
    if num_holdings is not None and num_holdings > 0:
        n_long = int(num_holdings)
    else:
        n_long = max(int(np.ceil(top_decile_fraction * n_assets)), 1) if n_assets > 0 else 0

    # Select top N assets
    top_assets = scores.sort_values(ascending=False).head(n_long).index if n_long > 0 else []
    cand = pd.Series(0.0, index=scores.index)
    if n_long == 1:
        # Always assign 1.0 to the top asset, even if score is zero, NaN, or negative
        if len(top_assets) == 1:
            cand[top_assets[0]] = 1.0
        elif len(scores) > 0:
            # Fallback: assign 1.0 to the first asset in the index (guaranteed nonzero weight)
            cand[scores.index[0]] = 1.0
    elif n_long > 0 and len(top_assets) > 0:
        if long_only:
            positive_assets = [a for a in top_assets if pd.notna(scores.at[a]) and float(scores.at[a]) > 0]
            if len(positive_assets) > 0:
                cand[positive_assets] = 1.0 / len(positive_assets)
            else:
                # If all scores are zero or NaN, assign equal weights to top N assets
                fallback_assets = [a for a in top_assets if pd.notna(scores.at[a])]
                if len(fallback_assets) > 0:
                    cand[fallback_assets] = 1.0 / len(fallback_assets)
                else:
                    cand[top_assets] = 1.0 / len(top_assets)
        else:
            nonzero_assets = [a for a in top_assets if pd.notna(scores.at[a]) and float(scores.at[a]) != 0]
            if len(nonzero_assets) > 0:
                cand[nonzero_assets] = 1.0 / len(nonzero_assets)
            else:
                fallback_assets = [a for a in top_assets if pd.notna(scores.at[a])]
                if len(fallback_assets) > 0:
                    cand[fallback_assets] = 1.0 / len(fallback_assets)
                else:
                    cand[top_assets] = 1.0 / len(top_assets)
    # Final fallback: if n_long == 1 and all weights are zero, assign 1.0 to the asset with the highest score
    if n_long == 1 and cand.sum() == 0 and len(scores) > 0:
        cand[scores.index[0]] = 1.0
    # For long/short, extend here as needed
    return cand
