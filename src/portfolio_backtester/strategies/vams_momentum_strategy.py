from typing import Set
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
from ..feature import DPVAMS, BenchmarkSMA, Feature


class VAMSMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Volatility Adjusted Momentum Scores (VAMS)."""

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "lookback_months", "alpha", "sma_filter_window"}

    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        """Specifies the features required by the VAMS momentum strategy."""
        features = set()
        params = strategy_config.get("strategy_params", {})
        
        if "lookback_months" in params and "alpha" in params:
            alpha_val = params["alpha"]
            features.add(DPVAMS(lookback_months=params["lookback_months"], alpha=f"{alpha_val:.2f}"))
        
        if "sma_filter_window" in params and params["sma_filter_window"] is not None:
            features.add(BenchmarkSMA(sma_filter_window=params["sma_filter_window"]))
            
        if "optimize" in strategy_config:
            for opt_spec in strategy_config["optimize"]:
                param_name = opt_spec["parameter"]
                min_val = opt_spec["min_value"]
                max_val = opt_spec["max_value"]
                step = opt_spec.get("step", 1)
                
                if param_name == "lookback_months":
                    for val in np.arange(min_val, max_val + step, step):
                        alpha_default_val = params.get("alpha", 0.5)
                        features.add(DPVAMS(lookback_months=int(val), alpha=f"{alpha_default_val:.2f}"))
                elif param_name == "alpha":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(DPVAMS(lookback_months=params.get("lookback_months", 6), alpha=f"{val:.2f}"))
                elif param_name == "sma_filter_window":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(BenchmarkSMA(sma_filter_window=int(val)))

        return features

    def _calculate_candidate_weights(self, look: pd.Series) -> pd.Series:
        """Calculates initial candidate weights based on VAMS."""
        if self.strategy_config.get('num_holdings'):
            num_holdings = self.strategy_config['num_holdings']
        else:
            num_holdings = max(int(np.ceil(self.strategy_config.get('top_decile_fraction', 0.1) * look.count())), 1)

        winners = look.nlargest(num_holdings).index
        losers = look.nsmallest(num_holdings).index

        cand = pd.Series(index=look.index, dtype=float).fillna(0.0)
        if len(winners) > 0:
            cand[winners] = 1 / len(winners)
        if not self.strategy_config['long_only'] and len(losers) > 0:
            cand[losers] = -1 / len(losers)
        return cand

    def _apply_leverage_and_smoothing(self, cand: pd.Series, w_prev: pd.Series) -> pd.Series:
        """Applies leverage scaling and path-dependent smoothing to weights."""
        leverage = self.strategy_config.get('leverage', 1.0)
        smoothing_lambda = self.strategy_config.get('smoothing_lambda', 0.5)

        # Apply smoothing
        w_new = smoothing_lambda * w_prev + (1 - smoothing_lambda) * cand

        # Normalize weights to maintain leverage if there are active signals
        if cand.abs().sum() > 1e-9:
            long_leverage = w_new[w_new > 0].sum()
            short_leverage = -w_new[w_new < 0].sum()

            if long_leverage > leverage:
                w_new[w_new > 0] *= leverage / long_leverage
            
            if short_leverage > leverage:
                 w_new[w_new < 0] *= leverage / short_leverage

        return w_new

    def generate_signals(self, prices: pd.DataFrame, features: dict, benchmark_data: pd.Series) -> pd.DataFrame:
        """Generates trading signals based on the VAMS momentum strategy."""
        lookback_months = self.strategy_config.get('lookback_months', 6)
        alpha_val = self.strategy_config.get('alpha', 0.5)
        dp_vams_feature_name = f"dp_vams_{lookback_months}m_{alpha_val:.2f}a"
        dp_vams_scores = features[dp_vams_feature_name]

        weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        w_prev = pd.Series(index=prices.columns, dtype=float).fillna(0.0)

        for date in prices.index:
            look = dp_vams_scores.loc[date]

            if look.count() == 0:
                weights.loc[date] = w_prev
                continue

            cand = self._calculate_candidate_weights(look)
            w_new = self._apply_leverage_and_smoothing(cand, w_prev)

            weights.loc[date] = w_new
            w_prev = w_new

        # Apply SMA filter if configured
        if self.strategy_config.get('sma_filter_window'):
            sma_window = self.strategy_config['sma_filter_window']
            sma_feature_name = f"benchmark_sma_{sma_window}m"
            risk_on = features[sma_feature_name].reindex(weights.index, fill_value=True)
            weights.loc[risk_on.index[~risk_on]] = 0.0

        return weights
