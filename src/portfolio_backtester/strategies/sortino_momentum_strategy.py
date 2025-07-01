from typing import Set
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
from ..feature import SortinoRatio, BenchmarkSMA, Feature


class SortinoMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Sortino ratio for ranking."""

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "rolling_window", "sma_filter_window", "target_return"}

    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        """Specifies the features required by the Sortino momentum strategy."""
        features = set()
        params = strategy_config.get("strategy_params", {})
        
        if "rolling_window" in params:
            features.add(SortinoRatio(rolling_window=params["rolling_window"], target_return=params.get("target_return", 0.0)))
        
        if "sma_filter_window" in params and params["sma_filter_window"] is not None:
            features.add(BenchmarkSMA(sma_filter_window=params["sma_filter_window"]))
            
        if "optimize" in strategy_config:
            for opt_spec in strategy_config["optimize"]:
                param_name = opt_spec["parameter"]
                min_val, max_val, step = opt_spec["min_value"], opt_spec["max_value"], opt_spec.get("step", 1)
                
                if param_name == "rolling_window":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(SortinoRatio(rolling_window=int(val), target_return=params.get("target_return", 0.0)))
                elif param_name == "sma_filter_window":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(BenchmarkSMA(sma_filter_window=int(val)))

        return features

    def _calculate_candidate_weights(self, look: pd.Series) -> pd.Series:
        """Calculates initial candidate weights based on Sortino ratio ranking."""
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
        """Generates trading signals based on the Sortino momentum strategy."""
        rolling_window = self.strategy_config.get('rolling_window', 6)
        sortino_feature_name = f"sortino_{rolling_window}m"
        rolling_sortino = features[sortino_feature_name]

        weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        w_prev = pd.Series(index=prices.columns, dtype=float).fillna(0.0)

        for date in prices.index:
            look = rolling_sortino.loc[date]

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
