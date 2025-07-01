from typing import Set
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
from ..feature import Momentum, BenchmarkSMA, Feature


class MomentumStrategy(BaseStrategy):
    """Momentum strategy implementation."""

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {
            "lookback_months", "num_holdings", "top_decile_fraction",
            "smoothing_lambda", "leverage", "long_only", "sma_filter_window",
            "derisk_days_under_sma",
        }

    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        """Specifies the features required by the momentum strategy."""
        features = set()
        params = strategy_config.get("strategy_params", {})
        
        if "lookback_months" in params:
            features.add(Momentum(lookback_months=params["lookback_months"]))
        
        if "sma_filter_window" in params and params["sma_filter_window"] is not None:
            features.add(BenchmarkSMA(sma_filter_window=params["sma_filter_window"]))
            
        if "optimize" in strategy_config:
            for opt_spec in strategy_config["optimize"]:
                param_name = opt_spec["parameter"]
                min_val, max_val, step = opt_spec["min_value"], opt_spec["max_value"], opt_spec.get("step", 1)
                
                if param_name == "lookback_months":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(Momentum(lookback_months=int(val)))
                elif param_name == "sma_filter_window":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(BenchmarkSMA(sma_filter_window=int(val)))

        return features

    def _calculate_candidate_weights(self, look: pd.Series) -> pd.Series:
        """Calculates initial candidate weights based on momentum."""
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
        """Generates trading signals based on the momentum strategy."""
        lookback_months = self.strategy_config.get('lookback_months', 6)
        momentum_feature_name = f"momentum_{lookback_months}m"
        momentum = features[momentum_feature_name]
        
        weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        w_prev = pd.Series(index=prices.columns, dtype=float).fillna(0.0)

        # --- Immediate derisk logic ---
        sma_window = self.strategy_config.get('sma_filter_window')
        derisk_days = self.strategy_config.get('derisk_days_under_sma', 10)
        if sma_window:
            sma_feature_name = f"benchmark_sma_{sma_window}m"
            risk_on_series = features[sma_feature_name].reindex(prices.index, fill_value=1)
            # Count consecutive days under SMA
            under_sma_counter = 0
            derisk_flags = pd.Series(False, index=prices.index)
            for date in prices.index:
                if risk_on_series.loc[date]:
                    under_sma_counter = 0
                else:
                    under_sma_counter += 1
                    if under_sma_counter > derisk_days:
                        derisk_flags.loc[date] = True
            # Now, in the main loop, use derisk_flags
        else:
            derisk_flags = pd.Series(False, index=prices.index)

        for date in prices.index:
            look = momentum.loc[date]

            if look.count() == 0:
                weights.loc[date] = w_prev
                continue
            
            look = look.dropna() # Drop NaNs to avoid issues

            cand = self._calculate_candidate_weights(look)
            w_new = self._apply_leverage_and_smoothing(cand, w_prev)

            # Immediate derisk if triggered
            if derisk_flags.loc[date]:
                w_new[:] = 0.0

            weights.loc[date] = w_new
            w_prev = w_new

        # Apply SMA filter if configured
        if self.strategy_config.get('sma_filter_window'):
            sma_window = self.strategy_config['sma_filter_window']
            sma_feature_name = f"benchmark_sma_{sma_window}m"
            risk_on = features[sma_feature_name].loc[weights.index]
            weights[~risk_on] = 0.0

        return weights
