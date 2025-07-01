import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    """Momentum strategy implementation."""

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {
            "lookback_months", "num_holdings", "top_decile_fraction",
            "smoothing_lambda", "leverage", "long_only", "sma_filter_window",
        }

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

    def generate_signals(self, data: pd.DataFrame, benchmark_data: pd.Series) -> pd.DataFrame:
        """Generates trading signals based on the momentum strategy."""
        rets = data.pct_change(fill_method=None)
        # Calculate momentum based on past returns only.
        momentum = (1 + rets).rolling(
            self.strategy_config.get('lookback_months', 6)
        ).apply(np.prod, raw=True) - 1
        momentum.fillna(0, inplace=True)
        
        weights = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
        w_prev = pd.Series(index=rets.columns, dtype=float).fillna(0.0)

        for date in rets.index:
            look = momentum.loc[date]

            if look.count() == 0:
                weights.loc[date] = w_prev
                continue
            
            look = look.dropna() # Drop NaNs to avoid issues

            cand = self._calculate_candidate_weights(look)
            w_new = self._apply_leverage_and_smoothing(cand, w_prev)

            weights.loc[date] = w_new
            w_prev = w_new

        # Apply SMA filter if configured
        if self.strategy_config.get('sma_filter_window'):
            sma = benchmark_data.rolling(self.strategy_config['sma_filter_window']).mean()
            risk_on = benchmark_data > sma
            # Assume risk-on if SMA is not available (at the beginning of the series)
            risk_on[sma.isna()] = True
            weights[~risk_on] = 0.0

        return weights
