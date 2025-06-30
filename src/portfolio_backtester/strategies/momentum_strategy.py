import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    """Momentum strategy implementation."""

    def _calculate_candidate_weights(self, look: pd.Series) -> pd.Series:
        """Calculates initial candidate weights based on momentum."""
        if self.strategy_config.get('num_holdings'):
            num_holdings = self.strategy_config['num_holdings']
        else:
            num_holdings = max(int(np.ceil(self.strategy_config['top_decile_fraction'] * look.count())), 1)

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
        if cand[cand > 0].sum():
            cand[cand > 0] /= cand[cand > 0].sum() / self.strategy_config['leverage']
        if cand[cand < 0].sum():
            cand[cand < 0] /= -cand[cand < 0].sum() / self.strategy_config['leverage']

        w_new = self.strategy_config['smoothing_lambda'] * w_prev + (1 - self.strategy_config['smoothing_lambda']) * cand

        if w_new[w_new > 0].sum():
            w_new[w_new > 0] /= w_new[w_new > 0].sum() / self.strategy_config['leverage']
        if w_new[w_new < 0].sum():
            w_new[w_new < 0] /= -w_new[w_new < 0].sum() / self.strategy_config['leverage']
        return w_new

    def generate_signals(self, data: pd.DataFrame, benchmark_data: pd.Series) -> pd.DataFrame:
        """Generates trading signals based on the momentum strategy."""
        rets = data.pct_change(fill_method=None)
        # Calculate momentum based on past returns only.
        momentum = (1 + rets).shift(1).rolling(self.strategy_config['lookback_months']).apply(np.prod, raw=True) - 1

        weights = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
        w_prev = pd.Series(index=rets.columns, dtype=float).fillna(0.0)

        for date in rets.index:
            look = momentum.loc[date]

            if look.count() == 0:
                weights.loc[date] = w_prev
                continue

            cand = self._calculate_candidate_weights(look)
            w_new = self._apply_leverage_and_smoothing(cand, w_prev)

            weights.loc[date] = w_new
            w_prev = w_new

        # Apply SMA filter if configured
        if self.strategy_config.get('sma_filter_window'):
            sma = benchmark_data.rolling(self.strategy_config['sma_filter_window']).mean()
            risk_on = benchmark_data.shift(1) > sma.shift(1)
            weights[~risk_on] = 0.0

        return weights