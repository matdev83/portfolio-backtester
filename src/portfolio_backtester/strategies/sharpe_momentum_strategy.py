import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

class SharpeMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Sharpe ratio for ranking."""

    def _calculate_rolling_sharpe(self, rets: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculates rolling Sharpe ratio for each asset."""
        # Annualization factor for monthly data
        CAL_FACTOR = np.sqrt(12)

        # Calculate rolling mean and standard deviation
        rolling_mean = rets.rolling(window).mean()
        rolling_std = rets.rolling(window).std()

        # Calculate Sharpe ratio
        # Add a small epsilon to avoid division by zero for assets with zero volatility
        sharpe_ratio = (rolling_mean * CAL_FACTOR) / (rolling_std * CAL_FACTOR).replace(0, np.nan)
        return sharpe_ratio.fillna(0) # Fill NaN (from division by zero) with 0

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
        # Scale candidate weights to target leverage
        if cand[cand > 0].sum():
            cand[cand > 0] /= cand[cand > 0].sum() / self.strategy_config['leverage']
        if cand[cand < 0].sum():
            cand[cand < 0] /= -cand[cand < 0].sum() / self.strategy_config['leverage']

        # Apply smoothing
        w_new = self.strategy_config['smoothing_lambda'] * w_prev + (1 - self.strategy_config['smoothing_lambda']) * cand

        # Re-scale smoothed weights to target leverage
        if w_new[w_new > 0].sum():
            w_new[w_new > 0] /= w_new[w_new > 0].sum() / self.strategy_config['leverage']
        if w_new[w_new < 0].sum():
            w_new[w_new < 0] /= -w_new[w_new < 0].sum() / self.strategy_config['leverage']
        return w_new

    def generate_signals(self, data: pd.DataFrame, benchmark_data: pd.Series) -> pd.DataFrame:
        """Generates trading signals based on the momentum strategy."""
        rets = data.pct_change(fill_method=None)
        
        # Calculate rolling Sharpe ratio
        rolling_sharpe = self._calculate_rolling_sharpe(rets, self.strategy_config['rolling_window'])

        weights = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
        w_prev = pd.Series(index=rets.columns, dtype=float).fillna(0.0)

        for date in rets.index:
            look = rolling_sharpe.loc[date]

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