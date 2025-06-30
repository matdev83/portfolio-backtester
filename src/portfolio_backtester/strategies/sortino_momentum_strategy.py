import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

class SortinoMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Sortino ratio for ranking."""

    def _calculate_rolling_sortino(self, rets: pd.DataFrame, window: int, target_return: float = 0.0) -> pd.DataFrame:
        """Calculates rolling Sortino ratio for each asset."""
        # Annualization factor for monthly data
        CAL_FACTOR = np.sqrt(12)

        # Calculate rolling mean
        rolling_mean = rets.rolling(window).mean()

        # Calculate downside deviation (semi-deviation)
        def downside_deviation(series):
            """Calculate downside deviation for a series."""
            downside_returns = series[series < target_return]
            if len(downside_returns) == 0:
                return 1e-9  # Return a small positive number to avoid infinite Sortino ratios
            return np.sqrt(np.mean((downside_returns - target_return) ** 2))

        # Calculate rolling downside deviation
        rolling_downside_dev = rets.rolling(window).apply(downside_deviation, raw=False)

        # Calculate Sortino ratio
        # Sortino = (Mean Return - Target Return) / Downside Deviation
        excess_return = rolling_mean - target_return
        
        # Ensure downside deviation is never too small to cause extreme ratios
        # Add a small epsilon to the denominator to prevent division by near-zero
        # This is a common technique to stabilize ratio calculations.
        stable_downside_dev = np.maximum(rolling_downside_dev, 1e-6) # Ensure minimum downside deviation

        sortino_ratio = (excess_return * CAL_FACTOR) / (stable_downside_dev * CAL_FACTOR)
        
        # Cap the Sortino ratio to prevent extreme values from dominating
        sortino_ratio = sortino_ratio.clip(-10.0, 10.0)
        
        sortino_ratio = pd.DataFrame(sortino_ratio, index=rets.index, columns=rets.columns)
        return sortino_ratio.fillna(0)

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

    def generate_signals(self, data: pd.DataFrame, benchmark_data: pd.Series) -> pd.DataFrame:
        """Generates trading signals based on the Sortino momentum strategy."""
        rets = data.pct_change(fill_method=None)
        
        # Calculate rolling Sortino ratio
        target_return = self.strategy_config.get('target_return', 0.0)
        rolling_sortino = self._calculate_rolling_sortino(
            rets, 
            self.strategy_config.get('rolling_window', 6),
            target_return
        )

        weights = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
        w_prev = pd.Series(index=rets.columns, dtype=float).fillna(0.0)

        for date in rets.index:
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
            sma = benchmark_data.rolling(self.strategy_config['sma_filter_window']).mean()
            risk_on = benchmark_data.shift(1) > sma.shift(1)
            weights[~risk_on] = 0.0

        return weights
