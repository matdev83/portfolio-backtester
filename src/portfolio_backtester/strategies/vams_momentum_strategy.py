import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

class VAMSMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Volatility Adjusted Momentum Scores (VAMS)."""

    def _calculate_downside_deviation(self, rets: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculates rolling downside deviation for each asset."""
        negative_rets = rets[rets < 0].fillna(0)
        downside_deviation = negative_rets.rolling(window).std()
        return downside_deviation.fillna(0)

    def _calculate_total_volatility(self, rets: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculates rolling total volatility (standard deviation) for each asset."""
        return rets.rolling(window).std().fillna(0)

    def _calculate_dp_vams(self, rets: pd.DataFrame, lookback_months: int, alpha: float) -> pd.DataFrame:
        """Calculates Downside Penalized Volatility Adjusted Momentum Scores (dp-VAMS)."""
        # Calculate rolling momentum (R_N)
        momentum = (1 + rets).rolling(lookback_months).apply(np.prod, raw=True) - 1

        # Calculate total volatility (sigma_N)
        total_vol = self._calculate_total_volatility(rets, lookback_months)

        # Calculate downside deviation (sigma_D)
        downside_dev = self._calculate_downside_deviation(rets, lookback_months)

        # Calculate the denominator: alpha * sigma_D + (1 - alpha) * sigma_N
        denominator = alpha * downside_dev + (1 - alpha) * total_vol
        
        # Avoid division by zero
        denominator = denominator.replace(0, np.nan)

        dp_vams = momentum / denominator
        return dp_vams.fillna(0) # Fill NaN (from division by zero) with 0

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

    def generate_signals(self, data: pd.DataFrame, benchmark_data: pd.Series) -> pd.DataFrame:
        """Generates trading signals based on the VAMS momentum strategy."""
        rets = data.pct_change(fill_method=None)
        
        # Calculate dp-VAMS scores
        dp_vams_scores = self._calculate_dp_vams(
            rets,
            self.strategy_config.get('lookback_months', 6),
            self.strategy_config.get('alpha', 0.5)
        )

        weights = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
        w_prev = pd.Series(index=rets.columns, dtype=float).fillna(0.0)

        for date in rets.index:
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
            sma = benchmark_data.rolling(self.strategy_config['sma_filter_window']).mean()
            risk_on = benchmark_data.shift(1) > sma.shift(1)
            weights[~risk_on] = 0.0

        return weights
