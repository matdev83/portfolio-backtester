import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

class CalmarMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Calmar ratio for ranking."""

    @classmethod
    def tunable_parameters(cls):
        return {"num_holdings", "rolling_window"}

    CAL_FACTOR = 12 # Annualization factor for monthly data

    def _calculate_rolling_calmar(self, rets: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculates the rolling Calmar ratio for a DataFrame of returns."""

        # Calculate rolling mean (annualized)
        rolling_mean = rets.rolling(window).mean() * self.CAL_FACTOR

        def max_drawdown(series):
            """Calculates maximum drawdown for a series."""
            series = series.dropna()
            if series.empty:
                return 0.0
            
            cumulative_returns = (1 + series).cumprod()
            peak = cumulative_returns.expanding(min_periods=1).max()
            
            # Avoid division by zero if the portfolio value drops to zero.
            peak = peak.replace(0, 1e-9)
            drawdown = (cumulative_returns / peak) - 1
            
            # Ensure drawdown is finite before taking min
            drawdown = drawdown.replace([np.inf, -np.inf], [0, 0]).fillna(0)
            
            min_drawdown = abs(drawdown.min())
            return min_drawdown

        # Calculate rolling maximum drawdown
        rolling_max_dd = rets.rolling(window).apply(max_drawdown, raw=False)

        # --- Calmar Ratio Calculation and Cleanup ---
        # Calculate ratio, suppressing expected division by zero warnings.
        with np.errstate(divide='ignore', invalid='ignore'):
            calmar_ratio = rolling_mean / rolling_max_dd

        # Replace inf/-inf with capped values.
        calmar_ratio.replace([np.inf, -np.inf], [10.0, -10.0], inplace=True)
        
        # Fill all NaNs with 0
        calmar_ratio = calmar_ratio.fillna(0)
        
        # Final clip to ensure everything is within bounds.
        calmar_ratio = calmar_ratio.clip(-10.0, 10.0)

        return calmar_ratio

    def _calculate_candidate_weights(self, look: pd.Series) -> pd.Series:
        """Calculates initial candidate weights based on Calmar ratio ranking."""
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
        """Generates trading signals based on the Calmar momentum strategy."""
        rets = data.pct_change(fill_method=None)
        
        # Calculate rolling Calmar ratio
        rolling_calmar = self._calculate_rolling_calmar(
            rets, 
            self.strategy_config.get('rolling_window', 6)
        )

        weights = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
        w_prev = pd.Series(index=rets.columns, dtype=float).fillna(0.0)

        for date in rets.index:
            look = rolling_calmar.loc[date]

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
            risk_on = benchmark_data > sma
            # Assume risk-on if SMA is not available (at the beginning of the series)
            risk_on[sma.isna()] = True
            weights[~risk_on] = 0.0

        return weights
