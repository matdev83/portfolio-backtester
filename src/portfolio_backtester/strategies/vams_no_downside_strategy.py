from typing import Set, Optional
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
# Removed VAMSSignalGenerator import


class VAMSNoDownsideStrategy(BaseStrategy):
    """Momentum strategy implementation using Volatility Adjusted Momentum Scores (VAMS),
    without downside volatility penalization."""

    # signal_generator_class = VAMSSignalGenerator # Removed

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        base_params = super().tunable_parameters()
        my_params = {
            "num_holdings",
            "lookback_months", # For VAMS calculation
            # "top_decile_fraction", # Can be used if num_holdings is not set, BaseStrategy handles
            "smoothing_lambda",
            "leverage",
            "apply_trading_lag",
            "long_only"
        }
        # "sma_filter_window" removed as it's not directly used here
        return base_params.union(my_params)

    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)

        self.lookback_months = self.strategy_config.get("lookback_months", 6)
        # No alpha parameter for this VAMS version
        self.apply_trading_lag = self.strategy_config.get("apply_trading_lag", False)
        self.num_holdings = self.strategy_config.get("num_holdings")
        self.long_only = self.strategy_config.get("long_only", True)

        self.weights_history = pd.DataFrame()

    def _calculate_vams(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Volatility Adjusted Momentum Scores (VAMS).
        Assumes price_data is a DataFrame of close prices, indexed by date, assets as columns.
        lookback_months is interpreted as number of periods.
        """
        effective_lookback_periods = self.lookback_months

        rets = price_data.pct_change().fillna(0)

        # Momentum calculation
        momentum = (1 + rets).rolling(window=effective_lookback_periods).apply(np.prod, raw=True) - 1
        momentum = momentum.fillna(0) # Fill NaNs from rolling window start

        # Total volatility: std of all returns
        total_vol = rets.rolling(effective_lookback_periods).std() # No fillna(0) here yet

        denominator = total_vol.replace(0, np.nan) # Avoid division by zero, let it be NaN

        vams_scores = momentum / denominator
        # The original VAMS feature did not fillna after division.
        # However, for consistency with DPVAMS and to avoid issues downstream, filling with 0 might be safer.
        # Let's check if test_vams_no_downside_strategy expects NaNs or 0s for these cases.
        # The test had: `self.assertTrue(vams_scores.iloc[:lookback_months-1].isna().all().all())`
        # This implies NaNs are expected where calculation isn't possible.
        # If momentum is 0 and vol is 0 (np.nan after replace), then 0/nan = nan.
        # If momentum is non-zero and vol is 0 (np.nan), then x/nan = nan.
        # So, let's not fillna here to match original VAMS behavior and test expectation.
        return vams_scores # Return potentially with NaNs

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame, # Assumed to be DataFrame of Close prices
        benchmark_historical_data: pd.DataFrame, # Not used by this strategy for VAMS
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generates trading signals based on VAMS (no downside penalization).
        """

        if start_date and current_date < start_date:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)
        if end_date and current_date > end_date:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        prices_for_vams = all_historical_data[all_historical_data.index <= current_date]

        if prices_for_vams.empty or len(prices_for_vams) < self.lookback_months:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        vams_scores_df = self._calculate_vams(prices_for_vams)

        if current_date not in vams_scores_df.index:
            # Not enough data or current_date out of bounds for calculated scores
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        current_vams_scores = vams_scores_df.loc[current_date].dropna() # Drop assets with NaN VAMS scores

        if current_vams_scores.empty:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        candidate_weights = self._calculate_candidate_weights(current_vams_scores)

        prev_weights_date = self.weights_history.index.asof(current_date - pd.Timedelta(days=1))
        if prev_weights_date is pd.NaT or self.weights_history.empty:
            w_prev = pd.Series(0.0, index=candidate_weights.index)
        else:
            w_prev = self.weights_history.loc[prev_weights_date]
            w_prev = w_prev.reindex(candidate_weights.index).fillna(0.0)

        common_index = candidate_weights.index.intersection(w_prev.index)
        w_prev_aligned = w_prev.loc[common_index]
        candidate_weights_aligned = candidate_weights.loc[common_index]

        if candidate_weights_aligned.empty and not candidate_weights.empty:
             w_prev_for_smoothing = pd.Series(0.0, index=candidate_weights.index)
             final_weights_series = self._apply_leverage_and_smoothing(candidate_weights, w_prev_for_smoothing)
        elif candidate_weights_aligned.empty:
            final_weights_series = pd.Series(dtype=float) # No signals to generate
        else:
            final_weights_series = self._apply_leverage_and_smoothing(candidate_weights_aligned, w_prev_aligned)

        final_weights_series = final_weights_series.reindex(all_historical_data.columns).fillna(0.0)

        current_weights_df_row = pd.DataFrame(final_weights_series, index=[current_date]).T

        if not current_weights_df_row.empty:
            if not self.weights_history.index.isin([current_date]).any():
                self.weights_history = pd.concat([self.weights_history, current_weights_df_row.T])
            else:
                self.weights_history.loc[current_date] = final_weights_series

        final_weights_df = pd.DataFrame(final_weights_series, index=all_historical_data.columns, columns=[current_date]).T
        final_weights_df = final_weights_df.reindex(columns=all_historical_data.columns).fillna(0.0)

        return final_weights_df

