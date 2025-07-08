from typing import Set, Optional
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
# Removed DPVAMSSignalGenerator import


class VAMSMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Downside Penalized Volatility Adjusted Momentum Scores (dp-VAMS)."""

    # signal_generator_class = DPVAMSSignalGenerator # Removed

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        base_params = super().tunable_parameters()
        my_params = {
            "num_holdings",
            "lookback_months", # For dp-VAMS calculation
            "alpha",           # For dp-VAMS calculation
            "apply_trading_lag",
            "long_only",
            "leverage",
            "smoothing_lambda"
        }
        # "sma_filter_window" is removed as it's not used by this strategy directly
        return base_params.union(my_params)

    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)

        self.lookback_months = self.strategy_config.get("lookback_months", 6)
        self.alpha = self.strategy_config.get("alpha", 0.5) # Parameter for dp-VAMS
        self.apply_trading_lag = self.strategy_config.get("apply_trading_lag", False)
        self.num_holdings = self.strategy_config.get("num_holdings")
        self.long_only = self.strategy_config.get("long_only", True)

        self.weights_history = pd.DataFrame()

    def _calculate_dp_vams(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Downside Penalized Volatility Adjusted Momentum Scores (dp-VAMS).
        Assumes price_data is a DataFrame of close prices, indexed by date, assets as columns.
        The lookback_months parameter is assumed to be in terms of number of periods in the data.
        If data is daily and lookback_months is, e.g., 6, it means 6 days.
        This might need adjustment if lookback_months is meant to be calendar months.
        For now, assume it's number of periods.
        """
        # Assuming price_data is already 'Close' prices for assets.
        # If it's OHLCV, selection of 'Close' needs to happen before this call.

        # The original DPVAMS feature had `self.lookback_months` which was an int (e.g. 6 for 6 months)
        # If data is daily, this needs to be converted to days, e.g., self.lookback_months * 21
        # For now, let's assume self.lookback_months IS the number of periods for .rolling()
        # This means if using daily data, config 'lookback_months' should be set to e.g. 126 for 6 months.
        # Or, we make the conversion here:
        # effective_lookback_periods = self.lookback_months * 21 # Approx days in month if data is daily
        # For simplicity, using self.lookback_months directly as number of periods:
        effective_lookback_periods = self.lookback_months

        rets = price_data.pct_change().fillna(0)

        # Momentum: (1 + rets).rolling().apply(np.prod, raw=True) - 1
        # Ensure momentum calculation doesn't yield -1 for all-zero returns due to log issues or prod of 1s.
        # (1+rets).cumprod() is often used for total return over a period.
        # For rolling product:
        momentum = rets.rolling(window=effective_lookback_periods).apply(lambda x: (1 + x).prod() - 1, raw=False)
        # The raw=True with np.prod might be problematic if there are NaNs from pct_change carefully.
        # Using raw=False is safer. Or (1+rets).rolling.agg(lambda x: x.prod()) -1
        # Let's stick to the original feature's way for now, assuming it was tested.
        # momentum = (1 + rets).rolling(effective_lookback_periods).apply(np.prod, raw=True) - 1
        # Corrected momentum calculation:
        # Calculate rolling product of (1 + return)
        # (1 + rets).rolling(window=effective_lookback_periods).apply(np.prod, raw=True) gives total return factor
        # Subtract 1 to get the net momentum (return over the period)
        momentum = (1 + rets).rolling(window=effective_lookback_periods).apply(np.prod, raw=True) - 1
        momentum = momentum.fillna(0) # Fill NaNs from rolling window start

        negative_rets = rets[rets < 0].fillna(0) # Consider only negative returns for downside deviation

        # Downside deviation: std of negative returns
        downside_dev = negative_rets.rolling(effective_lookback_periods).std().fillna(0)

        # Total volatility: std of all returns
        total_vol = rets.rolling(effective_lookback_periods).std().fillna(0)

        denominator = self.alpha * downside_dev + (1 - self.alpha) * total_vol
        denominator = denominator.replace(0, np.nan) # Avoid division by zero, let it be NaN then fill

        dp_vams_scores = momentum / denominator
        return dp_vams_scores.fillna(0) # Fill NaNs that arise from division by zero or insufficient data


    def generate_signals(
        self,
        all_historical_data: pd.DataFrame, # Assumed to be DataFrame of Close prices, assets as columns
        benchmark_historical_data: pd.DataFrame, # Not used by this strategy for dp-VAMS
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generates trading signals based on dp-VAMS.
        all_historical_data: DataFrame with 'Close' prices for assets, indexed by date.
        """

        if start_date and current_date < start_date:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)
        if end_date and current_date > end_date:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        # Data for dp-VAMS calculation (up to current_date)
        # Assuming all_historical_data is already 'Close' prices.
        prices_for_dpvams = all_historical_data[all_historical_data.index <= current_date]

        if prices_for_dpvams.empty or len(prices_for_dpvams) < self.lookback_months:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        dp_vams_scores_df = self._calculate_dp_vams(prices_for_dpvams)

        if current_date not in dp_vams_scores_df.index:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        current_dp_vams_scores = dp_vams_scores_df.loc[current_date].dropna()

        if current_dp_vams_scores.empty:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        candidate_weights = self._calculate_candidate_weights(current_dp_vams_scores)

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
            final_weights_series = pd.Series(dtype=float)
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


