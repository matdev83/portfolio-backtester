from typing import Set, Optional
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
# Removed SortinoSignalGenerator import


class SortinoMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Sortino ratio for ranking."""

    # signal_generator_class = SortinoSignalGenerator # Removed

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        base_params = super().tunable_parameters()
        my_params = {
            "num_holdings",
            "rolling_window",   # For Sortino ratio
            "target_return",    # For Sortino ratio
            "apply_trading_lag",
            "long_only",
            "leverage",
            "smoothing_lambda"
        }
        # "sma_filter_window" removed
        return base_params.union(my_params)

    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)

        self.rolling_window = self.strategy_config.get("rolling_window", 6)
        self.target_return_pct = self.strategy_config.get("target_return", 0.0) # Expect this as a decimal, e.g., 0.0 for 0%
        # Ensure target_return is per-period if returns are per-period
        # If daily data, and target_return is annual, it needs conversion.
        # Assuming target_return is per-period, matching frequency of 'rets'.

        self.apply_trading_lag = self.strategy_config.get("apply_trading_lag", False)
        self.num_holdings = self.strategy_config.get("num_holdings")
        self.long_only = self.strategy_config.get("long_only", True)

        self.weights_history = pd.DataFrame()

    def _calculate_sortino_ratio(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Sortino ratio for each asset.
        Assumes price_data is a DataFrame of close prices, indexed by date, assets as columns.
        """
        effective_rolling_window = self.rolling_window

        # Using .pct_change() on the input price_data directly.
        # Assuming price_data is already sliced correctly up to current_date.
        rets = price_data.pct_change() # First row will be NaN

        # Annualization factor for Sortino ratio often sqrt(252) for daily, sqrt(12) for monthly
        # The original feature used sqrt(12). If data is daily, this needs adjustment.
        # For now, assume data frequency matches this factor (e.g. monthly data if sqrt(12)).
        # Let's make this dependent on a presumed data frequency or make it a param.
        # Given other strategies, assuming daily data is common. Let's use sqrt(252) for daily.
        # If this strategy is meant for monthly, then sqrt(12) is fine.
        # The original feature used cal_factor = np.sqrt(12)
        # And rolling_window was described as "6m" in name, implying months.
        # If data is daily, rolling_window=6 means 6 days.
        # If rolling_window is in months (e.g. 6 for 6 months), and data is daily,
        # effective_rolling_window should be self.rolling_window * 21.
        # Let's assume self.rolling_window is number of periods (e.g. days if daily data)
        # And annualization_factor should match this.
        # If window is 126 days (6m), mean daily return * 252, downside dev daily * sqrt(252).
        # Sortino = (MeanRet - TargetRet) / DownsideDev. Annualized: (AnnMeanRet - AnnTargetRet) / AnnDownsideDev
        # AnnMeanRet = daily_mean_ret * 252
        # AnnDownsideDev = daily_downside_dev * sqrt(252)
        # So, Sortino_ann = (daily_mean_ret * 252 - daily_target_ret * 252) / (daily_downside_dev * sqrt(252))
        # Sortino_ann = (daily_mean_ret - daily_target_ret) * sqrt(252) / daily_downside_dev
        # This matches the structure (excess_return * cal_factor) / (stable_downside_dev * cal_factor) if cal_factor for excess_return is 1
        # and for stable_downside_dev is 1/sqrt(ann_periods)
        # Original code: (excess_return * cal_factor) / (stable_downside_dev * cal_factor) seems to cancel cal_factor.
        # Let's re-evaluate: Sortino = (Portfolio Return – Risk-Free Rate) / Downside Deviation
        # The cal_factor in original code for numerator and denominator is unusual.
        # A common way: (mean(rets) - target_ret) / downside_dev, then annualize this ratio by sqrt(periods_in_year)

        # Sticking closer to original feature's calculation structure first:
        # Original feature had:
        # rets = data.pct_change().fillna(0) <--- fillna(0) here might be an issue for first real return
        # cal_factor = np.sqrt(12)
        # rolling_mean = rets.rolling(self.rolling_window).mean()
        # downside_deviation_calc: series[series < self.target_return], returns np.sqrt(np.mean((vals - target)**2))
        # rolling_downside_dev = rets.rolling(self.rolling_window).apply(downside_deviation_calc, raw=False)
        # excess_return = rolling_mean - self.target_return
        # stable_downside_dev = np.maximum(rolling_downside_dev, 1e-6)
        # sortino_ratio = (excess_return * cal_factor) / (stable_downside_dev * cal_factor) -> cal_factor cancels
        # sortino_ratio = sortino_ratio.clip(-10.0, 10.0)
        # return pd.DataFrame(sortino_ratio, index=rets.index, columns=rets.columns).fillna(0)

        # Revised internal calculation:
        # Don't fillna rets with 0 globally yet, handle it in rolling or after.
        # `price_data.pct_change()` creates NaNs in the first row. These propagate.

        rolling_mean_rets = rets.rolling(window=effective_rolling_window).mean()

        # Target return per period (daily if data is daily)
        # self.target_return_pct is assumed to be an annualized percentage, e.g. 0.0 for 0%
        # Convert to per-period: (1 + AnnualTarget)^(1/NumPeriodsInYear) - 1
        # Or simpler: AnnualTarget / NumPeriodsInYear
        # For now, assume self.target_return_pct is already per-period for the rets.
        # If target_return=0, it's MAR=0.

        # Per-period target for comparison with per-period rets
        # Example: If data is daily, and self.target_return_pct is an annual rate (e.g. 0.02 for 2%)
        # then daily_target_return = (1 + self.target_return_pct)**(1/252) - 1.
        # For simplicity, if strategy_config.target_return is 0.0, it means 0.0 per period.
        # Let's assume self.target_return_pct is the per-period minimum acceptable return (MAR).
        mar_per_period = self.target_return_pct


        def calculate_downside_deviation(series):
            # series here are the returns for the current window for one asset
            downside_series = series[series < mar_per_period]
            if downside_series.empty or downside_series.isnull().all(): # check for empty or all-NaN
                return np.nan # Return NaN if no downside returns or only NaNs
            # np.nanmean will ignore NaNs if any slipped through, but series.dropna() is safer
            # downside_series = downside_series.dropna()
            # if downside_series.empty: return np.nan
            return np.sqrt(np.nanmean((downside_series - mar_per_period) ** 2))

        rolling_downside_dev = rets.rolling(window=effective_rolling_window).apply(calculate_downside_deviation, raw=False)

        excess_returns = rolling_mean_rets - mar_per_period

        # To prevent division by zero or very small numbers inflating Sortino.
        # Original feature used 1e-6. Using a slightly larger epsilon for stability.
        stable_rolling_downside_dev = np.maximum(rolling_downside_dev, 1e-9)

        sortino_ratios = excess_returns / stable_rolling_downside_dev

        # Annualization: If all inputs (rets, mar_per_period) are per-period,
        # then the ratio (mean_excess_ret / downside_dev) should be annualized by sqrt(N)
        # where N is number of periods in a year (e.g., 252 for daily, 12 for monthly)
        # Let's assume N=252 for daily data, common case. This should be a parameter or derived.
        annualization_factor = np.sqrt(252) # TODO: Make this configurable or based on data freq
        sortino_ratios_annualized = sortino_ratios * annualization_factor

        sortino_ratios_annualized = sortino_ratios_annualized.clip(-10.0, 10.0)

        # Fill NaNs at the very end (e.g. from initial rolling window, or if downside_dev was NaN)
        return sortino_ratios_annualized.fillna(0)


    def generate_signals(
        self,
        all_historical_data: pd.DataFrame, # Assumed to be DataFrame of Close prices
        benchmark_historical_data: pd.DataFrame, # Not used by this strategy
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generates trading signals based on Sortino ratio.
        """

        if start_date and current_date < start_date:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)
        if end_date and current_date > end_date:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        prices_for_sortino = all_historical_data[all_historical_data.index <= current_date]

        if prices_for_sortino.empty or len(prices_for_sortino) < self.rolling_window: # rolling_window is periods
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        sortino_ratios_df = self._calculate_sortino_ratio(prices_for_sortino)

        if current_date not in sortino_ratios_df.index:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        current_sortino_ratios = sortino_ratios_df.loc[current_date].dropna()

        if current_sortino_ratios.empty:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        candidate_weights = self._calculate_candidate_weights(current_sortino_ratios)

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


