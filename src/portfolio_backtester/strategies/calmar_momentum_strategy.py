from typing import Set, Optional
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
# Removed CalmarSignalGenerator import as it's no longer used


class CalmarMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Calmar ratio for ranking."""

    # signal_generator_class = CalmarSignalGenerator # Removed

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        # Added 'rolling_window' as it's now a direct parameter for this strategy's Calmar calculation.
        # 'sma_filter_window' might be for a benchmark filter, which is handled differently now.
        # BaseStrategy tunable_parameters will be inherited if needed.
        base_params = super().tunable_parameters()
        my_params = {"num_holdings", "rolling_window", "apply_trading_lag", "long_only", "leverage", "smoothing_lambda"}
        return base_params.union(my_params)

    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        # Ensure 'apply_trading_lag' and other relevant params have defaults
        # strategy_config might be nested under "strategy_params" by the config loader

        # Consistently access parameters from self.strategy_config
        self.rolling_window = self.strategy_config.get("rolling_window", 6) # Default if not provided
        self.apply_trading_lag = self.strategy_config.get("apply_trading_lag", False)
        self.num_holdings = self.strategy_config.get("num_holdings") # Can be None
        self.long_only = self.strategy_config.get("long_only", True)

        # These might be used by _apply_leverage_and_smoothing if called from here
        # self.leverage = self.strategy_config.get("leverage", 1.0)
        # self.smoothing_lambda = self.strategy_config.get("smoothing_lambda", 0.5)

        self.weights_history = pd.DataFrame() # To store weights for smoothing if needed

    def _calculate_calmar_ratio(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Calmar ratio for each asset.
        Assumes price_data is a DataFrame with 'Close' prices, indexed by date.
        """
        # Ensure we are working with 'Close' prices if multiple columns are present
        if isinstance(price_data, pd.DataFrame) and 'Close' in price_data.columns:
            close_prices = price_data['Close']
        elif isinstance(price_data, pd.Series): # If already a Series (e.g. of close prices)
            close_prices = price_data
        else:
            # Fallback: try to use the first column if it's a single-column DataFrame not named 'Close'
            # Or raise an error if the structure is unexpected.
            # For now, let's assume it's a Series of close prices or a DataFrame with a 'Close' column.
            if isinstance(price_data, pd.DataFrame) and len(price_data.columns) == 1:
                close_prices = price_data.iloc[:, 0]
            else:
                raise ValueError("price_data for _calculate_calmar_ratio must be a Series of close prices or a DataFrame with a 'Close' column.")

        # Group by asset if multiple assets are in a single DataFrame (multi-index or wide format)
        # For now, assume price_data is for a single asset or already grouped by asset if wide.
        # If it's a wide DataFrame (assets as columns), apply per column.
        if isinstance(close_prices, pd.DataFrame): # Assets as columns
            return close_prices.apply(lambda x: self._calculate_single_asset_calmar(x, self.rolling_window))
        else: # Single asset series
            return self._calculate_single_asset_calmar(close_prices, self.rolling_window)

    def _calculate_single_asset_calmar(self, asset_prices: pd.Series, window: int) -> pd.Series:
        """Helper to calculate Calmar for a single asset series."""
        rets = asset_prices.pct_change().fillna(0)
        cal_factor = 12  # Annualization factor (assuming monthly window for Calmar, adjust if daily)
        # If rolling_window is in months, and data is daily, need to adjust window.
        # For now, assume rolling_window is in number of periods of the data.
        # If data is daily, a rolling_window of 6 means 6 days.
        # The original feature used 'm' in its name, implying months.
        # Let's assume self.rolling_window is in months, and data is daily.
        # A common Calmar is 36 months, so window approx 36 * 21 days.
        # For simplicity here, let's assume self.rolling_window IS the number of periods for the .rolling() call.
        # User needs to set it appropriately (e.g. 252 for 1 year daily).
        # Or, if 'rolling_window' is meant to be months, convert it:
        # effective_window = window * 21 # Approximate trading days in a month

        effective_window = window # Keeping it simple: window is in number of data periods.

        rolling_mean_annualized = rets.rolling(effective_window).mean() * 252 # Annualize daily returns

        def max_drawdown(series):
            series = series.dropna()
            if series.empty:
                return 0.0 # Changed from np.nan to 0.0 to avoid issues in division
            cumulative_returns = (1 + series).cumprod()
            peak = cumulative_returns.expanding(min_periods=1).max()
            # peak = peak.replace(0, 1e-9) # Avoid division by zero if peak is 0
            drawdown = (cumulative_returns / peak.replace(0, 1e-9)) - 1 # Added replace here
            # drawdown = drawdown.replace([np.inf, -np.inf], [0, 0]).fillna(0) # Handled by replace below
            min_drawdown = abs(drawdown.min())
            return min_drawdown if min_drawdown > 0 else 1e-9 # Avoid division by zero for Calmar

        rolling_max_dd = rets.rolling(effective_window).apply(max_drawdown, raw=False)

        with np.errstate(divide='ignore', invalid='ignore'):
            calmar_ratio_series = rolling_mean_annualized / rolling_max_dd

        # Replace inf/-inf with large/small numbers, then clip.
        # Using a large positive number for inf (good calmar) and a specific indicator for undefined (e.g. when max_dd is 0)
        calmar_ratio_series.replace([np.inf], 10.0, inplace=True) # Positive Calmar is good
        calmar_ratio_series.replace([-np.inf], -10.0, inplace=True) # Negative Calmar is bad
        # If max_drawdown was 0 (or 1e-9), and rolling_mean was 0, ratio is 0.
        # If max_drawdown was 0 (or 1e-9), and rolling_mean was >0, ratio is large (capped at 10).
        # If max_drawdown was 0 (or 1e-9), and rolling_mean was <0, ratio is large negative (capped at -10).
        calmar_ratio_series.fillna(0, inplace=True) # Fill NaNs that might arise if not enough data
        calmar_ratio_series = calmar_ratio_series.clip(-10.0, 10.0)
        return calmar_ratio_series


    def generate_signals(
        self,
        all_historical_data: pd.DataFrame, # OHLCV data, columns are assets, index is date
        benchmark_historical_data: pd.DataFrame, # Not used by this strategy directly for Calmar
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generates trading signals based on Calmar ratio.
        all_historical_data: DataFrame with 'Close' prices for assets, indexed by date.
        """

        # 1. WFO Window Check
        if start_date and current_date < start_date:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)
        if end_date and current_date > end_date:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        # 2. Data Slicing for Calmar Ratio Calculation
        # Calmar ratio needs data up to and including current_date for its lookback period.
        # The 'prices' for Calmar should be 'Close' prices.
        # Assuming all_historical_data has MultiIndex columns (asset, OHLCV) or just Close for each asset.
        # For simplicity, let's assume all_historical_data provided to generate_signals
        # is a DataFrame where columns are asset tickers and values are Close prices.
        # If it's OHLCV, we need to select 'Close'.

        # Let's assume 'all_historical_data' is a DataFrame of Close prices, assets as columns.
        # If not, this part needs adjustment based on actual data structure.
        # Example: if data has MultiIndex columns: prices_for_calmar = all_historical_data.xs('Close', level=1, axis=1)
        # For now, assuming it's already asset columns of close prices:
        prices_for_calmar = all_historical_data[all_historical_data.index <= current_date]

        if prices_for_calmar.empty:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        # 3. Calculate Calmar Ratios
        # _calculate_calmar_ratio expects a DataFrame where columns are assets.
        calmar_ratios_df = self._calculate_calmar_ratio(prices_for_calmar)

        if current_date not in calmar_ratios_df.index:
            # Not enough data to calculate Calmar for current_date, or current_date is out of bounds
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        current_calmar_ratios = calmar_ratios_df.loc[current_date].dropna() # Get ratios for current_date, drop assets with NaN

        if current_calmar_ratios.empty:
            return pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)

        # 4. Calculate Candidate Weights (using BaseStrategy helper)
        # _calculate_candidate_weights expects a Series of scores (higher is better)
        candidate_weights = self._calculate_candidate_weights(current_calmar_ratios) # This returns a Series

        # 5. Apply Leverage and Smoothing (using BaseStrategy helper)
        # This requires previous weights. The backtester should manage and pass previous_weights.
        # For now, let's assume previous_weights are all zero if it's the first period or not available.
        # A more robust solution would involve the backtester passing previous_weights,
        # or the strategy maintaining state.

        # Get previous weights for smoothing
        # If current_date is the first date signals are generated for, w_prev is all zeros.
        # Otherwise, it's the weights from the previous signal generation date.
        # This requires self.weights_history to be updated by the backtester or this class.

        prev_weights_date = self.weights_history.index.asof(current_date - pd.Timedelta(days=1))
        if prev_weights_date is pd.NaT or self.weights_history.empty:
            w_prev = pd.Series(0.0, index=candidate_weights.index)
        else:
            w_prev = self.weights_history.loc[prev_weights_date]
            # Align w_prev with current candidate_weights assets, fill missing with 0
            w_prev = w_prev.reindex(candidate_weights.index).fillna(0.0)

        # Ensure w_prev and candidate_weights have the same assets in the same order
        # This should be handled by reindex if assets change, but good to be cautious.
        common_index = candidate_weights.index.intersection(w_prev.index)
        w_prev = w_prev.loc[common_index]
        candidate_weights_aligned = candidate_weights.loc[common_index]

        # If after alignment, candidate_weights_aligned is empty (e.g. no common assets with scores)
        if candidate_weights_aligned.empty and not candidate_weights.empty: # if candidate_weights was not empty before
             # This case might happen if w_prev was from a completely different universe.
             # Fallback to using original candidate_weights and zero prev_weights for them.
             w_prev_for_smoothing = pd.Series(0.0, index=candidate_weights.index)
             final_weights_series = self._apply_leverage_and_smoothing(candidate_weights, w_prev_for_smoothing)
        elif candidate_weights_aligned.empty:
            final_weights_series = pd.Series(dtype=float) # No signals to generate
        else:
            final_weights_series = self._apply_leverage_and_smoothing(candidate_weights_aligned, w_prev)


        # Reindex to include all original assets, filling missing ones with 0
        final_weights_series = final_weights_series.reindex(all_historical_data.columns).fillna(0.0)

        # Store these weights for next iteration's smoothing (and for trading lag application)
        # The backtester should call generate_signals iteratively.
        # If trading lag is applied, what's stored should be pre-lag.
        # The shift for lag happens *after* all calculations for current_date's decision.

        # Let's make a DataFrame for the current date's weights
        current_weights_df = pd.DataFrame(final_weights_series, index=[current_date]).T # Transpose to get assets as columns

        # Update history (handle potential duplicates if called multiple times for same date, though unlikely in backtest loop)
        if not current_weights_df.empty:
             # Store the pre-lagged weights. The actual trading lag is handled by the backtester
             # based on the strategy's `apply_trading_lag` flag.
             # Or, if this method is supposed to return the final, possibly lagged, weights:
            if not self.weights_history.index.isin([current_date]).any():
                self.weights_history = pd.concat([self.weights_history, current_weights_df.T]) # Store row for current_date
            else:
                self.weights_history.loc[current_date] = final_weights_series # Update if already exists


        # 6. Apply Trading Lag (if configured)
        # The generate_signals in the old CalmarMomentumStrategy did weights.shift(1).
        # This means that weights calculated based on `current_date`'s data are applied on `current_date + 1 day/period`.
        # The backtester usually handles this: it asks for signals for `current_date`,
        # receives them, and if lag is needed, it applies them on the next period.
        # If the strategy itself must return the lagged signals, then it needs to look up
        # weights calculated on a *previous* date.
        #
        # Given the new BaseStrategy structure, `generate_signals` is for `current_date`.
        # If `apply_trading_lag` is True, it means the signals for `current_date`
        # should have been decided based on data from `current_date - 1 lag period`.
        #
        # Let's reconsider the `shift(1)`:
        # If `weights` is a Series/DataFrame with a DatetimeIndex, `shift(1)` moves data one period forward.
        # So, weight for date D becomes weight for date D+1. Signal for D is used at D+1.
        # If this method is to return weights for `current_date`, and lag is True,
        # then the actual calculation should have been based on data from `current_date - lag_period`.
        #
        # The previous implementation: `weights = super().generate_signals(...)` then `weights.shift(1)`.
        # This means `super().generate_signals` produced weights for date D, D+1, etc.
        # And then these were all shifted.
        #
        # For the new model where `generate_signals` is called for a specific `current_date`:
        # If `apply_trading_lag` is True, the weights returned for `current_date`
        # should be those that were *calculated* using data up to `current_date - lag_period`.
        # This is complex if `lag_period` is not 1.
        #
        # Simpler: The strategy calculates signals based on `current_date` data.
        # The backtester applies these signals at `current_date + lag` if `apply_trading_lag` is True.
        # Let's assume this model. So, this function returns weights decided by `current_date`'s data.
        # The `apply_trading_lag` flag in `self.strategy_config` is a signal to the backtester.
        #
        # So, `CalmarMomentumStrategy` does NOT apply the shift itself.
        # It just computes weights for `current_date` based on data up to `current_date`.
        # The `apply_trading_lag` in `tunable_parameters` and `__init__` is for the backtester's knowledge.

        # Ensure output is a DataFrame with current_date as index and assets as columns
        if not isinstance(final_weights_series, pd.Series): # Should be a series at this point
            # This case should not happen if logic above is correct
            final_weights_df = pd.DataFrame(columns=all_historical_data.columns, index=[current_date]).fillna(0.0)
        else:
            final_weights_df = pd.DataFrame(final_weights_series, index=all_historical_data.columns, columns=[current_date]).T

        # Ensure all assets from input are present in output, with 0 for those not selected
        final_weights_df = final_weights_df.reindex(columns=all_historical_data.columns).fillna(0.0)

        return final_weights_df

