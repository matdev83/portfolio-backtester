from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

# Import Numba optimization with fallback
try:
    from ..numba_optimized import rolling_sharpe_fast_portfolio
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Removed SharpeSignalGenerator and feature imports

class SharpeMomentumStrategy(BaseStrategy):
    """Strategy that uses Sharpe ratio for ranking assets."""

    # Removed signal_generator_class

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)

        self.w_prev: Optional[pd.Series] = None
        self.current_derisk_flag: bool = False
        self.consecutive_periods_under_sma: int = 0

        defaults = {
            "rolling_window": 6, # Months for Sharpe ratio calculation
            "num_holdings": None,
            "top_decile_fraction": 0.1,
            "smoothing_lambda": 0.5,
            "leverage": 1.0,
            "long_only": True,
            "sma_filter_window": None,
            "derisk_days_under_sma": 10, # Periods (months)
            "apply_trading_lag": False, # This is now more about internal signal lag if any
            "price_column_asset": "Close",
            "price_column_benchmark": "Close",
            "annualization_factor": 12 # For monthly returns
        }

        params_dict_to_update = self.strategy_config
        if "strategy_params" in self.strategy_config:
            if self.strategy_config["strategy_params"] is None:
                 self.strategy_config["strategy_params"] = {}
            params_dict_to_update = self.strategy_config["strategy_params"]

        for k, v in defaults.items():
            params_dict_to_update.setdefault(k, v)

        self.entry_prices: pd.Series | None = None

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {
            "rolling_window", "num_holdings", "top_decile_fraction",
            "smoothing_lambda", "leverage", "long_only",
            "sma_filter_window", "derisk_days_under_sma", "apply_trading_lag"
        }

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods for SharpeMomentumStrategy.
        Requires: rolling_window for Sharpe ratio calculation + SMA filter window
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        
        # Sharpe ratio rolling window requirement
        rolling_window = params.get("rolling_window", 6)
        
        # SMA filter requirement (if enabled)
        sma_filter_window = params.get("sma_filter_window")
        sma_requirement = sma_filter_window if sma_filter_window and sma_filter_window > 0 else 0
        
        # Take the maximum of all requirements plus buffer
        total_requirement = max(rolling_window, sma_requirement)
        
        # Add 2-month buffer for reliable calculations
        return total_requirement + 2

    # Removed get_required_features

    def _calculate_sharpe_scores(
        self,
        asset_daily_ohlc: pd.DataFrame,
        rolling_window_months: int,
        annualization_factor: float,
        price_column: str,
        current_date: pd.Timestamp
    ) -> pd.Series:
        """
        Calculates Sharpe ratio scores for assets as of current_date.
        asset_daily_ohlc: DataFrame with daily OHLCV data for assets.
        """
        if asset_daily_ohlc.empty:
            return pd.Series(dtype=float)

        # Extract daily close prices
        if isinstance(asset_daily_ohlc.columns, pd.MultiIndex) and 'Ticker' in asset_daily_ohlc.columns.names:
            daily_closes = asset_daily_ohlc.xs(price_column, level='Field', axis=1)
        else: # Assume flat columns, already close prices or single field
            daily_closes = asset_daily_ohlc

        # Filter up to current_date
        daily_closes = daily_closes[daily_closes.index <= current_date]

        if daily_closes.empty:
            return pd.Series(dtype=float, index=asset_daily_ohlc.columns.get_level_values(0).unique() if isinstance(asset_daily_ohlc.columns, pd.MultiIndex) else asset_daily_ohlc.columns)

        # We need the Sharpe ratio *as of* the month-end corresponding to current_date or just before.
        current_month_end = current_date.to_period('M').to_timestamp('M')

        if NUMBA_AVAILABLE:
            # Vectorised Sharpe calculation using Numba kernel
            daily_returns = daily_closes.pct_change(fill_method=None)
            if daily_returns.empty:
                return pd.Series(0.0, index=daily_closes.columns)

            returns_np = daily_returns.to_numpy(dtype=np.float64)
            window_days = rolling_window_months * 21  # Approximate trading days per month

            try:
                from ..numba_optimized import sharpe_fast
                sharpe_mat = sharpe_fast(returns_np, window_days, annualization_factor=252.0)
                sharpe_ratio_calculated = pd.DataFrame(sharpe_mat, index=daily_returns.index, columns=daily_returns.columns)
            except ImportError:
                # Fallback to per-asset fast function
                sharpe_ratio_calculated = pd.DataFrame(index=daily_closes.index, columns=daily_closes.columns)
                for asset in daily_closes.columns:
                    sharpe_ratio_calculated[asset] = rolling_sharpe_fast_portfolio(daily_closes[asset].values, rolling_window_months, annualization_factor)
        else:
            # Resample to monthly, calculate monthly returns
            monthly_closes = daily_closes.resample('ME').last()
            monthly_rets = monthly_closes.pct_change(fill_method=None).fillna(0)

            # Ensure we have enough data for rolling calculation up to current_date's month-end
            # The monthly_rets will be indexed by month-ends.
            relevant_monthly_rets = monthly_rets[monthly_rets.index <= current_month_end]

            if len(relevant_monthly_rets) < rolling_window_months:
                return pd.Series(0.0, index=daily_closes.columns) # Not enough history for any asset

            rolling_mean = relevant_monthly_rets.rolling(window=rolling_window_months).mean()
            rolling_std = relevant_monthly_rets.rolling(window=rolling_window_months).std()

            # Avoid division by zero; replace zero std with NaN, then fill resulting NaN Sharpe with 0
            sharpe_ratio = (rolling_mean * annualization_factor) / (rolling_std.replace(0, np.nan) * np.sqrt(annualization_factor))
            # Using np.sqrt(annualization_factor) for std dev based on common practice if returns are already annualized mean
            # Or, if returns are not annualized in mean, then (mean * ann_factor) / (std * sqrt(ann_factor))
            # The original code did: (rolling_mean * cal_factor) / (rolling_std * cal_factor).replace(0, np.nan)
            # where cal_factor was sqrt(12). This implies (mean / std) * sqrt(ann_factor) if mean/std are not annualized.
            # Let's stick to (mean_monthly_ret * ann_factor) / (std_dev_monthly_ret * sqrt(ann_factor))
            # This simplifies to (mean_monthly_ret / std_dev_monthly_ret) * sqrt(ann_factor)

            sharpe_ratio_calculated = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(annualization_factor)


        # Get the latest calculated Sharpe ratio (for the current_month_end)
        if not sharpe_ratio_calculated.empty and current_month_end in sharpe_ratio_calculated.index:
            latest_sharpe = sharpe_ratio_calculated.loc[current_month_end].fillna(0.0)
            return latest_sharpe
        else: # Not enough rolling data, or current_month_end not in index
            return pd.Series(0.0, index=daily_closes.columns)


    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:

        # --- Data Sufficiency Validation ---
        is_sufficient, reason = self.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )
        if not is_sufficient:
            # Return zero weights if insufficient data
            columns = (all_historical_data.columns.get_level_values(0).unique() 
                      if isinstance(all_historical_data.columns, pd.MultiIndex) 
                      else all_historical_data.columns)
            return pd.DataFrame(0.0, index=[current_date], columns=columns)

        params = self._get_params()
        price_col_asset = params["price_column_asset"]
        price_col_benchmark = params["price_column_benchmark"]

        if start_date and current_date < start_date:
            return pd.DataFrame(index=[current_date], columns=all_historical_data.columns.get_level_values(0).unique() if isinstance(all_historical_data.columns, pd.MultiIndex) else all_historical_data.columns).fillna(0.0)
        if end_date and current_date > end_date:
            return pd.DataFrame(index=[current_date], columns=all_historical_data.columns.get_level_values(0).unique() if isinstance(all_historical_data.columns, pd.MultiIndex) else all_historical_data.columns).fillna(0.0)

        # Prepare asset data (universe)
        if isinstance(all_historical_data.columns, pd.MultiIndex) and 'Ticker' in all_historical_data.columns.names:
             current_universe_tickers = all_historical_data.columns.get_level_values('Ticker').unique()
             asset_data_for_scores = all_historical_data.xs(price_col_asset, level='Field', axis=1)
        else:
             current_universe_tickers = all_historical_data.columns
             asset_data_for_scores = all_historical_data # Assuming it's already 'Close' prices or single field

        if self.w_prev is None:
            self.w_prev = pd.Series(0.0, index=current_universe_tickers)
        else:
            self.w_prev = self.w_prev.reindex(current_universe_tickers).fillna(0.0)

        if self.entry_prices is None:
            self.entry_prices = pd.Series(np.nan, index=current_universe_tickers)
        else:
            self.entry_prices = self.entry_prices.reindex(current_universe_tickers).fillna(np.nan)

        # Calculate Sharpe scores
        scores_at_current_date = self._calculate_sharpe_scores(
            all_historical_data, # Pass full OHLCV, scorer will pick 'Close'
            params["rolling_window"],
            params["annualization_factor"],
            price_col_asset,
            current_date
        )

        if scores_at_current_date.isna().all() or scores_at_current_date.empty:
            weights_at_current_date = self.w_prev.copy()
        else:
            cand_weights = self._calculate_candidate_weights(scores_at_current_date)
            w_target_pre_filter = self._apply_leverage_and_smoothing(cand_weights, self.w_prev)

            # Update Entry Prices (using daily close prices from all_historical_data)
            asset_closes_hist = asset_data_for_scores[asset_data_for_scores.index <= current_date]
            current_prices_for_assets_at_date = asset_closes_hist.loc[current_date] if current_date in asset_closes_hist.index else pd.Series(dtype=float)

            for asset in current_universe_tickers:
                if not current_prices_for_assets_at_date.empty and asset in current_prices_for_assets_at_date.index:
                    asset_current_price = current_prices_for_assets_at_date[asset]
                    if pd.isna(asset_current_price): continue

                    if (self.w_prev.get(asset, 0) == 0 and w_target_pre_filter.get(asset, 0) != 0) or \
                       (np.sign(self.w_prev.get(asset, 0)) != np.sign(w_target_pre_filter.get(asset, 0)) and w_target_pre_filter.get(asset, 0) != 0):
                        self.entry_prices[asset] = asset_current_price
                    elif w_target_pre_filter.get(asset, 0) == 0:
                        self.entry_prices[asset] = np.nan
            # PERFORMANCE OPTIMIZATION: Only copy if we need to modify

            weights_at_current_date = w_target_pre_filter

        # --- Apply Stop Loss ---
        sl_handler = self.get_stop_loss_handler()
        asset_ohlc_hist_for_sl = all_historical_data[all_historical_data.index <= current_date]
        current_prices_for_sl_check = asset_data_for_scores.loc[current_date] if current_date in asset_data_for_scores.index else pd.Series(dtype=float, index=current_universe_tickers)

        stop_levels = sl_handler.calculate_stop_levels(current_date, asset_ohlc_hist_for_sl, self.w_prev, self.entry_prices)
        weights_after_sl = sl_handler.apply_stop_loss(current_date, current_prices_for_sl_check, weights_at_current_date, self.entry_prices, stop_levels)

        for asset in current_universe_tickers:
            if weights_at_current_date.get(asset, 0) != 0 and weights_after_sl.get(asset, 0) == 0:
                self.entry_prices[asset] = np.nan
        # PERFORMANCE OPTIMIZATION: Only copy if we need to modify

        weights_at_current_date = weights_after_sl

        # --- Apply Risk Filters (SMA, RoRo) ---
        final_weights = weights_at_current_date
        benchmark_prices_hist = benchmark_historical_data[benchmark_historical_data.index <= current_date]

        sma_filter_window = params.get("sma_filter_window")
        if sma_filter_window and sma_filter_window > 0:
            benchmark_price_series_for_sma = benchmark_prices_hist.xs(price_col_benchmark, level='Field', axis=1) if isinstance(benchmark_prices_hist.columns, pd.MultiIndex) else benchmark_prices_hist[price_col_benchmark]
            benchmark_sma_series = self._calculate_benchmark_sma(benchmark_prices_hist, sma_filter_window, price_col_benchmark)

            if current_date in benchmark_price_series_for_sma.index and current_date in benchmark_sma_series.index:
                current_benchmark_price_val = benchmark_price_series_for_sma.loc[[current_date]]
                current_benchmark_sma_val = benchmark_sma_series.loc[[current_date]]
                derisk_periods = params.get("derisk_days_under_sma", 10)

                self.current_derisk_flag, self.consecutive_periods_under_sma = \
                    self._calculate_derisk_flags(current_benchmark_price_val, current_benchmark_sma_val, derisk_periods, self.current_derisk_flag, self.consecutive_periods_under_sma)

                if self.current_derisk_flag or \
                   (not current_benchmark_price_val.empty and not current_benchmark_sma_val.empty and current_benchmark_price_val.iloc[0] < current_benchmark_sma_val.iloc[0]):
                    final_weights[:] = 0.0

        roro_signal_instance = self.get_roro_signal()
        if roro_signal_instance:
            is_roro_risk_off = not roro_signal_instance.generate_signal(all_historical_data, benchmark_historical_data, current_date)
            if is_roro_risk_off:
                final_weights[:] = 0.0

        for asset in current_universe_tickers:
            if weights_at_current_date.get(asset, 0) != 0 and final_weights.get(asset, 0) == 0:
                 self.entry_prices[asset] = np.nan

        # PERFORMANCE OPTIMIZATION: Store reference, copy only if strategy modifies weights later


        self.w_prev = final_weights

        output_weights_df = pd.DataFrame(0.0, index=[current_date], columns=current_universe_tickers)
        output_weights_df.loc[current_date] = final_weights
        return output_weights_df

    def _get_params(self) -> Dict[str, Any]:
        return self.strategy_config.get("strategy_params", self.strategy_config)
