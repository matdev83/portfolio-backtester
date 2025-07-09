from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
# Removed imports for signal_generators, features as they are now internalized


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy implementation.
    Calculates momentum for assets and applies SMA-based and RoRo risk filters.
    """

    # Removed signal_generator_class

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)

        # Initialize stateful variables for iterative calls
        self.w_prev: Optional[pd.Series] = None
        self.current_derisk_flag: bool = False
        self.consecutive_periods_under_sma: int = 0

        # Default parameters
        defaults = {
            "lookback_months": 6,
            "skip_months": 0, # Standard momentum skip
            "num_holdings": None,
            "top_decile_fraction": 0.1,
            "smoothing_lambda": 0.5,
            "leverage": 1.0,
            "long_only": True,
            "sma_filter_window": None, # e.g., 10 (months for benchmark SMA)
            "derisk_days_under_sma": 10, # This is now periods (e.g. months)
            "apply_trading_lag": False,
            "price_column_asset": "Close", # Column in all_historical_data for asset prices
            "price_column_benchmark": "Close", # Column in benchmark_historical_data for benchmark prices
        }

        params_dict_to_update = self.strategy_config
        if "strategy_params" in self.strategy_config:
            if self.strategy_config["strategy_params"] is None:
                 self.strategy_config["strategy_params"] = {}
            params_dict_to_update = self.strategy_config["strategy_params"]

        for k, v in defaults.items():
            params_dict_to_update.setdefault(k, v)

        # Ensure entry_prices is initialized (moved from old base generate_signals)
        # This will be populated based on the assets in the first call to generate_signals
        self.entry_prices: pd.Series | None = None


    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {
            "lookback_months", "skip_months", "num_holdings", "top_decile_fraction",
            "smoothing_lambda", "leverage", "long_only", "sma_filter_window",
            "derisk_days_under_sma", "apply_trading_lag",
        }

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods for MomentumStrategy.
        Requires: lookback_months + skip_months + buffer for calculations
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        
        # Base momentum calculation requirement
        lookback_months = params.get("lookback_months", 6)
        skip_months = params.get("skip_months", 0)
        momentum_requirement = lookback_months + skip_months
        
        # SMA filter requirement (if enabled)
        sma_filter_window = params.get("sma_filter_window")
        sma_requirement = sma_filter_window if sma_filter_window and sma_filter_window > 0 else 0
        
        # ATR requirement for stop loss (if enabled)
        atr_requirement = 0
        stop_loss_config = self.strategy_config.get("stop_loss_config", {})
        if stop_loss_config.get("type") == "AtrBasedStopLoss":
            # ATR needs daily data, but we approximate with monthly + buffer
            atr_length = stop_loss_config.get("atr_length", 14)
            # Convert daily periods to months (roughly) and add buffer
            atr_requirement = max(1, atr_length // 20)  # ~20 trading days per month
        
        # Downside volatility sizer requirement (if used)
        dvol_requirement = 0
        if self.strategy_config.get("position_sizer") == "rolling_downside_volatility":
            dvol_window = params.get("sizer_dvol_window", 12)
            dvol_requirement = dvol_window
        
        # Take the maximum of all requirements plus a buffer
        total_requirement = max(momentum_requirement, sma_requirement, atr_requirement, dvol_requirement)
        
        # Add 2-month buffer for reliable calculations
        return total_requirement + 2

    # Removed get_required_features

    def _calculate_momentum_scores(
        self,
        asset_prices: pd.DataFrame,
        lookback_months: int,
        skip_months: int,
        current_date: pd.Timestamp
    ) -> pd.Series:
        """
        Calculates momentum scores for assets as of current_date.
        asset_prices: DataFrame with historical prices (e.g., 'Close'), indexed by date, columns by asset.
        """
        # Ensure data is monthly for month-based lookback/skip
        # Assuming asset_prices are already at the correct frequency (e.g. monthly end)
        # and contains data up to current_date.

        # Filter data up to current_date (inclusive)
        relevant_prices = asset_prices[asset_prices.index <= current_date]

        if relevant_prices.empty:
            return pd.Series(dtype=float, index=asset_prices.columns)

        # Calculate start and end points for momentum calculation based on current_date
        # Prices at t - skip_months
        date_t_minus_skip = current_date - pd.DateOffset(months=skip_months)

        # Prices at t - skip_months - lookback_months
        date_t_minus_skip_minus_lookback = current_date - pd.DateOffset(months=skip_months + lookback_months)

        try:
            prices_now = relevant_prices.asof(date_t_minus_skip) # Get the row closest to (and before or at) the date
            prices_then = relevant_prices.asof(date_t_minus_skip_minus_lookback)
        except KeyError: # Should not happen with asof if dates are in index range
            return pd.Series(dtype=float, index=asset_prices.columns)


        if prices_now is None or prices_then is None: # Not enough historical data
             return pd.Series(dtype=float, index=asset_prices.columns)

        # Replace 0s or negative prices with NaN to avoid division errors or misleading momentum
        prices_then = prices_then.replace(0, np.nan)
        if prices_then.ndim > 0 : # If it's a Series (multiple assets)
            prices_then[prices_then < 0] = np.nan
        elif prices_then < 0: # scalar
             prices_then = np.nan


        momentum_scores = (prices_now / prices_then) - 1
        return momentum_scores.fillna(0.0) # Fill NaN scores with 0, or handle as per strategy needs


    def generate_signals(
        self,
        all_historical_data: pd.DataFrame, # Universe asset data (OHLCV)
        benchmark_historical_data: pd.DataFrame, # Benchmark asset data (OHLCV)
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generates trading signals for the MomentumStrategy.
        """

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

        # --- Filter Universe by Data Availability ---
        # Only include assets that have sufficient historical data for this date
        valid_assets = self.filter_universe_by_data_availability(
            all_historical_data, current_date
        )
        
        if not valid_assets:
            logger.warning(f"No assets have sufficient data for {current_date.strftime('%Y-%m-%d')}")
            columns = (all_historical_data.columns.get_level_values(0).unique() 
                      if isinstance(all_historical_data.columns, pd.MultiIndex) 
                      else all_historical_data.columns)
            return pd.DataFrame(0.0, index=[current_date], columns=columns)

        params = self.strategy_config.get("strategy_params", self.strategy_config)
        price_col_asset = params["price_column_asset"]
        price_col_benchmark = params["price_column_benchmark"]

        # --- Date Window Filtering ---
        # If current_date is outside the WFO window, return empty weights for this date
        if start_date and current_date < start_date:
            return pd.DataFrame(index=[current_date], columns=all_historical_data.columns.get_level_values(0).unique() if isinstance(all_historical_data.columns, pd.MultiIndex) else all_historical_data.columns).fillna(0.0)
        if end_date and current_date > end_date:
            return pd.DataFrame(index=[current_date], columns=all_historical_data.columns.get_level_values(0).unique() if isinstance(all_historical_data.columns, pd.MultiIndex) else all_historical_data.columns).fillna(0.0)

        # --- Prepare Data ---
        # Extract relevant price column for assets. Assuming 'all_historical_data' has a 'Ticker' level if MultiIndex.
        if isinstance(all_historical_data.columns, pd.MultiIndex) and 'Ticker' in all_historical_data.columns.names:
            asset_prices_for_mom = all_historical_data.xs(price_col_asset, level='Field', axis=1)
            # Filter to only valid assets
            valid_asset_cols = [col for col in asset_prices_for_mom.columns if col in valid_assets]
            asset_prices_for_mom = asset_prices_for_mom[valid_asset_cols]
        else: # Assume single level column index with tickers, or already just Close prices
            # Filter to only valid assets
            valid_asset_cols = [col for col in all_historical_data.columns if col in valid_assets]
            asset_prices_for_mom = all_historical_data[valid_asset_cols]

        # Filter data up to current_date for calculations
        asset_prices_hist = asset_prices_for_mom[asset_prices_for_mom.index <= current_date]
        benchmark_prices_hist = benchmark_historical_data[benchmark_historical_data.index <= current_date]

        # Initialize w_prev if first run for the assets
        current_universe_tickers = asset_prices_hist.columns
        if self.w_prev is None:
            self.w_prev = pd.Series(0.0, index=current_universe_tickers)
        else:
            # Align w_prev with current universe, adding new assets with 0 weight, keeping existing
            self.w_prev = self.w_prev.reindex(current_universe_tickers).fillna(0.0)

        if self.entry_prices is None:
            self.entry_prices = pd.Series(np.nan, index=current_universe_tickers)
        else:
            self.entry_prices = self.entry_prices.reindex(current_universe_tickers).fillna(np.nan)

        # --- Calculate Scores (Momentum) ---
        scores_at_current_date = self._calculate_momentum_scores(
            asset_prices_hist,
            params["lookback_months"],
            params["skip_months"],
            current_date
        )

        if scores_at_current_date.isna().all() or scores_at_current_date.empty: # No valid scores
            weights_at_current_date = self.w_prev.copy()
            # Apply risk filters even if weights are unchanged
        else:
            # Calculate candidate weights based on scores
            cand_weights = self._calculate_candidate_weights(scores_at_current_date)
            # Apply leverage and smoothing
            w_target_pre_filter = self._apply_leverage_and_smoothing(cand_weights, self.w_prev)


            # --- Update Entry Prices ---
            current_prices_for_assets_at_date = asset_prices_hist.loc[current_date] if current_date in asset_prices_hist.index else pd.Series(dtype=float)

            for asset in current_universe_tickers:
                if asset not in w_target_pre_filter.index: # Should not happen if w_target_pre_filter is aligned
                    continue
                if not current_prices_for_assets_at_date.empty and asset in current_prices_for_assets_at_date.index:
                    asset_current_price = current_prices_for_assets_at_date[asset]
                    if pd.isna(asset_current_price): # If current price is NaN, cannot determine entry price
                        continue

                    if (self.w_prev.get(asset, 0) == 0 and w_target_pre_filter.get(asset, 0) != 0) or \
                       (np.sign(self.w_prev.get(asset, 0)) != np.sign(w_target_pre_filter.get(asset, 0)) and w_target_pre_filter.get(asset, 0) != 0):
                        self.entry_prices[asset] = asset_current_price
                    elif w_target_pre_filter.get(asset, 0) == 0:
                        self.entry_prices[asset] = np.nan

            weights_at_current_date = w_target_pre_filter.copy()


        # --- Apply Stop Loss ---
        sl_handler = self.get_stop_loss_handler()
        # The sl_handler methods will need to be updated to use all_historical_data and current_date
        # For now, we pass what's available and assume internal adaptation or future changes in Step 3

        # Data for stop loss: OHLCV up to current_date for assets, and current prices
        # OPTIMIZATION: Only pass the data needed for ATR calculation (last 30 periods + buffer)
        # to avoid processing the entire historical dataset for every rebalancing date
        atr_length = sl_handler.stop_loss_specific_config.get("atr_length", 14) if hasattr(sl_handler, 'stop_loss_specific_config') else 14
        buffer_periods = max(30, atr_length * 2)  # Ensure enough data for ATR calculation
        recent_data = all_historical_data[all_historical_data.index <= current_date].tail(buffer_periods)
        asset_ohlc_hist = recent_data

        current_prices_for_sl = asset_ohlc_hist.xs(price_col_asset, level='Field', axis=1).loc[current_date] if isinstance(asset_ohlc_hist.columns, pd.MultiIndex) and current_date in asset_ohlc_hist.index else (asset_ohlc_hist.loc[current_date] if current_date in asset_ohlc_hist.index else pd.Series(dtype=float))

        # TODO: The stop loss handler's interface needs to change to accept `asset_ohlc_hist` and `current_date`
        # instead of `features` and separate `prices`. This is part of Step 3 of the plan.
        # Placeholder for how it *might* be called after SL interface is updated:
        # stop_levels = sl_handler.calculate_stop_levels(current_date, asset_ohlc_hist, self.w_prev, self.entry_prices)
        # weights_after_sl = sl_handler.apply_stop_loss(current_date, current_prices_for_sl, weights_at_current_date, self.entry_prices, stop_levels)

        # Updated interface - no features needed
        stop_levels = sl_handler.calculate_stop_levels(
             current_date, asset_ohlc_hist, self.w_prev, self.entry_prices
        )
        weights_after_sl = sl_handler.apply_stop_loss(
             current_date, current_prices_for_sl, weights_at_current_date, self.entry_prices, stop_levels
        )

        for asset in current_universe_tickers: # Update entry prices if SL closed positions
            if weights_at_current_date.get(asset, 0) != 0 and weights_after_sl.get(asset, 0) == 0:
                self.entry_prices[asset] = np.nan
        weights_at_current_date = weights_after_sl


        # --- Apply Risk Filters (SMA, RoRo) ---
        final_weights = weights_at_current_date.copy()

        # SMA Filter
        sma_filter_window = params.get("sma_filter_window")
        if sma_filter_window and sma_filter_window > 0:
            benchmark_price_series_for_sma = benchmark_prices_hist.xs(price_col_benchmark, level='Field', axis=1) if isinstance(benchmark_prices_hist.columns, pd.MultiIndex) else benchmark_prices_hist[price_col_benchmark]
            benchmark_sma = self._calculate_benchmark_sma(benchmark_prices_hist, sma_filter_window, price_col_benchmark)

            if current_date in benchmark_price_series_for_sma.index and current_date in benchmark_sma.index:
                current_benchmark_price = benchmark_price_series_for_sma.loc[[current_date]] # Keep as Series
                current_benchmark_sma = benchmark_sma.loc[[current_date]] # Keep as Series

                derisk_periods = params.get("derisk_days_under_sma", 10) # periods, not days

                # Update stateful derisk flags
                self.current_derisk_flag, self.consecutive_periods_under_sma = \
                    self._calculate_derisk_flags(
                        current_benchmark_price,
                        current_benchmark_sma,
                        derisk_periods,
                        self.current_derisk_flag,
                        self.consecutive_periods_under_sma
                    )

                if self.current_derisk_flag:
                    final_weights[:] = 0.0
                # General SMA filter (if price < SMA, regardless of consecutive days)
                elif not current_benchmark_price.empty and not current_benchmark_sma.empty and \
                     current_benchmark_price.iloc[0].item() < current_benchmark_sma.iloc[0].item():
                    final_weights[:] = 0.0


        # RoRo Filter
        roro_signal_instance = self.get_roro_signal()
        if roro_signal_instance:
            # TODO: roro_signal_instance.generate_signal needs to be updated to accept
            # (all_historical_data, benchmark_historical_data, current_date)
            # This is part of Step 3 of the plan.
            # For now, assume it might take current_date and use internal data, or needs full history.
            # Let's assume it needs the same data as the strategy for now.
            # roro_signal_values = roro_signal_instance.generate_signal(all_historical_data, benchmark_historical_data, current_date)
            # Awaiting RoRo refactor. For now, let's assume it works on a date index.

            # Placeholder call assuming it works on a single date or small window around it
            # This will likely need adjustment after RoRo refactoring.
            # If RoRo needs its own data slicing, that should happen inside the RoRo class.
            # We pass the full history up to current_date.
            roro_output = roro_signal_instance.generate_signal(
                all_historical_data[all_historical_data.index <= current_date], # Pass asset data
                benchmark_historical_data[benchmark_historical_data.index <= current_date], # Pass benchmark data
                current_date
            ) # Expected to return a boolean or 0/1 for current_date

            # Assuming roro_output is a Series/DataFrame indexed by date, or a scalar for current_date
            is_roro_risk_off = False
            if isinstance(roro_output, (pd.Series, pd.DataFrame)):
                if current_date in roro_output.index:
                    is_roro_risk_off = not roro_output.loc[current_date] # Assuming True means risk-on
                # else: handle missing roro signal for date - default to risk-on or previous state?
            elif isinstance(roro_output, (bool, int, float)): # Scalar output
                is_roro_risk_off = not bool(roro_output)

            if is_roro_risk_off:
                final_weights[:] = 0.0

        # Update entry prices if risk filters zeroed out positions
        for asset in current_universe_tickers:
            if weights_at_current_date.get(asset, 0) != 0 and final_weights.get(asset, 0) == 0:
                 self.entry_prices[asset] = np.nan

        self.w_prev = final_weights.copy()

        # Create a DataFrame for the current date's weights
        output_weights_df = pd.DataFrame(0.0, index=[current_date], columns=current_universe_tickers)
        output_weights_df.loc[current_date] = final_weights

        # Apply trading lag if configured
        # Note: The backtester applies a global shift. If lag is strategy-specific AND
        # the strategy is supposed to return weights for the *actual* holding period,
        # then this shift here is correct. Otherwise, it might be double-counted.
        # For now, keeping the original logic.
        # If apply_trading_lag is True, the strategy's output for current_date
        # are the weights that should have been decided on current_date-1.
        # This requires careful handling in the backtester.
        # A simpler model is strategy generates signals for date D, backtester applies on D+1.
        # If `apply_trading_lag` is True, it means the signals computed for `current_date`
        # are based on data available prior to `current_date`, and intended for `current_date`.
        # The backtester's shift(1) handles the execution lag.
        # So, if `apply_trading_lag` is true here, it effectively means the calculation
        # itself should use data from `current_date - 1 day/period`.
        # This is getting complex. The original `weights.shift(1)` in the old code
        # was applied *after* a full series of weights was generated.
        # Here, we are generating for a single `current_date`.
        # Let's assume `apply_trading_lag` means the backtester will handle the shift,
        # and the strategy provides weights for `current_date` based on data up to `current_date`.
        # The config `apply_trading_lag` in strategy seems redundant if backtester handles it.
        # For now, I will remove the direct shift here, assuming the backtester's global shift is sufficient.
        # If a strategy-specific lag logic (e.g. signals are valid for D+N) is needed, it's more complex.

        # Revisit apply_trading_lag: The old code had it at the end of generate_signals, shifting the *entire*
        # output DataFrame. Since we now output for a single current_date, a shift here doesn't make sense
        # in the same way. The backtester's daily reindex and shift(1) should handle execution lag.
        # The `apply_trading_lag` parameter in strategy_config might be for strategies that have an *additional*
        # inherent lag in their signal generation logic itself (e.g. using data from t-2 to decide for t).
        # This is not the case for simple momentum.
        # I'll assume the strategy computes weights for `current_date` based on data available at `current_date` (or `current_date-1` if monthly).
        # The backtester will then shift these daily weights by 1 for execution.
        # So, `apply_trading_lag` in strategy config is likely for internal signal calculation lag, not execution.
        # The Momentum feature's `skip_months` already handles this type of lag.

        return output_weights_df

    # Helper to get params, similar to old BaseSignalGenerator._params()
    def _get_params(self) -> Dict[str, Any]:
        return self.strategy_config.get("strategy_params", self.strategy_config)
