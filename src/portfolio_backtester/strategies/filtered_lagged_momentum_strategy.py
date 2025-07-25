from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy

# Import Numba optimization with fallback
try:
    from ..numba_optimized import momentum_scores_fast_vectorized
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False



class FilteredLaggedMomentumStrategy(BaseStrategy):
    """
    Filtered Lagged Momentum strategy implementation.
    Combines standard and predictive momentum signals with SMA filtering.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)

        self.w_prev: Optional[pd.Series] = None
        self.current_derisk_flag: bool = False
        self.consecutive_periods_under_sma: int = 0

        # Default parameters
        defaults = {
            "momentum_lookback_standard": 6,
            "momentum_lookback_predictive": 6,
            "momentum_skip_standard": 1,
            "momentum_skip_predictive": 0,
            "blending_lambda": 0.5,
            "num_holdings": None,
            "top_decile_fraction": 0.1,
            "smoothing_lambda": 0.5,
            "leverage": 1.0,
            "long_only": True,
            "sma_filter_window": 10,
            "derisk_days_under_sma": 10,
            "apply_trading_lag": False,
            "price_column_asset": "Close",
            "price_column_benchmark": "Close",
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
            "momentum_lookback_standard", "momentum_lookback_predictive",
            "momentum_skip_standard", "momentum_skip_predictive", "blending_lambda",
            "num_holdings", "top_decile_fraction", "smoothing_lambda", "leverage",
            "long_only", "sma_filter_window", "derisk_days_under_sma", "apply_trading_lag"
        }

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods for FilteredLaggedMomentumStrategy.
        Requires: max(standard_lookback + standard_skip, predictive_lookback + predictive_skip) + SMA filter
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        
        # Standard momentum requirement
        standard_lookback = params.get("momentum_lookback_standard", 6)
        standard_skip = params.get("momentum_skip_standard", 1)
        standard_requirement = standard_lookback + standard_skip
        
        # Predictive momentum requirement
        predictive_lookback = params.get("momentum_lookback_predictive", 6)
        predictive_skip = params.get("momentum_skip_predictive", 0)
        predictive_requirement = predictive_lookback + predictive_skip
        
        # SMA filter requirement
        sma_filter_window = params.get("sma_filter_window", 10)
        
        # Take the maximum of all requirements
        total_requirement = max(standard_requirement, predictive_requirement, sma_filter_window)
        
        # Add 2-month buffer for reliable calculations
        return total_requirement + 2

    def _calculate_momentum_scores(
        self,
        asset_prices: pd.DataFrame,
        lookback_months: int,
        skip_months: int,
        current_date: pd.Timestamp
    ) -> pd.Series:
        """Calculate momentum scores for given lookback and skip periods."""
        relevant_prices = asset_prices[asset_prices.index <= current_date]

        if relevant_prices.empty:
            return pd.Series(dtype=float, index=asset_prices.columns)

        # Calculate start and end points for momentum calculation
        date_t_minus_skip = current_date - pd.DateOffset(months=skip_months)
        date_t_minus_skip_minus_lookback = current_date - pd.DateOffset(months=skip_months + lookback_months)

        try:
            prices_now = relevant_prices.asof(date_t_minus_skip)
            prices_then = relevant_prices.asof(date_t_minus_skip_minus_lookback)
        except KeyError:
            return pd.Series(dtype=float, index=asset_prices.columns)

        if prices_now is None or prices_then is None:
            return pd.Series(dtype=float, index=asset_prices.columns)

        if NUMBA_AVAILABLE:
            momentum_values = momentum_scores_fast_vectorized(prices_now.values, prices_then.values)
            momentum_scores = pd.Series(momentum_values, index=prices_now.index)
        else:
            # Replace 0s or negative prices with NaN
            prices_then = prices_then.replace(0, np.nan)
            if prices_then.ndim > 0:
                prices_then[prices_then < 0] = np.nan
            elif prices_then < 0:
                prices_then = np.nan

            momentum_scores = (prices_now / prices_then) - 1
        return momentum_scores.fillna(0.0)

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generates trading signals for the FilteredLaggedMomentumStrategy.
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

        params = self.strategy_config.get("strategy_params", self.strategy_config)
        price_col_asset = params["price_column_asset"]
        price_col_benchmark = params["price_column_benchmark"]

        # --- Date Window Filtering ---
        if start_date and current_date < start_date:
            columns = (all_historical_data.columns.get_level_values(0).unique() 
                      if isinstance(all_historical_data.columns, pd.MultiIndex) 
                      else all_historical_data.columns)
            return pd.DataFrame(0.0, index=[current_date], columns=columns)
        if end_date and current_date > end_date:
            columns = (all_historical_data.columns.get_level_values(0).unique() 
                      if isinstance(all_historical_data.columns, pd.MultiIndex) 
                      else all_historical_data.columns)
            return pd.DataFrame(0.0, index=[current_date], columns=columns)

        # --- Prepare Data ---
        if isinstance(all_historical_data.columns, pd.MultiIndex) and 'Ticker' in all_historical_data.columns.names:
            asset_prices_for_momentum = all_historical_data.xs(price_col_asset, level='Field', axis=1)
        else:
            asset_prices_for_momentum = all_historical_data

        asset_prices_hist = asset_prices_for_momentum[asset_prices_for_momentum.index <= current_date]
        benchmark_prices_hist = benchmark_historical_data[benchmark_historical_data.index <= current_date]

        current_universe_tickers = asset_prices_hist.columns
        if self.w_prev is None:
            self.w_prev = pd.Series(0.0, index=current_universe_tickers)
        else:
            self.w_prev = self.w_prev.reindex(current_universe_tickers).fillna(0.0)

        if self.entry_prices is None:
            self.entry_prices = pd.Series(np.nan, index=current_universe_tickers)
        else:
            self.entry_prices = self.entry_prices.reindex(current_universe_tickers).fillna(np.nan)

        # --- Calculate Blended Momentum Scores ---
        standard_scores = self._calculate_momentum_scores(
            asset_prices_hist,
            params["momentum_lookback_standard"],
            params["momentum_skip_standard"],
            current_date
        )

        predictive_scores = self._calculate_momentum_scores(
            asset_prices_hist,
            params["momentum_lookback_predictive"],
            params["momentum_skip_predictive"],
            current_date
        )

        # Blend the two momentum signals
        blending_lambda = params["blending_lambda"]
        blended_scores = (blending_lambda * standard_scores + 
                         (1 - blending_lambda) * predictive_scores)

        if blended_scores.isna().all() or blended_scores.empty:
            weights_at_current_date = self.w_prev.copy()
        else:
            # Calculate candidate weights based on blended scores
            cand_weights = self._calculate_candidate_weights(blended_scores)
            # Apply leverage and smoothing
            w_target_pre_filter = self._apply_leverage_and_smoothing(cand_weights, self.w_prev)

            # Update Entry Prices
            current_prices_for_assets_at_date = asset_prices_hist.loc[current_date] if current_date in asset_prices_hist.index else pd.Series(dtype=float)

            for asset in current_universe_tickers:
                if asset not in w_target_pre_filter.index:
                    continue
                if not current_prices_for_assets_at_date.empty and asset in current_prices_for_assets_at_date.index:
                    asset_current_price = current_prices_for_assets_at_date[asset]
                    if pd.isna(asset_current_price):
                        continue

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
        current_prices_for_sl_check = asset_prices_hist.loc[current_date] if current_date in asset_prices_hist.index else pd.Series(dtype=float, index=current_universe_tickers)

        stop_levels = sl_handler.calculate_stop_levels(current_date, asset_ohlc_hist_for_sl, self.w_prev, self.entry_prices)
        weights_after_sl = sl_handler.apply_stop_loss(current_date, current_prices_for_sl_check, weights_at_current_date, self.entry_prices, stop_levels)

        for asset in current_universe_tickers:
            if weights_at_current_date.get(asset, 0) != 0 and weights_after_sl.get(asset, 0) == 0:
                self.entry_prices[asset] = np.nan
        # PERFORMANCE OPTIMIZATION: Only copy if we need to modify

        weights_at_current_date = weights_after_sl

        # --- Apply Risk Filters (SMA, RoRo) ---
        final_weights = weights_at_current_date

        sma_filter_window = params.get("sma_filter_window")
        if sma_filter_window and sma_filter_window > 0:
            benchmark_price_series_for_sma = benchmark_prices_hist.xs(price_col_benchmark, level='Field', axis=1) if isinstance(benchmark_prices_hist.columns, pd.MultiIndex) else benchmark_prices_hist[price_col_benchmark]
            benchmark_sma = self._calculate_benchmark_sma(benchmark_prices_hist, sma_filter_window, price_col_benchmark)

            if current_date in benchmark_price_series_for_sma.index and current_date in benchmark_sma.index:
                current_benchmark_price_raw = benchmark_price_series_for_sma.loc[current_date]
                current_benchmark_sma_raw = benchmark_sma.loc[current_date]
                current_benchmark_price = pd.Series([current_benchmark_price_raw], index=[current_date])
                current_benchmark_sma = pd.Series([current_benchmark_sma_raw], index=[current_date])

                derisk_periods = params.get("derisk_days_under_sma", 10)

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
                elif not current_benchmark_price.empty and not current_benchmark_sma.empty and \
                     current_benchmark_price_raw.item() < current_benchmark_sma_raw.item():
                    final_weights[:] = 0.0

        roro_signal_instance = self.get_roro_signal()
        if roro_signal_instance:
            roro_output = roro_signal_instance.generate_signal(
                all_historical_data[all_historical_data.index <= current_date],
                benchmark_historical_data[benchmark_historical_data.index <= current_date],
                current_date
            )

            is_roro_risk_off = False
            if isinstance(roro_output, (pd.Series, pd.DataFrame)):
                if current_date in roro_output.index:
                    is_roro_risk_off = not roro_output.loc[current_date]
            elif isinstance(roro_output, (bool, int, float)):
                is_roro_risk_off = not bool(roro_output)

            if is_roro_risk_off:
                final_weights[:] = 0.0

        for asset in current_universe_tickers:
            if weights_at_current_date.get(asset, 0) != 0 and final_weights.get(asset, 0) == 0:
                self.entry_prices[asset] = np.nan

        # PERFORMANCE OPTIMIZATION: Store reference, copy only if strategy modifies weights later


        self.w_prev = final_weights

        output_weights_df = pd.DataFrame(0.0, index=[current_date], columns=current_universe_tickers)
        output_weights_df.loc[current_date, :] = final_weights

        if self.strategy_config.get("apply_trading_lag", False):
            return output_weights_df.shift(1)
        return output_weights_df 