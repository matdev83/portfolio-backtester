from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from ..features.sortino_ratio import SortinoRatio

from .base_strategy import BaseStrategy


class SortinoMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Sortino ratio for ranking."""

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)

        self.w_prev: Optional[pd.Series] = None
        self.current_derisk_flag: bool = False
        self.consecutive_periods_under_sma: int = 0

        params_dict_to_update = self.strategy_config
        if "strategy_params" in self.strategy_config:
            if self.strategy_config["strategy_params"] is None:
                 self.strategy_config["strategy_params"] = {}
            params_dict_to_update = self.strategy_config["strategy_params"]

        params_dict_to_update.setdefault("apply_trading_lag", False)
        params_dict_to_update.setdefault("rolling_window", 6)
        params_dict_to_update.setdefault("target_return", 0.0)
        params_dict_to_update.setdefault("sma_filter_window", None)
        params_dict_to_update.setdefault("derisk_days_under_sma", 10)
        params_dict_to_update.setdefault("price_column_asset", "Close")
        params_dict_to_update.setdefault("price_column_benchmark", "Close")

        self.sortino_feature = SortinoRatio(
            rolling_window=params_dict_to_update["rolling_window"],
            target_return=params_dict_to_update["target_return"]
        )

        self.entry_prices: pd.Series | None = None

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "rolling_window", "sma_filter_window", "target_return", "apply_trading_lag"}

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods for SortinoMomentumStrategy.
        Requires: rolling_window for Sortino ratio calculation + SMA filter window
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        
        # Sortino ratio rolling window requirement
        rolling_window = params.get("rolling_window", 6)
        
        # SMA filter requirement (if enabled)
        sma_filter_window = params.get("sma_filter_window")
        sma_requirement = sma_filter_window if sma_filter_window and sma_filter_window > 0 else 0
        
        # Take the maximum of all requirements plus buffer
        total_requirement = max(rolling_window, sma_requirement)
        
        # Add 2-month buffer for reliable calculations
        return total_requirement + 2

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generates trading signals based on historical data and current date.
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
        if (start_date and current_date < start_date) or (end_date and current_date > end_date):
            columns = all_historical_data.columns.get_level_values(0).unique() if isinstance(all_historical_data.columns, pd.MultiIndex) else all_historical_data.columns
            return pd.DataFrame(0.0, index=[current_date], columns=columns)

        # --- Prepare Data ---
        if isinstance(all_historical_data.columns, pd.MultiIndex) and 'Field' in all_historical_data.columns.names:
            asset_prices_for_sortino = all_historical_data.xs(price_col_asset, level='Field', axis=1)
        else:
            asset_prices_for_sortino = all_historical_data.to_frame() if isinstance(all_historical_data, pd.Series) else all_historical_data

        asset_prices_hist = asset_prices_for_sortino[asset_prices_for_sortino.index <= current_date]
        # Ensure asset_prices_hist is a DataFrame
        if isinstance(asset_prices_hist, pd.Series):
            asset_prices_hist = asset_prices_hist.to_frame()

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

        # --- Calculate Scores (Sortino Ratio) ---
        scores = self.sortino_feature.compute(asset_prices_hist)
        scores_at_current_date = scores.loc[current_date] if current_date in scores.index else pd.Series(dtype=float)

        # Fix for Pylance error: Invalid conditional operand
        if bool(scores_at_current_date.isna().all()) or scores_at_current_date.empty:
            weights_at_current_date = self.w_prev.copy()
        else:
            # Ensure scores_at_current_date is a Series before passing to _calculate_candidate_weights
            if isinstance(scores_at_current_date, pd.DataFrame):
                scores_at_current_date = scores_at_current_date.squeeze()
            if not isinstance(scores_at_current_date, pd.Series):
                # This case should ideally not happen if .loc[current_date] is used correctly,
                # but as a safeguard, convert scalar to Series if necessary.
                scores_at_current_date = pd.Series([scores_at_current_date], index=[current_universe_tickers[0]]) if not current_universe_tickers.empty else pd.Series(dtype=float)

            cand_weights = self._calculate_candidate_weights(scores_at_current_date)
            w_target_pre_filter = self._apply_leverage_and_smoothing(cand_weights, self.w_prev)

            current_prices_for_assets_at_date: pd.Series = pd.Series(np.nan, index=current_universe_tickers)
            if current_date in asset_prices_for_sortino.index:
                temp_prices = asset_prices_for_sortino.loc[current_date]
                if isinstance(temp_prices, pd.DataFrame):
                    temp_prices = temp_prices.squeeze()
                if not isinstance(temp_prices, pd.Series):
                    temp_prices = pd.Series([temp_prices], index=[current_universe_tickers[0]]) if not current_universe_tickers.empty else pd.Series(dtype=float)
                current_prices_for_assets_at_date = temp_prices.reindex(current_universe_tickers).fillna(np.nan)
 
            for asset in current_universe_tickers:
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
        current_prices_for_sl_check: pd.Series = pd.Series(np.nan, index=current_universe_tickers)
        if current_date in asset_prices_for_sortino.index:
            temp_prices_sl = asset_prices_for_sortino.loc[current_date]
            if isinstance(temp_prices_sl, pd.DataFrame):
                temp_prices_sl = temp_prices_sl.squeeze()
            if not isinstance(temp_prices_sl, pd.Series):
                temp_prices_sl = pd.Series([temp_prices_sl], index=[current_universe_tickers[0]]) if not current_universe_tickers.empty else pd.Series(dtype=float)
            current_prices_for_sl_check = temp_prices_sl.reindex(current_universe_tickers).fillna(np.nan)

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
                current_benchmark_price: pd.Series = pd.Series([current_benchmark_price_raw], index=[current_date])
                current_benchmark_sma: pd.Series = pd.Series([current_benchmark_sma_raw], index=[current_date])

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
