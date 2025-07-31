from ..candidate_weights import default_candidate_weights
from ..leverage_and_smoothing import apply_leverage_and_smoothing
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from ..base.portfolio_strategy import PortfolioStrategy


class VAMSNoDownsideStrategy(PortfolioStrategy):
    def _calculate_candidate_weights(self, scores: pd.Series) -> pd.Series:
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        return default_candidate_weights(scores, params)

    def _apply_leverage_and_smoothing(self, candidate_weights: pd.Series, prev_weights: Optional[pd.Series]) -> pd.Series:
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        return apply_leverage_and_smoothing(candidate_weights, prev_weights, params)
    """Momentum strategy implementation using Volatility Adjusted Momentum Scores (VAMS), without downside volatility penalization."""

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)

        self.w_prev: Optional[pd.Series] = None
        self.current_derisk_flag: bool = False
        self.consecutive_periods_under_sma: int = 0

        defaults = {
            "lookback_months": 6,
            "num_holdings": None,
            "top_decile_fraction": 0.1,
            "smoothing_lambda": 0.5,
            "leverage": 1.0,
            "long_only": True,
            "sma_filter_window": None,
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
        return {"num_holdings", "lookback_months", "top_decile_fraction", "smoothing_lambda", "leverage",
                "sma_filter_window", "derisk_days_under_sma", "apply_trading_lag"}

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods for VAMSNoDownsideStrategy.
        Requires: lookback_months for VAMS calculation + SMA filter window
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        
        # VAMS lookback requirement
        lookback_months = params.get("lookback_months", 6)
        
        # SMA filter requirement (if enabled)
        sma_filter_window = params.get("sma_filter_window")
        sma_requirement = sma_filter_window if sma_filter_window and sma_filter_window > 0 else 0
        
        # Take the maximum of all requirements plus buffer
        total_requirement = max(lookback_months, sma_requirement)
        return total_requirement + 2

    def _calculate_vams_scores_no_downside(
        self,
        asset_prices: pd.DataFrame,
        lookback_months: int,
        current_date: pd.Timestamp
    ) -> pd.Series:
        """
        Calculates Volatility Adjusted Momentum Scores (VAMS) for assets as of current_date,
        without penalizing for downside volatility (equivalent to alpha=0).
        """
        relevant_prices = asset_prices[asset_prices.index <= current_date]

        if relevant_prices.empty:
            return pd.Series(dtype=float, index=asset_prices.columns)

        try:
            price_now = relevant_prices.loc[current_date]
            price_l_months_ago_df = relevant_prices.asof(current_date - pd.DateOffset(months=lookback_months))
            if price_l_months_ago_df.empty:
                return pd.Series(dtype=float, index=asset_prices.columns)
            price_l_months_ago = price_l_months_ago_df.iloc[0] # This should already be a Series

            # Ensure both are Series before arithmetic operations
            price_now = price_now.astype(float)
            price_l_months_ago = price_l_months_ago.astype(float)

            momentum = (price_now / price_l_months_ago) - 1
            return pd.Series(momentum.values, index=momentum.index).fillna(0.0)

        except Exception as e:
            print(f"Error in _calculate_vams_scores_no_downside for date {current_date}: {e}")
            return pd.Series(dtype=float, index=asset_prices.columns).fillna(0.0)



    def generate_signals(
        self,
        all_historical_data: pd.DataFrame, # Universe asset data (OHLCV)
        benchmark_historical_data: pd.DataFrame, # Benchmark asset data (OHLCV)
        non_universe_historical_data: pd.DataFrame, # For signature compatibility, not used here
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generates trading signals for the VAMSNoDownsideStrategy.
        Refactored: all data extraction and conversion to Numpy at the start, vectorized entry/exit logic, minimized Python loops.
        """
        # --- Data Sufficiency Validation ---
        is_sufficient, reason = self.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )
        if not is_sufficient:
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

        # --- Prepare Data: convert to Numpy early ---
        if isinstance(all_historical_data.columns, pd.MultiIndex) and 'Ticker' in all_historical_data.columns.names:
            asset_prices_for_vams = all_historical_data.xs(price_col_asset, level='Field', axis=1)
        else:
            asset_prices_for_vams = all_historical_data.to_frame() if isinstance(all_historical_data, pd.Series) else all_historical_data

        asset_prices_hist = asset_prices_for_vams[asset_prices_for_vams.index <= current_date]
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

        # --- Calculate Scores (VAMS) ---
        scores_at_current_date = self._calculate_vams_scores_no_downside(
            asset_prices_hist,
            params["lookback_months"],
            current_date
        )

        # --- Vectorized candidate weights and entry/exit logic ---
        if scores_at_current_date.isna().all() or scores_at_current_date.empty:
            weights_at_current_date = self.w_prev.copy()
        else:
            cand_weights = self._calculate_candidate_weights(scores_at_current_date)
            w_target_pre_filter = self._apply_leverage_and_smoothing(cand_weights, self.w_prev)

            # Vectorized entry/exit logic
            if current_date in asset_prices_for_vams.index:
                temp_prices = asset_prices_for_vams.loc[current_date]
                if isinstance(temp_prices, pd.DataFrame):
                    temp_prices = temp_prices.squeeze()
                if not isinstance(temp_prices, pd.Series):
                    temp_prices = pd.Series([temp_prices], index=[current_universe_tickers[0]]) if not current_universe_tickers.empty else pd.Series(dtype=float)
                current_prices_for_assets_at_date = temp_prices.reindex(current_universe_tickers).fillna(np.nan)
            else:
                current_prices_for_assets_at_date = pd.Series(np.nan, index=current_universe_tickers)

            prev = np.asarray(self.w_prev)
            target = np.asarray(w_target_pre_filter)
            entry = np.asarray(self.entry_prices).copy()
            prices = np.asarray(current_prices_for_assets_at_date)

            # Entry: where prev==0 and target!=0, or sign change
            entry_mask = ((prev == 0) & (target != 0)) | ((np.sign(prev) != np.sign(target)) & (target != 0))
            exit_mask = (target == 0)
            entry[entry_mask] = prices[entry_mask]
            entry[exit_mask] = np.nan
            self.entry_prices = pd.Series(entry, index=current_universe_tickers)
            weights_at_current_date = w_target_pre_filter

        # --- Apply Stop Loss ---
        sl_handler = self.get_stop_loss_handler()
        asset_ohlc_hist_for_sl = all_historical_data[all_historical_data.index <= current_date]
        if current_date in asset_prices_for_vams.index:
            temp_prices_sl = asset_prices_for_vams.loc[current_date]
            if isinstance(temp_prices_sl, pd.DataFrame):
                temp_prices_sl = temp_prices_sl.squeeze()
            if not isinstance(temp_prices_sl, pd.Series):
                temp_prices_sl = pd.Series([temp_prices_sl], index=[current_universe_tickers[0]]) if not current_universe_tickers.empty else pd.Series(dtype=float)
            current_prices_for_sl_check = temp_prices_sl.reindex(current_universe_tickers).fillna(np.nan)
        else:
            current_prices_for_sl_check = pd.Series(np.nan, index=current_universe_tickers)

        stop_levels = sl_handler.calculate_stop_levels(current_date, asset_ohlc_hist_for_sl, self.w_prev, self.entry_prices)
        weights_after_sl = sl_handler.apply_stop_loss(current_date, current_prices_for_sl_check, weights_at_current_date, self.entry_prices, stop_levels)

        # Vectorized exit update after stop loss
        entry = np.asarray(self.entry_prices).copy()
        before_sl = np.asarray(weights_at_current_date)
        after_sl = np.asarray(weights_after_sl)
        exit_mask = (before_sl != 0) & (after_sl == 0)
        entry[exit_mask] = np.nan
        self.entry_prices = pd.Series(entry, index=current_universe_tickers)
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
                elif not current_benchmark_price.empty and not current_benchmark_sma.empty:
                    def to_scalar(x):
                        if hasattr(x, 'iloc'):
                            return float(x.iloc[0])
                        return float(x)
                    if to_scalar(current_benchmark_price_raw) < to_scalar(current_benchmark_sma_raw):
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

        # Vectorized exit update after risk filters
        entry = np.asarray(self.entry_prices).copy()
        before_risk = np.asarray(weights_at_current_date)
        after_risk = np.asarray(final_weights)
        exit_mask = (before_risk != 0) & (after_risk == 0)
        entry[exit_mask] = np.nan
        self.entry_prices = pd.Series(entry, index=current_universe_tickers)

        self.w_prev = final_weights

        output_weights_df = pd.DataFrame(0.0, index=[current_date], columns=current_universe_tickers)
        output_weights_df.loc[current_date, :] = final_weights
        return output_weights_df


    def _get_params(self) -> Dict[str, Any]:
        return self.strategy_config.get("strategy_params", self.strategy_config)