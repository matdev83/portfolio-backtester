from ..candidate_weights import default_candidate_weights
from ..leverage_and_smoothing import apply_leverage_and_smoothing
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np # Added for np.nan
from ...features.calmar_ratio import CalmarRatio # Updated import

from ..base.portfolio_strategy import PortfolioStrategy


class CalmarMomentumStrategy(PortfolioStrategy):
    def _calculate_candidate_weights(self, scores: pd.Series) -> pd.Series:
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        return default_candidate_weights(scores, params)

    def _apply_leverage_and_smoothing(self, candidate_weights: pd.Series, prev_weights: Optional[pd.Series]) -> pd.Series:
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        return apply_leverage_and_smoothing(candidate_weights, prev_weights, params)
    """Momentum strategy implementation using Calmar ratio for ranking."""

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)

        self.w_prev: Optional[pd.Series] = None # Initialize w_prev
        self.current_derisk_flag: bool = False # Initialize derisk flag
        self.consecutive_periods_under_sma: int = 0 # Initialize consecutive periods

        params_dict_to_update = self.strategy_config
        if "strategy_params" in self.strategy_config:
            if self.strategy_config["strategy_params"] is None:
                 self.strategy_config["strategy_params"] = {}
            params_dict_to_update = self.strategy_config["strategy_params"]

        params_dict_to_update.setdefault("apply_trading_lag", False)
        params_dict_to_update.setdefault("rolling_window", 6) # Default for Calmar
        params_dict_to_update.setdefault("sma_filter_window", None) # Ensure SMA filter is initialized
        params_dict_to_update.setdefault("price_column_asset", "Close")
        params_dict_to_update.setdefault("price_column_benchmark", "Close")

        # Instantiate CalmarRatio feature
        self.calmar_feature = CalmarRatio(
            rolling_window=params_dict_to_update["rolling_window"]
        )

        self.entry_prices: pd.Series | None = None # Initialize entry_prices

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {
            "num_holdings", "rolling_window", "sma_filter_window", "apply_trading_lag",
            "lookback_months", "top_decile_fraction", "smoothing_lambda", "leverage", "long_only"
        }

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods for CalmarMomentumStrategy.
        Requires: rolling_window for Calmar ratio calculation + SMA filter window
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        
        # Calmar ratio rolling window requirement
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
        all_historical_data: pd.DataFrame, # Universe asset data (OHLCV)
        benchmark_historical_data: pd.DataFrame, # Benchmark asset data (OHLCV)
        non_universe_historical_data: Optional[pd.DataFrame] = None, # Optional, for compatibility
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generates trading signals based on historical data and current date.
        
        Parameters:
            all_historical_data: pd.DataFrame
            benchmark_historical_data: pd.DataFrame
            non_universe_historical_data: pd.DataFrame (optional, unused)
            current_date: pd.Timestamp
            start_date: Optional[pd.Timestamp]
            end_date: Optional[pd.Timestamp]
        """
        # Handle None current_date gracefully - use the last date in the data
        if current_date is None:
            current_date = all_historical_data.index[-1]
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
        if (start_date is not None and current_date < start_date) or (end_date is not None and current_date > end_date):
            columns = all_historical_data.columns.get_level_values(0).unique() if isinstance(all_historical_data.columns, pd.MultiIndex) else all_historical_data.columns
            return pd.DataFrame(0.0, index=[current_date], columns=columns)

        # --- Prepare Data ---
        if isinstance(all_historical_data.columns, pd.MultiIndex) and 'Field' in all_historical_data.columns.names:
            asset_prices_for_calmar = all_historical_data.xs(price_col_asset, level='Field', axis=1)
        else:
            asset_prices_for_calmar = all_historical_data.to_frame() if isinstance(all_historical_data, pd.Series) else all_historical_data

        asset_prices_hist = asset_prices_for_calmar[asset_prices_for_calmar.index <= current_date]
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

        # --- Calculate Scores (Calmar Ratio) ---
        scores = self.calmar_feature.compute(asset_prices_hist)
        scores_at_current_date = scores.loc[current_date] if current_date in scores.index else pd.Series(dtype=float)

        # Fix for Pylance error: Invalid conditional operand
        if bool(scores_at_current_date.isna().all()) or scores_at_current_date.empty:
            weights_at_current_date = self.w_prev.copy()
        else:
            if isinstance(scores_at_current_date, pd.DataFrame):
                scores_at_current_date = scores_at_current_date.squeeze()
            if not isinstance(scores_at_current_date, pd.Series):
                scores_at_current_date = pd.Series([scores_at_current_date], index=[current_universe_tickers[0]]) if not current_universe_tickers.empty else pd.Series(dtype=float)

            cand_weights = self._calculate_candidate_weights(scores_at_current_date)
            w_target_pre_filter = self._apply_leverage_and_smoothing(cand_weights, self.w_prev)

            # Vectorized entry/exit logic
            if current_date in asset_prices_for_calmar.index:
                temp_prices = asset_prices_for_calmar.loc[current_date]
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

            entry_mask = ((prev == 0) & (target != 0)) | ((np.sign(prev) != np.sign(target)) & (target != 0))
            exit_mask = (target == 0)
            entry[entry_mask] = prices[entry_mask]
            entry[exit_mask] = np.nan
            self.entry_prices = pd.Series(entry, index=current_universe_tickers)
            weights_at_current_date = w_target_pre_filter

        # --- Apply Stop Loss ---
        sl_handler = self.get_stop_loss_handler()
        asset_ohlc_hist_for_sl = all_historical_data[all_historical_data.index <= current_date]
        # Get current prices for stop loss check using utility function
        from ...utils.price_data_utils import extract_current_prices
        # Ensure asset_prices_for_calmar is always a DataFrame
        if isinstance(asset_prices_for_calmar, pd.Series):
            asset_prices_for_calmar_df = asset_prices_for_calmar.to_frame()
        else:
            asset_prices_for_calmar_df = asset_prices_for_calmar
        current_prices_for_sl_check = extract_current_prices(
            asset_prices_for_calmar_df, current_date, current_universe_tickers
        )

        # Vectorized stop loss logic
        stop_levels = sl_handler.calculate_stop_levels(current_date, asset_ohlc_hist_for_sl, self.w_prev, self.entry_prices)
        weights_after_sl = sl_handler.apply_stop_loss(current_date, current_prices_for_sl_check, weights_at_current_date, self.entry_prices, stop_levels)

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
                # Extract scalar if Series/DataFrame
                if isinstance(current_benchmark_price_raw, pd.Series):
                    current_benchmark_price_scalar = float(current_benchmark_price_raw.iloc[0])
                elif isinstance(current_benchmark_price_raw, pd.DataFrame):
                    current_benchmark_price_scalar = float(current_benchmark_price_raw.values[0, 0])
                else:
                    current_benchmark_price_scalar = float(current_benchmark_price_raw)
                if isinstance(current_benchmark_sma_raw, pd.Series):
                    current_benchmark_sma_scalar = float(current_benchmark_sma_raw.iloc[0])
                elif isinstance(current_benchmark_sma_raw, pd.DataFrame):
                    current_benchmark_sma_scalar = float(current_benchmark_sma_raw.values[0, 0])
                else:
                    current_benchmark_sma_scalar = float(current_benchmark_sma_raw)
                current_benchmark_price: pd.Series = pd.Series([current_benchmark_price_scalar], index=[current_date])
                current_benchmark_sma: pd.Series = pd.Series([current_benchmark_sma_scalar], index=[current_date])

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
                     float(current_benchmark_price_scalar) < float(current_benchmark_sma_scalar):
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

        entry = np.asarray(self.entry_prices).copy()
        before_risk = np.asarray(weights_at_current_date)
        after_risk = np.asarray(final_weights)
        exit_mask = (before_risk != 0) & (after_risk == 0)
        entry[exit_mask] = np.nan
        self.entry_prices = pd.Series(entry, index=current_universe_tickers)

        self.w_prev = final_weights

        output_weights_df = pd.DataFrame(0.0, index=[current_date], columns=current_universe_tickers)
        output_weights_df.loc[current_date, :] = final_weights

        if self.strategy_config.get("apply_trading_lag", False):
            return output_weights_df.shift(1)
        return output_weights_df