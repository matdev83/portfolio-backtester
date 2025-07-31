from ..candidate_weights import default_candidate_weights
from ..leverage_and_smoothing import apply_leverage_and_smoothing
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..base.portfolio_strategy import PortfolioStrategy

# Import Numba optimization with fallback
try:
    from ...numba_optimized import rolling_sharpe_fast_portfolio
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Removed SharpeSignalGenerator and feature imports

import logging

class SharpeMomentumStrategy(PortfolioStrategy):
    """Strategy that uses Sharpe ratio for ranking assets."""

    def _calculate_candidate_weights(self, scores: pd.Series) -> pd.Series:
        # Build params dict with required keys for candidate weights logic
        config = self.strategy_config.get("strategy_params", self.strategy_config)
        params = {
            "num_holdings": config.get("num_holdings"),
            "top_decile_fraction": config.get("top_decile_fraction", 0.1),
            "long_only": config.get("long_only", True)
        }
        logger = logging.getLogger(__name__)
        logger.debug(f"[SharpeMomentumStrategy] _calculate_candidate_weights: scores={scores.to_dict()} params={params}")
        if scores is None or len(scores) == 0:
            logger.debug("[SharpeMomentumStrategy] No scores, returning zeros.")
            return pd.Series(0.0, index=getattr(scores, 'index', None))
        cand = default_candidate_weights(scores, params)
        # FINAL FALLBACK: If all candidate weights are zero and num_holdings==1, assign 1.0 to the top asset
        if params.get("num_holdings", None) == 1 and (cand.abs().sum() == 0 or cand.isna().all()):
            top_asset = scores.sort_values(ascending=False).index[0] if len(scores) > 0 else None
            if top_asset is not None:
                cand[:] = 0.0
                cand[top_asset] = 1.0
                logger.debug(f"[SharpeMomentumStrategy] FINAL FALLBACK: For num_holdings=1, forced 1.0 weight to {top_asset}")
        logger.debug(f"[SharpeMomentumStrategy] candidate_weights={cand.to_dict()}")
        return cand

    def _apply_leverage_and_smoothing(self, candidate_weights: pd.Series, prev_weights: Optional[pd.Series]) -> pd.Series:
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        logger = logging.getLogger(__name__)
        logger.debug(f"[SharpeMomentumStrategy] _apply_leverage_and_smoothing: candidate_weights={candidate_weights.to_dict()} prev_weights={None if prev_weights is None else prev_weights.to_dict()} params={params}")
        result = apply_leverage_and_smoothing(candidate_weights, prev_weights, params)
        logger.debug(f"[SharpeMomentumStrategy] after leverage/smoothing: {result.to_dict()}")
        return result
    # Removed signal_generator_class

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)
        # Instance variables
        self.w_prev = None  # type: Optional[pd.Series]
        self.current_derisk_flag = False  # type: bool
        self.consecutive_periods_under_sma = 0  # type: int

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

        self.entry_prices = None  # type: Optional[pd.Series]

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
                from ...numba_optimized import sharpe_fast
                sharpe_mat = sharpe_fast(returns_np, window_days, annualization_factor=252.0)
                cols = getattr(daily_returns, 'columns', None)
                if isinstance(cols, (pd.Index, list)):
                    sharpe_ratio_calculated = pd.DataFrame(sharpe_mat, index=daily_returns.index, columns=cols)
                else:
                    sharpe_ratio_calculated = pd.DataFrame(sharpe_mat, index=daily_returns.index)
            except ImportError:
                # Fallback to per-asset fast function
                from ...numba_optimized import rolling_sharpe_fast_portfolio
                sharpe_ratio_calculated = pd.DataFrame(index=daily_closes.index, columns=daily_closes.columns)
                for asset in daily_closes.columns:
                    sharpe_ratio_calculated[asset] = rolling_sharpe_fast_portfolio(daily_closes[asset].values, rolling_window_months, annualization_factor)
        else:
            # Resample to monthly, calculate monthly returns
            monthly_closes = daily_closes.resample('ME').last()
            # Always return a Series
            if isinstance(latest_sharpe, pd.DataFrame):
                latest_sharpe = latest_sharpe.squeeze(axis=0)
            if not isinstance(latest_sharpe, pd.Series):
                latest_sharpe = pd.Series(latest_sharpe, index=daily_closes.columns)
            return latest_sharpe.fillna(0.0)
            if len(relevant_monthly_rets) < rolling_window_months:
                return pd.Series(0.0, index=daily_closes.columns) # Not enough history for any asset

            rolling_mean = relevant_monthly_rets.rolling(window=rolling_window_months).mean()
            rolling_std = relevant_monthly_rets.rolling(window=rolling_window_months).std()

            # Avoid division by zero; replace zero std with NaN, then fill resulting NaN Sharpe with 0
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
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs,
    ) -> pd.DataFrame:
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

        params = self._get_params()
        price_col_asset = params["price_column_asset"]
        price_col_benchmark = params["price_column_benchmark"]

        if start_date is not None and current_date is not None and current_date < start_date:
            return pd.DataFrame(index=[current_date], columns=all_historical_data.columns.get_level_values(0).unique() if isinstance(all_historical_data.columns, pd.MultiIndex) else all_historical_data.columns).fillna(0.0)
        if end_date is not None and current_date is not None and current_date > end_date:
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
            if current_date is not None and current_date in asset_closes_hist.index:
                temp_prices = asset_closes_hist.loc[current_date]
                if isinstance(temp_prices, pd.DataFrame):
                    temp_prices = temp_prices.squeeze()
                if not isinstance(temp_prices, pd.Series):
                    temp_prices = pd.Series([temp_prices], index=[current_universe_tickers[0]]) if len(current_universe_tickers) > 0 else pd.Series(dtype=float)
                current_prices_for_assets_at_date = temp_prices.reindex(current_universe_tickers).fillna(np.nan)
            else:
                current_prices_for_assets_at_date = pd.Series(np.nan, index=current_universe_tickers)

            # Vectorized entry/exit logic
            prev = np.asarray([self.w_prev.get(asset, 0) for asset in current_universe_tickers])
            target = np.asarray([w_target_pre_filter.get(asset, 0) for asset in current_universe_tickers])
            entry = np.asarray([self.entry_prices.get(asset, np.nan) for asset in current_universe_tickers])
            prices = np.asarray([current_prices_for_assets_at_date.get(asset, np.nan) for asset in current_universe_tickers])

            entry_mask = ((prev == 0) & (target != 0)) | ((np.sign(prev) != np.sign(target)) & (target != 0))
            exit_mask = (target == 0)
            entry[entry_mask] = prices[entry_mask]
            entry[exit_mask] = np.nan
            self.entry_prices = pd.Series(entry, index=current_universe_tickers)
            weights_at_current_date = w_target_pre_filter

        # --- Apply Stop Loss ---
        sl_handler = self.get_stop_loss_handler()
        asset_ohlc_hist_for_sl = all_historical_data[all_historical_data.index <= current_date]
        if current_date is not None and current_date in asset_data_for_scores.index:
            temp_prices = asset_data_for_scores.loc[current_date]
            if isinstance(temp_prices, pd.DataFrame):
                temp_prices = temp_prices.squeeze()
            if not isinstance(temp_prices, pd.Series):
                temp_prices = pd.Series([temp_prices], index=[current_universe_tickers[0]]) if len(current_universe_tickers) > 0 else pd.Series(dtype=float)
            current_prices_for_sl_check = temp_prices.reindex(current_universe_tickers).fillna(np.nan)
        else:
            current_prices_for_sl_check = pd.Series(np.nan, index=current_universe_tickers)

        stop_levels = sl_handler.calculate_stop_levels(current_date, asset_ohlc_hist_for_sl, self.w_prev, self.entry_prices)
        weights_after_sl = sl_handler.apply_stop_loss(current_date, current_prices_for_sl_check, weights_at_current_date, self.entry_prices, stop_levels)

        # Vectorized stop loss exit logic
        before_sl = np.asarray([weights_at_current_date.get(asset, 0) for asset in current_universe_tickers])
        after_sl = np.asarray([weights_after_sl.get(asset, 0) for asset in current_universe_tickers])
        entry = np.asarray([self.entry_prices.get(asset, np.nan) for asset in current_universe_tickers])
        exit_mask = (before_sl != 0) & (after_sl == 0)
        entry[exit_mask] = np.nan
        self.entry_prices = pd.Series(entry, index=current_universe_tickers)
        weights_at_current_date = weights_after_sl


        # --- Apply Risk Filters (SMA, RoRo) ---
        final_weights = weights_at_current_date
        benchmark_prices_hist = benchmark_historical_data[benchmark_historical_data.index <= current_date]

        sma_filter_window = params.get("sma_filter_window")
        logger = logging.getLogger(__name__)
        if sma_filter_window and sma_filter_window > 0:
            benchmark_price_series_for_sma = benchmark_prices_hist.xs(price_col_benchmark, level='Field', axis=1) if isinstance(benchmark_prices_hist.columns, pd.MultiIndex) else benchmark_prices_hist[price_col_benchmark]
            benchmark_sma_series = self._calculate_benchmark_sma(benchmark_prices_hist, sma_filter_window, price_col_benchmark)

            if current_date in benchmark_price_series_for_sma.index and current_date in benchmark_sma_series.index:
                current_benchmark_price_val = benchmark_price_series_for_sma.loc[current_date]
                current_benchmark_sma_val = benchmark_sma_series.loc[current_date]
                # Convert to float scalars for logic
                def to_scalar(val):
                    if isinstance(val, pd.Series):
                        if val.empty:
                            return np.nan
                        return float(val.iloc[0])
                    elif isinstance(val, pd.DataFrame):
                        if val.empty:
                            return np.nan
                        return float(val.values[0, 0])
                    try:
                        return float(val)
                    except Exception:
                        return np.nan
                price_scalar = to_scalar(current_benchmark_price_val)
                sma_scalar = to_scalar(current_benchmark_sma_val)
                derisk_periods = params.get("derisk_days_under_sma", 10)
                # Pass as Series of length 1 to _calculate_derisk_flags
                price_series = pd.Series([price_scalar], index=[current_date])
                sma_series = pd.Series([sma_scalar], index=[current_date])
                self.current_derisk_flag, self.consecutive_periods_under_sma = \
                    self._calculate_derisk_flags(price_series, sma_series, derisk_periods, self.current_derisk_flag, self.consecutive_periods_under_sma)
                if self.current_derisk_flag or (not np.isnan(price_scalar) and not np.isnan(sma_scalar) and price_scalar < sma_scalar):
                    logger.debug(f"[SharpeMomentumStrategy] SMA filter zeroed weights on {current_date}")
                    final_weights[:] = 0.0

        roro_signal_instance = self.get_roro_signal()
        if roro_signal_instance:
            roro_signal = roro_signal_instance.generate_signal(all_historical_data, benchmark_historical_data, current_date)
            logger.debug(f"[SharpeMomentumStrategy] RoRo signal for {current_date}: {roro_signal}")
            is_roro_risk_off = not roro_signal
            if is_roro_risk_off:
                logger.debug(f"[SharpeMomentumStrategy] RoRo filter zeroed weights on {current_date}")
                final_weights[:] = 0.0

        # Vectorized risk filter exit logic
        before_risk = np.asarray([weights_at_current_date.get(asset, 0) for asset in current_universe_tickers])
        after_risk = np.asarray([final_weights.get(asset, 0) for asset in current_universe_tickers])
        entry = np.asarray([self.entry_prices.get(asset, np.nan) for asset in current_universe_tickers])
        exit_mask = (before_risk != 0) & (after_risk == 0)
        entry[exit_mask] = np.nan
        self.entry_prices = pd.Series(entry, index=current_universe_tickers)

        # PERFORMANCE OPTIMIZATION: Store reference, copy only if strategy modifies weights later


        self.w_prev = final_weights

        output_weights_df = pd.DataFrame(0.0, index=[current_date], columns=current_universe_tickers)
        output_weights_df.loc[current_date, :] = final_weights
        return output_weights_df

    def _get_params(self) -> Dict[str, Any]:
        return self.strategy_config.get("strategy_params", self.strategy_config)