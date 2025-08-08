from typing import Any, Dict, cast, Optional
import pandas as pd
import numpy as np

from .base_momentum_portfolio_strategy import BaseMomentumPortfolioStrategy
from ...features.dp_vams import DPVAMS
from ...utils.price_data_utils import extract_current_prices, normalize_price_series_to_dataframe


class VamsMomentumPortfolioStrategy(BaseMomentumPortfolioStrategy):
    """Momentum strategy implementation using Volatility Adjusted Momentum Scores (VAMS)."""

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)

        params_dict_to_update = self.strategy_config.get("strategy_params", {})
        vams_defaults = {
            "lookback_months": 6,
            "alpha": 0.5,  # for DPVAMS
        }
        for k, v in vams_defaults.items():
            params_dict_to_update.setdefault(k, v)

    @classmethod
    def tunable_parameters(cls) -> dict[str, dict[str, Any]]:
        return {
            param: {"type": "float", "min": 0, "max": 1}
            for param in [
                "num_holdings",
                "lookback_months",
                "alpha",
                "sma_filter_window",
                "apply_trading_lag",
                "top_decile_fraction",
                "smoothing_lambda",
                "leverage",
                "trade_longs",
                "trade_shorts",
                "derisk_days_under_sma",
            ]
        }

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods for VAMSMomentumPortfolioStrategy.
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

        # Add 2-month buffer for reliable calculations
        return int(total_requirement + 2)

    def _calculate_scores(
        self,
        asset_prices: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        """
        Calculates Volatility Adjusted Momentum Scores (VAMS) for assets as of current_date.
        This implementation uses the optimized DPVAMS (Downside Penalized VAMS) feature.
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        lookback_months = params.get("lookback_months", 6)
        alpha = params.get("alpha", 0.5)

        relevant_prices = asset_prices[asset_prices.index <= current_date]

        if relevant_prices.empty:
            return pd.Series(dtype=float, index=asset_prices.columns)

        # Use the optimized DP-VAMS feature for consistent, reliable calculations
        dpvams_feature = DPVAMS(lookback_months=lookback_months, alpha=alpha)
        dpvams_scores = dpvams_feature.compute(relevant_prices)

        # Return the scores for the current date
        if current_date in dpvams_scores.index:
            return cast(pd.Series, dpvams_scores.loc[current_date])
        else:
            # If current_date not in index, return the last available scores
            return (
                dpvams_scores.iloc[-1]
                if not dpvams_scores.empty
                else pd.Series(0.0, index=asset_prices.columns)
            )

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,  # Universe asset data (OHLCV)
        benchmark_historical_data: pd.DataFrame,  # Benchmark asset data (OHLCV)
        non_universe_historical_data: Optional[pd.DataFrame] = None,  # Optional, for compatibility
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generates trading signals for the VAMSMomentumPortfolioStrategy.

        Parameters:
            all_historical_data: pd.DataFrame
            benchmark_historical_data: pd.DataFrame
            non_universe_historical_data: pd.DataFrame (optional, unused)
            current_date: pd.Timestamp
            start_date: Optional[pd.Timestamp]
            end_date: Optional[pd.Timestamp]
        """

        # --- Data Sufficiency Validation ---
        if current_date is None:
            # Handle None current_date gracefully - use the last date in the data
            current_date = all_historical_data.index[-1]
        is_sufficient, reason = self.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, cast(pd.Timestamp, current_date)
        )
        if not is_sufficient:
            columns = (
                all_historical_data.columns.get_level_values(0).unique()
                if isinstance(all_historical_data.columns, pd.MultiIndex)
                else all_historical_data.columns
            )
            return pd.DataFrame(0.0, index=[current_date], columns=columns)

        params = self.strategy_config.get("strategy_params", self.strategy_config)
        price_col_asset = params["price_column_asset"]
        price_col_benchmark = params["price_column_benchmark"]

        # --- Date Window Filtering ---
        if (start_date is not None and cast(pd.Timestamp, current_date) < start_date) or (
            end_date is not None and cast(pd.Timestamp, current_date) > end_date
        ):
            columns = (
                all_historical_data.columns.get_level_values(0).unique()
                if isinstance(all_historical_data.columns, pd.MultiIndex)
                else all_historical_data.columns
            )
            return pd.DataFrame(0.0, index=[current_date], columns=columns)

        # --- Prepare Data ---
        # Extract price data using polymorphic approach
        if (
            isinstance(all_historical_data.columns, pd.MultiIndex)
            and "Ticker" in all_historical_data.columns.names
        ):
            # Extract the specific price column from MultiIndex data
            asset_prices_for_vams = all_historical_data.xs(price_col_asset, level="Field", axis=1)
        else:
            # Use the data directly if not MultiIndex
            asset_prices_for_vams = all_historical_data

        # Ensure we have a DataFrame (normalize if needed)
        asset_prices_for_vams = normalize_price_series_to_dataframe(asset_prices_for_vams)

        asset_prices_hist = asset_prices_for_vams[asset_prices_for_vams.index <= current_date]

        benchmark_prices_hist = benchmark_historical_data[
            benchmark_historical_data.index <= current_date
        ]

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
        scores_at_current_date = self._calculate_scores(
            asset_prices_hist,
            cast(pd.Timestamp, current_date),
        )

        if scores_at_current_date.isna().all() or scores_at_current_date.empty:
            weights_at_current_date = self.w_prev.copy()
        else:
            cand_weights = self._calculate_candidate_weights(scores_at_current_date)
            w_target_pre_filter = self._apply_leverage_and_smoothing(cand_weights, self.w_prev)

            # Use polymorphic price extraction interface for entry prices
            current_prices_for_assets_at_date = extract_current_prices(
                asset_prices_for_vams, cast(pd.Timestamp, current_date), current_universe_tickers
            )

            prev = np.asarray(self.w_prev)
            target = np.asarray(w_target_pre_filter)
            entry = np.asarray(self.entry_prices).copy()
            prices = np.asarray(current_prices_for_assets_at_date)

            entry_mask = ((prev == 0) & (target != 0)) | (
                (np.sign(prev) != np.sign(target)) & (target != 0)
            )
            exit_mask = target == 0
            entry[entry_mask] = prices[entry_mask]
            entry[exit_mask] = np.nan
            self.entry_prices = pd.Series(entry, index=current_universe_tickers)
            weights_at_current_date = w_target_pre_filter

        # --- Apply Stop Loss ---
        sl_handler = self.get_stop_loss_handler()
        asset_ohlc_hist_for_sl = all_historical_data[all_historical_data.index <= current_date]
        # Use polymorphic price extraction for stop loss
        current_prices_for_sl_check = extract_current_prices(
            asset_prices_for_vams, cast(pd.Timestamp, current_date), current_universe_tickers
        )

        stop_levels = sl_handler.calculate_stop_levels(
            cast(pd.Timestamp, current_date), asset_ohlc_hist_for_sl, self.w_prev, self.entry_prices
        )
        weights_after_sl = sl_handler.apply_stop_loss(
            cast(pd.Timestamp, current_date),
            current_prices_for_sl_check,
            weights_at_current_date,
            self.entry_prices,
            stop_levels,
        )

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
            # Explicitly re-declare benchmark_prices_hist to help Pylance with scope
            benchmark_prices_hist = benchmark_historical_data[
                benchmark_historical_data.index <= current_date
            ]

            benchmark_price_series_for_sma = (
                benchmark_prices_hist.xs(price_col_benchmark, level="Field", axis=1)
                if isinstance(benchmark_prices_hist.columns, pd.MultiIndex)
                else benchmark_prices_hist[price_col_benchmark]
            )
            benchmark_sma = self._calculate_benchmark_sma(
                benchmark_prices_hist, sma_filter_window, price_col_benchmark
            )

            if (
                current_date in benchmark_price_series_for_sma.index
                and current_date in benchmark_sma.index
            ):
                current_benchmark_price_raw = benchmark_price_series_for_sma.loc[
                    cast(pd.Timestamp, current_date)
                ]
                current_benchmark_sma_raw = benchmark_sma.loc[cast(pd.Timestamp, current_date)]
                # Extract scalar if Series
                if isinstance(current_benchmark_price_raw, pd.Series):
                    current_benchmark_price_scalar = current_benchmark_price_raw.iloc[0]
                elif isinstance(current_benchmark_price_raw, pd.DataFrame):
                    current_benchmark_price_scalar = current_benchmark_price_raw.iloc[0, 0]
                else:
                    current_benchmark_price_scalar = float(current_benchmark_price_raw)
                if isinstance(current_benchmark_sma_raw, pd.Series):
                    current_benchmark_sma_scalar = current_benchmark_sma_raw.iloc[0]
                elif isinstance(current_benchmark_sma_raw, pd.DataFrame):
                    current_benchmark_sma_scalar = current_benchmark_sma_raw.iloc[0, 0]
                else:
                    current_benchmark_sma_scalar = float(current_benchmark_sma_raw)
                current_benchmark_price: pd.Series = pd.Series(
                    [current_benchmark_price_scalar], index=[current_date]
                )
                current_benchmark_sma: pd.Series = pd.Series(
                    [current_benchmark_sma_scalar], index=[current_date]
                )

                derisk_periods = params.get("derisk_days_under_sma", 10)

                self.current_derisk_flag, self.consecutive_periods_under_sma = (
                    self._calculate_derisk_flags(
                        current_benchmark_price,
                        current_benchmark_sma,
                        derisk_periods,
                        self.current_derisk_flag,
                        self.consecutive_periods_under_sma,
                    )
                )

                if self.current_derisk_flag:
                    final_weights[:] = 0.0
                elif (
                    not current_benchmark_price.empty
                    and not current_benchmark_sma.empty
                    and float(cast(float, current_benchmark_price_scalar))
                    < float(cast(float, current_benchmark_sma_scalar))
                ):
                    final_weights[:] = 0.0

        roro_signal_instance = self.get_roro_signal()
        if roro_signal_instance:
            roro_output = roro_signal_instance.generate_signal(
                all_historical_data[all_historical_data.index <= current_date],
                benchmark_historical_data[benchmark_historical_data.index <= current_date],
                cast(pd.Timestamp, current_date),
            )

            is_roro_risk_off = False
            if isinstance(roro_output, (pd.Series, pd.DataFrame)):
                if not roro_output.index.empty and current_date in roro_output.index:
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

        output_weights_df = pd.DataFrame(
            0.0, index=[current_date], columns=current_universe_tickers
        )
        output_weights_df.loc[cast(pd.Timestamp, current_date), :] = final_weights
        return output_weights_df

    def _get_params(self) -> Dict[str, Any]:
        return cast(
            Dict[str, Any], self.strategy_config.get("strategy_params", self.strategy_config)
        )
