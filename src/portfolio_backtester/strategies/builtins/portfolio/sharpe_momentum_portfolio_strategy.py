from __future__ import annotations

from typing import Any, Dict, Optional, cast, Union

import numpy as np
import pandas as pd
import logging

from .base_momentum_portfolio_strategy import BaseMomentumPortfolioStrategy
from portfolio_backtester.numba_optimized import sharpe_fast_fixed
from portfolio_backtester.utils.price_data_utils import (
    normalize_price_series_to_dataframe,
    extract_current_prices,
)

logger = logging.getLogger(__name__)


class SharpeMomentumPortfolioStrategy(BaseMomentumPortfolioStrategy):
    """Strategy that uses Sharpe ratio for ranking assets."""

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)

        params_dict_to_update = self.strategy_config.get("strategy_params", {})
        sharpe_defaults = {
            "rolling_window": 6,  # Months for Sharpe ratio calculation
            "annualization_factor": 12,  # For monthly returns
        }
        for k, v in sharpe_defaults.items():
            params_dict_to_update.setdefault(k, v)

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            # Core momentum parameters
            "rolling_window": {"type": "int", "min": 1, "max": 24, "default": 6},
            "num_holdings": {"type": "int", "min": 1, "max": 100, "default": 10},
            "top_decile_fraction": {
                "type": "float",
                "min": 0.05,
                "max": 0.5,
                "default": 0.1,
            },
            # Smoothing and leverage
            "smoothing_lambda": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "leverage": {"type": "float", "min": 1.0, "max": 2.0, "default": 1.0},
            # Trading toggles
            "trade_longs": {"type": "bool", "default": True},
            "trade_shorts": {"type": "bool", "default": False},
            # Risk filters
            "sma_filter_window": {"type": "int", "min": 0, "max": 252, "default": 20},
            "derisk_days_under_sma": {"type": "int", "min": 1, "max": 63, "default": 10},
            "apply_trading_lag": {"type": "bool", "default": False},
        }

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods for SharpeMomentumPortfolioStrategy.
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
        return int(total_requirement + 2)

    # Removed get_required_features

    def _calculate_scores(
        self,
        asset_prices: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        """
        Calculates Sharpe ratio scores for assets as of current_date.
        asset_prices: DataFrame with historical prices, indexed by date, columns by asset.
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        rolling_window_months = params.get("rolling_window", 6)

        if asset_prices.empty:
            return pd.Series(dtype=float)

        # Ensure we have DataFrame format
        daily_closes = normalize_price_series_to_dataframe(asset_prices)

        # Filter up to current_date
        daily_closes = daily_closes[daily_closes.index <= current_date]

        if daily_closes.empty:
            return pd.Series(dtype=float, index=asset_prices.columns)

        # We need the Sharpe ratio as of the month-end corresponding to current_date
        current_month_end = current_date.to_period("M").to_timestamp("M")

        # Use optimized Sharpe calculation with proper ddof=1 handling
        daily_returns = daily_closes.pct_change(fill_method=None).fillna(0)
        if daily_returns.empty:
            return pd.Series(0.0, index=daily_closes.columns)

        returns_np = daily_returns.to_numpy(dtype=np.float64)
        window_days = rolling_window_months * 21  # Approximate trading days per month

        sharpe_mat = sharpe_fast_fixed(returns_np, window_days, annualization_factor=252.0)
        sharpe_ratio_calculated = pd.DataFrame(
            sharpe_mat, index=daily_returns.index, columns=daily_returns.columns
        )

        # Get the latest calculated Sharpe ratio (for the current_month_end)
        if not sharpe_ratio_calculated.empty and current_month_end in sharpe_ratio_calculated.index:
            latest_sharpe = sharpe_ratio_calculated.loc[current_month_end].fillna(0.0)
            if isinstance(latest_sharpe, pd.DataFrame):
                latest_sharpe = cast(pd.Series, latest_sharpe.squeeze())
            return latest_sharpe
        else:  # Not enough rolling data, or current_month_end not in index
            return pd.Series(0.0, index=daily_closes.columns)

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if current_date is None:
            current_date = pd.Timestamp(all_historical_data.index[-1])

        is_sufficient, reason = self.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )
        if not is_sufficient:
            # Log the reason to aid debugging and satisfy static analysis
            logger.debug(
                "Insufficient data for %s at %s: %s",
                self.__class__.__name__,
                current_date,
                reason,
            )
            columns = (
                all_historical_data.columns.get_level_values(0).unique()
                if isinstance(all_historical_data.columns, pd.MultiIndex)
                else all_historical_data.columns
            )
            return pd.DataFrame(0.0, index=[current_date], columns=columns)

        params = self._get_params()
        price_col_asset = params["price_column_asset"]
        price_col_benchmark = params["price_column_benchmark"]

        if start_date is not None and current_date is not None and current_date < start_date:
            return pd.DataFrame(
                index=[current_date],
                columns=(
                    all_historical_data.columns.get_level_values(0).unique()
                    if isinstance(all_historical_data.columns, pd.MultiIndex)
                    else all_historical_data.columns
                ),
            ).fillna(0.0)
        if end_date is not None and current_date is not None and current_date > end_date:
            return pd.DataFrame(
                index=[current_date],
                columns=(
                    all_historical_data.columns.get_level_values(0).unique()
                    if isinstance(all_historical_data.columns, pd.MultiIndex)
                    else all_historical_data.columns
                ),
            ).fillna(0.0)

        if (
            isinstance(all_historical_data.columns, pd.MultiIndex)
            and "Ticker" in all_historical_data.columns.names
        ):
            current_universe_tickers = all_historical_data.columns.get_level_values(
                "Ticker"
            ).unique()
            asset_data_for_scores = all_historical_data.xs(price_col_asset, level="Field", axis=1)
        else:
            current_universe_tickers = all_historical_data.columns
            asset_data_for_scores = all_historical_data

        asset_data_for_scores = normalize_price_series_to_dataframe(asset_data_for_scores)

        if self.w_prev is None:
            self.w_prev = pd.Series(0.0, index=current_universe_tickers)
        else:
            self.w_prev = self.w_prev.reindex(current_universe_tickers).fillna(0.0)

        if self.entry_prices is None:
            self.entry_prices = pd.Series(np.nan, index=current_universe_tickers)
        else:
            self.entry_prices = self.entry_prices.reindex(current_universe_tickers).fillna(np.nan)

        scores_at_current_date = self._calculate_scores(all_historical_data, current_date)

        if scores_at_current_date.isna().all() or scores_at_current_date.empty:
            weights_at_current_date = self.w_prev.copy()
        else:
            cand_weights = self._calculate_candidate_weights(scores_at_current_date)
            w_target_pre_filter = self._apply_leverage_and_smoothing(cand_weights, self.w_prev)

            asset_closes_hist = asset_data_for_scores[asset_data_for_scores.index <= current_date]
            current_prices_for_assets_at_date = extract_current_prices(
                asset_closes_hist, current_date, current_universe_tickers
            )

            prev = np.asarray([self.w_prev.get(asset, 0) for asset in current_universe_tickers])
            target = np.asarray(
                [w_target_pre_filter.get(asset, 0) for asset in current_universe_tickers]
            )
            entry = np.asarray(
                [self.entry_prices.get(asset, np.nan) for asset in current_universe_tickers]
            )
            prices = np.asarray(
                [
                    current_prices_for_assets_at_date.get(asset, np.nan)
                    for asset in current_universe_tickers
                ]
            )

            entry_mask = ((prev == 0) & (target != 0)) | (
                (np.sign(prev) != np.sign(target)) & (target != 0)
            )
            exit_mask = target == 0
            entry[entry_mask] = prices[entry_mask]
            entry[exit_mask] = np.nan
            self.entry_prices = pd.Series(entry, index=current_universe_tickers)
            weights_at_current_date = w_target_pre_filter

        final_weights = weights_at_current_date
        benchmark_prices_hist = benchmark_historical_data[
            benchmark_historical_data.index <= current_date
        ]

        sma_filter_window = params.get("sma_filter_window")
        if sma_filter_window and sma_filter_window > 0:
            benchmark_price_series_for_sma = (
                benchmark_prices_hist.xs(price_col_benchmark, level="Field", axis=1)
                if isinstance(benchmark_prices_hist.columns, pd.MultiIndex)
                else benchmark_prices_hist[price_col_benchmark]
            )
            benchmark_sma_series = self._calculate_benchmark_sma(
                benchmark_prices_hist, sma_filter_window, price_col_benchmark
            )

            if (
                current_date in benchmark_price_series_for_sma.index
                and current_date in benchmark_sma_series.index
            ):
                current_benchmark_price_val = benchmark_price_series_for_sma.loc[current_date]
                current_benchmark_sma_val = benchmark_sma_series.loc[current_date]

                def to_scalar(val: Union[pd.Series, pd.DataFrame, float, int, Any]) -> float:
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
                price_series = pd.Series([price_scalar], index=[current_date])
                sma_series = pd.Series([sma_scalar], index=[current_date])
                self.current_derisk_flag, self.consecutive_periods_under_sma = (
                    self._calculate_derisk_flags(
                        price_series,
                        sma_series,
                        derisk_periods,
                        self.current_derisk_flag,
                        self.consecutive_periods_under_sma,
                    )
                )
                if self.current_derisk_flag or (
                    not np.isnan(price_scalar)
                    and not np.isnan(sma_scalar)
                    and price_scalar < sma_scalar
                ):
                    final_weights[:] = 0.0

        risk_off_generator = self.get_risk_off_signal_generator()
        if risk_off_generator:
            risk_off_signal = risk_off_generator.generate_risk_off_signal(
                all_historical_data,
                benchmark_historical_data,
                non_universe_historical_data or pd.DataFrame(),
                current_date,
            )
            if risk_off_signal:
                final_weights[:] = 0.0

        before_risk = np.asarray(
            [weights_at_current_date.get(asset, 0) for asset in current_universe_tickers]
        )
        after_risk = np.asarray([final_weights.get(asset, 0) for asset in current_universe_tickers])
        entry = np.asarray(
            [self.entry_prices.get(asset, np.nan) for asset in current_universe_tickers]
        )
        exit_mask = (before_risk != 0) & (after_risk == 0)
        entry[exit_mask] = np.nan
        self.entry_prices = pd.Series(entry, index=current_universe_tickers)

        self.w_prev = final_weights

        output_weights_df = pd.DataFrame(
            0.0, index=[current_date], columns=current_universe_tickers
        )
        output_weights_df.loc[current_date, :] = final_weights
        return output_weights_df

    def _get_params(self) -> Dict[str, Any]:
        return cast(
            Dict[str, Any], self.strategy_config.get("strategy_params", self.strategy_config)
        )


__all__ = ["SharpeMomentumPortfolioStrategy"]
