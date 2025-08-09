from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd
import logging

from portfolio_backtester.strategies._core.base import PortfolioStrategy
from portfolio_backtester.utils.portfolio_utils import default_candidate_weights
from portfolio_backtester.utils.portfolio_utils import apply_leverage_and_smoothing
from portfolio_backtester.utils.price_data_utils import extract_current_prices

logger = logging.getLogger(__name__)


class BaseMomentumPortfolioStrategy(PortfolioStrategy):
    """
    Abstract base class for momentum-based portfolio strategies.

    This class uses the Template Method design pattern to provide a skeleton
    for momentum strategies. Subclasses must implement the `_calculate_scores`
    method to define their specific momentum scoring logic.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)
        self.w_prev: Optional[pd.Series] = None
        self.current_derisk_flag: bool = False
        self.consecutive_periods_under_sma: int = 0
        self.entry_prices: Optional[pd.Series] = None

        params_dict_to_update = self.strategy_config
        if "strategy_params" in self.strategy_config:
            if self.strategy_config["strategy_params"] is None:
                self.strategy_config["strategy_params"] = {}
            params_dict_to_update = self.strategy_config["strategy_params"]

        defaults = {
            "num_holdings": None,
            "top_decile_fraction": 0.1,
            "smoothing_lambda": 0.5,
            "leverage": 1.0,
            "trade_longs": True,
            "trade_shorts": False,
            "sma_filter_window": None,
            "derisk_days_under_sma": 10,
            "apply_trading_lag": False,
            "price_column_asset": "Close",
            "price_column_benchmark": "Close",
        }
        for k, v in defaults.items():
            params_dict_to_update.setdefault(k, v)

    @abstractmethod
    def _calculate_scores(
        self,
        asset_prices: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        """
        Calculate the momentum scores for each asset.

        Args:
            asset_prices: DataFrame of historical prices for assets.
            current_date: The current date for score calculation.

        Returns:
            A Series of momentum scores for each asset.
        """
        raise NotImplementedError

    def _calculate_candidate_weights(self, scores: pd.Series) -> pd.Series:
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        result = default_candidate_weights(scores, params)
        return pd.Series(result) if not isinstance(result, pd.Series) else result

    def _apply_leverage_and_smoothing(
        self, candidate_weights: pd.Series, prev_weights: Optional[pd.Series]
    ) -> pd.Series:
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        result = apply_leverage_and_smoothing(candidate_weights, prev_weights, params)
        return pd.Series(result) if not isinstance(result, pd.Series) else result

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if current_date is None:
            current_date = all_historical_data.index[-1]

        current_date = cast(pd.Timestamp, current_date)

        is_sufficient, _ = self.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )

        original_assets = (
            all_historical_data.columns.get_level_values("Ticker").unique()
            if isinstance(all_historical_data.columns, pd.MultiIndex)
            else all_historical_data.columns
        )

        if not is_sufficient:
            return pd.DataFrame(0.0, index=[current_date], columns=original_assets)

        valid_assets = self.filter_universe_by_data_availability(all_historical_data, current_date)
        if not valid_assets:
            return pd.DataFrame(0.0, index=[current_date], columns=original_assets)

        params = self.strategy_config.get("strategy_params", self.strategy_config)
        price_col_asset = params["price_column_asset"]
        price_col_benchmark = params["price_column_benchmark"]

        if (start_date and current_date < start_date) or (end_date and current_date > end_date):
            return pd.DataFrame(0.0, index=[current_date], columns=original_assets)

        if isinstance(all_historical_data.columns, pd.MultiIndex):
            asset_prices = all_historical_data.xs(price_col_asset, level="Field", axis=1)
            asset_prices = asset_prices.loc[:, asset_prices.columns.isin(valid_assets)]
        else:
            asset_prices = all_historical_data.loc[
                :, all_historical_data.columns.isin(valid_assets)
            ]

        asset_prices_hist = asset_prices[asset_prices.index <= current_date]
        benchmark_prices_hist = benchmark_historical_data[
            benchmark_historical_data.index <= current_date
        ]

        # Ensure asset_prices_hist is a DataFrame for type safety
        if isinstance(asset_prices_hist, pd.Series):
            asset_prices_hist = asset_prices_hist.to_frame()

        current_universe_tickers = asset_prices_hist.columns

        if self.w_prev is None:
            self.w_prev = pd.Series(0.0, index=current_universe_tickers)
        else:
            self.w_prev = self.w_prev.reindex(current_universe_tickers).fillna(0.0)

        if self.entry_prices is None:
            self.entry_prices = pd.Series(np.nan, index=current_universe_tickers)
        else:
            self.entry_prices = self.entry_prices.reindex(current_universe_tickers).fillna(np.nan)

        scores = self._calculate_scores(asset_prices_hist, current_date)

        if scores.empty or scores.isna().all():
            weights_at_current_date = self.w_prev.copy()
        else:
            cand_weights = self._calculate_candidate_weights(scores)
            weights_at_current_date = self._apply_leverage_and_smoothing(cand_weights, self.w_prev)

            current_prices = extract_current_prices(
                asset_prices_hist, current_date, current_universe_tickers
            )

            price_update_mask = ((self.w_prev == 0) & (weights_at_current_date != 0)) | (
                (np.sign(self.w_prev) != np.sign(weights_at_current_date))
                & (weights_at_current_date != 0)
            )

            valid_prices = current_prices.dropna()
            update_assets = price_update_mask.index.intersection(valid_prices.index)
            self.entry_prices.loc[update_assets] = valid_prices.loc[update_assets]

            zeroed_assets = weights_at_current_date.index[weights_at_current_date == 0]
            self.entry_prices.loc[zeroed_assets] = np.nan

        # NOTE: Stop loss is now handled by DailyStopLossMonitor in WindowEvaluator
        # This provides schedule-independent daily monitoring instead of only
        # triggering on strategy rebalance dates (monthly/quarterly)
        # Legacy stop loss code removed to prevent duplicate risk management

        # Apply Risk Filters
        final_weights = self._apply_risk_filters(
            weights_at_current_date,
            benchmark_prices_hist,
            current_date,
            price_col_benchmark,
            all_historical_data,
            benchmark_historical_data,
        )

        zeroed_by_risk_filter = weights_at_current_date.index[
            (weights_at_current_date != 0) & (final_weights == 0)
        ]
        if not zeroed_by_risk_filter.empty:
            self.entry_prices.loc[zeroed_by_risk_filter] = np.nan

        self.w_prev = final_weights

        output_df = pd.DataFrame(0.0, index=[current_date], columns=original_assets)
        output_df.loc[current_date, final_weights.index] = final_weights

        # Enforce trade direction constraints - this will raise an exception if violated
        output_df = self._enforce_trade_direction_constraints(output_df)

        return output_df

    def _apply_risk_filters(
        self,
        weights: pd.Series,
        benchmark_prices_hist: pd.DataFrame,
        current_date: pd.Timestamp,
        price_col_benchmark: str,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
    ) -> pd.Series:
        final_weights = weights.copy()
        params = self.strategy_config.get("strategy_params", self.strategy_config)

        # SMA Filter
        sma_filter_window = params.get("sma_filter_window")
        if sma_filter_window and sma_filter_window > 0:
            benchmark_sma = self._calculate_benchmark_sma(
                benchmark_prices_hist, sma_filter_window, price_col_benchmark
            )

            if current_date in benchmark_prices_hist.index and current_date in benchmark_sma.index:
                price_series = (
                    benchmark_prices_hist.xs(price_col_benchmark, level="Field", axis=1)
                    if isinstance(benchmark_prices_hist.columns, pd.MultiIndex)
                    else benchmark_prices_hist[price_col_benchmark]
                )

                # Extract scalar values (handle Series case)
                current_price_val = price_series.loc[current_date]
                if isinstance(current_price_val, pd.Series):
                    current_price_val = current_price_val.iloc[0]

                current_sma_val = benchmark_sma.loc[current_date]
                if isinstance(current_sma_val, pd.Series):
                    current_sma_val = current_sma_val.iloc[0]

                derisk_periods = params.get("derisk_days_under_sma", 10)

                self.current_derisk_flag, self.consecutive_periods_under_sma = (
                    self._calculate_derisk_flags(
                        pd.Series([current_price_val], index=[current_date]),
                        pd.Series([current_sma_val], index=[current_date]),
                        derisk_periods,
                        self.current_derisk_flag,
                        self.consecutive_periods_under_sma,
                    )
                )

                # Ensure current_derisk_flag is a boolean (handle Series case)
                derisk_flag = (
                    self.current_derisk_flag
                    if isinstance(self.current_derisk_flag, bool)
                    else bool(self.current_derisk_flag)
                )

                if derisk_flag or (
                    pd.notna(current_price_val)
                    and pd.notna(current_sma_val)
                    and current_price_val < current_sma_val
                ):
                    final_weights[:] = 0.0
                    return final_weights

        # RoRo Filter
        roro_signal_instance = self.get_roro_signal()
        if roro_signal_instance:
            roro_output = roro_signal_instance.generate_signal(
                all_historical_data[all_historical_data.index <= current_date],
                benchmark_historical_data[benchmark_historical_data.index <= current_date],
                current_date,
            )
            is_roro_risk_off = False
            if isinstance(roro_output, (pd.Series, pd.DataFrame)):
                if not roro_output.empty and current_date in roro_output.index:
                    is_roro_risk_off = not roro_output.loc[current_date]
            elif isinstance(roro_output, (bool, int, float)):
                is_roro_risk_off = not bool(roro_output)

            if is_roro_risk_off:
                final_weights[:] = 0.0

        return final_weights


__all__ = ["BaseMomentumPortfolioStrategy"]
