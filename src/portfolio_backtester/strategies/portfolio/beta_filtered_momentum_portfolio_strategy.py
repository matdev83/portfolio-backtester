from __future__ import annotations

from typing import Any, Dict, List, Set, Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

from ..builtins.portfolio.momentum_unfiltered_atr_portfolio_strategy import (
    MomentumUnfilteredAtrPortfolioStrategy,
)

__all__ = ["MomentumBetaFilteredPortfolioStrategy"]


class MomentumBetaFilteredPortfolioStrategy(MomentumUnfilteredAtrPortfolioStrategy):
    """Momentum strategy variant that removes the highest-beta stocks from the long leg
    and allows tactical shorts when those high-beta names become overbought.

    Key differences versus :class:`MomentumUnfilteredAtrPortfolioStrategy`:
    1.  For every rebalancing date, calculate a *rolling* market beta for each asset
        over *beta_lookback_days* (default ``21`` trading days).
    2.  Identify the *num_high_beta_to_exclude* assets with the **highest** betas
        (default ``3``).  These names are **excluded** from the standard momentum
        long selection.
    3.  If any of the excluded high-beta assets have an RSI greater than or equal
        to *rsi_overbought* (default ``85``) – calculated with a *rsi_length*
        period (default ``3``) – the strategy opens an *equal-weight* short
        position in those names.
    4.  Shorts are covered **only** when the current month’s close falls below
        the *prior* month’s low.

    All added hyper-parameters are exposed via :py:meth:`tunable_parameters` so
    that the optimiser can search over sensible ranges.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(self, strategy_config: Dict[str, Any]):
        # ------------------------------------------------------------------
        # Inject defaults for the new hyper-parameters
        # ------------------------------------------------------------------
        beta_defaults = {
            "beta_lookback_days": 21,
            "num_high_beta_to_exclude": 3,  # must be >=1
            "rsi_length": 3,
            "rsi_overbought": 70,  # Reduced from 85 for more short opportunities
            "short_max_holding_days": 30,  # Maximum holding period for short positions
            # Allow both longs and shorts for beta filtering
            "trade_longs": True,
            "trade_shorts": True,
        }

        # Ensure nested dict exists and apply defaults
        strategy_params = strategy_config.setdefault("strategy_params", {})
        for k, v in beta_defaults.items():
            strategy_params.setdefault(k, v)

        super().__init__(strategy_config)

        # Per-rebalancing state — populated inside *generate_signals*
        self._assets_to_exclude_from_longs: Set[str] = set()
        self._short_assets: List[str] = []
        self._short_entry_dates: Dict[str, pd.Timestamp] = (
            {}
        )  # Track entry dates for time-based exit

    # ------------------------------------------------------------------
    # Optimiser support
    # ------------------------------------------------------------------
    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        parent_params: Dict[str, Dict[str, Any]] = super().tunable_parameters()
        parent_params.update(
            {
                "beta_lookback_days": {
                    "type": "int",
                    "min": 5,
                    "max": 252,
                    "default": 21,
                },
                "num_high_beta_to_exclude": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "default": 3,
                },
                "rsi_length": {"type": "int", "min": 2, "max": 20, "default": 3},
                "rsi_overbought": {
                    "type": "float",
                    "min": 50,
                    "max": 100,
                    "default": 70,
                },
                "short_max_holding_days": {
                    "type": "int",
                    "min": 1,
                    "max": 90,
                    "default": 30,
                },
            }
        )
        return parent_params

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_close_prices(
        df: pd.DataFrame, price_column: str = "Close"
    ) -> pd.DataFrame | pd.Series:
        """Return a *DataFrame* (assets as columns) of close prices.

        The function gracefully handles *MultiIndex* columns with a ``Field``
        level (as used throughout the code-base) **and** single-level columns.
        """
        if isinstance(df.columns, pd.MultiIndex):
            if "Field" in df.columns.names:
                return df.xs(price_column, level="Field", axis=1)
            # Fallback – treat the entire frame as closes
            return df
        return df

    @staticmethod
    def _latest_rolling_betas(
        asset_returns: pd.DataFrame, bench_returns: pd.Series, window: int
    ) -> pd.Series:
        """Vectorised *pandas* implementation – sufficient for small universes."""
        # Align indices
        bench_aligned = bench_returns.reindex(asset_returns.index)
        rolling_var = bench_aligned.rolling(window).var()
        # *DataFrame.cov* with *Series* returns a frame → we take the *last* row
        cov = asset_returns.rolling(window).cov(bench_aligned)
        betas = cov.div(rolling_var, axis=0)
        return betas.iloc[-1]

    # (method removed: _calculate_rsi was unused)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: "Optional[pd.DataFrame]" = None,
        current_date: "Optional[pd.Timestamp]" = None,
        start_date: "Optional[pd.Timestamp]" = None,
        end_date: "Optional[pd.Timestamp]" = None,
    ) -> pd.DataFrame:
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        # Handle None current_date gracefully - use the last date in the data
        if current_date is None:
            current_date = all_historical_data.index[-1]

        price_col_asset = params.get("price_column_asset", "Close")
        price_col_bench = params.get("price_column_benchmark", "Close")
        beta_window = int(params.get("beta_lookback_days", 21))
        num_exclude = max(1, int(params.get("num_high_beta_to_exclude", 3)))
        rsi_len = int(params.get("rsi_length", 3))
        rsi_over = float(params.get("rsi_overbought", 70))  # Updated default
        short_max_holding_days = int(params.get("short_max_holding_days", 30))

        # --------------------------------------------------------------
        # Build *daily* close price series up to *current_date*
        # --------------------------------------------------------------
        asset_close = self._extract_close_prices(all_historical_data, price_col_asset)
        bench_close_df = self._extract_close_prices(benchmark_historical_data, price_col_bench)

        # Choose the benchmark column (first if multi-asset)
        benchmark_close = (
            bench_close_df.iloc[:, 0]
            if isinstance(bench_close_df, pd.DataFrame)
            else bench_close_df
        )

        # Restrict history to current date and keep a small buffer to ensure
        # we always have at least *beta_window* observations.
        asset_close = asset_close[asset_close.index <= current_date].tail(beta_window + 40)
        benchmark_close = benchmark_close[benchmark_close.index <= current_date].tail(
            beta_window + 40
        )

        # --------------------------------------------------------------
        # Compute *daily* returns
        # --------------------------------------------------------------
        asset_rets = asset_close.pct_change(fill_method=None)
        bench_rets = benchmark_close.pct_change(fill_method=None)

        # Drop assets with no data whatsoever
        # Workaround: drop columns with all NaNs by transposing, dropping rows, and transposing back
        asset_rets = asset_rets.T.dropna(how="all", axis=0).T

        # --------------------------------------------------------------
        # Rolling beta – pick the latest value for each asset
        # --------------------------------------------------------------
        # Ensure asset_rets is always a DataFrame
        if isinstance(asset_rets, pd.Series):
            asset_rets = asset_rets.to_frame()
        betas = self._latest_rolling_betas(asset_rets, bench_rets, beta_window)
        betas = betas.dropna()

        top_beta_assets = (
            betas.sort_values(ascending=False).head(num_exclude).index.tolist()
            if not betas.empty
            else []
        )
        self._assets_to_exclude_from_longs = set(top_beta_assets)

        # --------------------------------------------------------------
        # RSI check for high-beta names → decide which to short
        # --------------------------------------------------------------
        # Vectorized RSI check for high-beta names
        short_candidates = []
        if top_beta_assets:
            high_beta_prices = asset_close[top_beta_assets].dropna(how="all")

            # Compute RSI for each asset using library implementation
            def rsi_vec(prices):
                if prices.size < rsi_len + 1:
                    return np.nan
                # Use ta library for RSI calculation
                rsi = RSIIndicator(close=prices, window=rsi_len).rsi()
                return rsi.iloc[-1] if len(rsi) > 0 else np.nan

            rsi_vals = high_beta_prices.apply(rsi_vec, axis=0)
            short_candidates = rsi_vals[rsi_vals >= rsi_over].index.tolist()

        # --------------------------------------------------------------
        # Maintain existing shorts until the cover condition is met
        # --------------------------------------------------------------
        shorts_to_keep = []
        if self.w_prev is not None and not self.w_prev.empty:
            prev_shorts = [
                asset
                for asset in self.w_prev[self.w_prev < 0].index
                if asset in asset_close.columns
            ]
            if len(prev_shorts) > 0:
                # Workaround: drop columns with all NaNs by transposing, dropping rows, and transposing back
                price_df = asset_close[prev_shorts].T.dropna(how="all", axis=0).T
                # Ensure DatetimeIndex for .month/.year
                if not isinstance(price_df.index, pd.DatetimeIndex):
                    price_df.index = pd.to_datetime(price_df.index)
                current_close = price_df.iloc[-1]

                # Time-based exit logic
                assets_to_remove = []
                for asset in prev_shorts:
                    # Check if maximum holding period has been reached
                    if asset in self._short_entry_dates and current_date is not None:
                        entry_date = self._short_entry_dates[asset]
                        holding_days = (current_date - entry_date).days
                        if holding_days >= short_max_holding_days:
                            assets_to_remove.append(asset)
                            # Remove from entry dates tracking
                            if asset in self._short_entry_dates:
                                del self._short_entry_dates[asset]

                # Filter out assets that have reached maximum holding period
                prev_shorts = [asset for asset in prev_shorts if asset not in assets_to_remove]

                # Original exit logic - close when current month's close falls below prior month's low
                if current_date is not None:
                    prev_month_date = current_date - pd.DateOffset(months=1)
                    mask_prev_month = (price_df.index.month == prev_month_date.month) & (
                        price_df.index.year == prev_month_date.year
                    )
                    prev_month_lows = price_df[mask_prev_month].min(axis=0)
                    for asset in prev_shorts:
                        prev_low = prev_month_lows[asset]
                        curr_close = current_close[asset]
                        if np.isnan(prev_low) or bool(curr_close >= prev_low):
                            shorts_to_keep.append(asset)

        # Final list of shorts for this rebalance
        # Update entry dates for new short positions
        for asset in short_candidates:
            if asset not in self._short_entry_dates and current_date is not None:
                self._short_entry_dates[asset] = current_date

        # Remove entry dates for assets that are no longer shorted
        current_shorts = set(shorts_to_keep + short_candidates)
        for asset in list(self._short_entry_dates.keys()):
            if asset not in current_shorts:
                del self._short_entry_dates[asset]

        self._short_assets = list({*shorts_to_keep, *short_candidates})

        # --------------------------------------------------------------
        # Delegate the heavy-lifting to the parent class
        # --------------------------------------------------------------
        return super().generate_signals(
            all_historical_data,
            benchmark_historical_data,
            non_universe_historical_data,
            current_date,
            start_date,
            end_date,
        )

    # ------------------------------------------------------------------
    # Custom candidate weight logic – invoked by *BaseMomentumPortfolioStrategy*
    # ------------------------------------------------------------------
    def _calculate_candidate_weights(self, look: pd.Series) -> pd.Series:
        params = self.strategy_config.get("strategy_params", self.strategy_config)

        # ---- Long leg (momentum winners) ---------------------------------
        num_holdings = params.get("num_holdings")
        if num_holdings is not None and num_holdings > 0:
            n_long = int(num_holdings)
        else:
            n_long = max(int(np.ceil(params.get("top_decile_fraction", 0.1) * look.count())), 1)

        # Select momentum winners while skipping excluded tickers. Keep iterating
        # through the ranked list until we collect *n_long* names or run out.
        winners_filtered: List[str] = []
        for ticker in look.sort_values(ascending=False).index:
            if ticker in self._assets_to_exclude_from_longs:
                continue
            winners_filtered.append(ticker)
            if len(winners_filtered) == n_long:
                break

        cand = pd.Series(index=look.index, dtype=float).fillna(0.0)
        if winners_filtered:
            cand[winners_filtered] = 1.0 / len(winners_filtered)

        # ---- Short leg ---------------------------------------------------
        if self._short_assets:
            cand[self._short_assets] = -1.0 / len(self._short_assets)

        return cand
