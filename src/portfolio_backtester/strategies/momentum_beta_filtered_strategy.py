from __future__ import annotations

from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from .momentum_unfiltered_atr_strategy import MomentumUnfilteredAtrStrategy


class MomentumBetaFilteredStrategy(MomentumUnfilteredAtrStrategy):
    """Momentum strategy variant that removes the highest-beta stocks from the long leg
    and allows tactical shorts when those high-beta names become overbought.

    Key differences versus :class:`MomentumUnfilteredAtrStrategy`:
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
            "rsi_overbought": 85,
            # Force long_only = False so that shorts are permitted
            "long_only": False,
        }

        # Ensure nested dict exists and apply defaults
        strategy_params = strategy_config.setdefault("strategy_params", {})
        for k, v in beta_defaults.items():
            strategy_params.setdefault(k, v)

        super().__init__(strategy_config)

        # Per-rebalancing state — populated inside *generate_signals*
        self._assets_to_exclude_from_longs: Set[str] = set()
        self._short_assets: List[str] = []

    # ------------------------------------------------------------------
    # Optimiser support
    # ------------------------------------------------------------------
    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return super().tunable_parameters() | {
            "beta_lookback_days",
            "num_high_beta_to_exclude",
            "rsi_length",
            "rsi_overbought",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_close_prices(df: pd.DataFrame, price_column: str = "Close") -> pd.DataFrame | pd.Series:
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
    def _calculate_rsi(price_series: pd.Series, window: int) -> float:
        """Simple (SMA-based) Relative Strength Index for the **latest** value."""
        if price_series.size < window + 1:
            return np.nan

        delta = price_series.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.tail(window).mean()
        avg_loss = losses.tail(window).mean()

        if avg_loss == 0 or np.isnan(avg_loss):
            return 100.0  # Extreme overbought

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _latest_rolling_betas(asset_returns: pd.DataFrame, bench_returns: pd.Series, window: int) -> pd.Series:
        """Vectorised *pandas* implementation – sufficient for small universes."""
        # Align indices
        bench_aligned = bench_returns.reindex(asset_returns.index)
        rolling_var = bench_aligned.rolling(window).var()
        # *DataFrame.cov* with *Series* returns a frame → we take the *last* row
        cov = asset_returns.rolling(window).cov(bench_aligned)
        betas = cov.div(rolling_var, axis=0)
        return betas.iloc[-1]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        params = self.strategy_config.get("strategy_params", self.strategy_config)

        price_col_asset = params.get("price_column_asset", "Close")
        price_col_bench = params.get("price_column_benchmark", "Close")
        beta_window = int(params.get("beta_lookback_days", 21))
        num_exclude = max(1, int(params.get("num_high_beta_to_exclude", 3)))
        rsi_len = int(params.get("rsi_length", 3))
        rsi_over = float(params.get("rsi_overbought", 85))

        # --------------------------------------------------------------
        # Discard any extra kwargs supplied by the Backtester interface
        # (e.g. *non_universe_historical_data*) that the parent class does
        # **not** accept.
        # --------------------------------------------------------------
        kwargs.pop("non_universe_historical_data", None)

        # --------------------------------------------------------------
        # Build *daily* close price series up to *current_date*
        # --------------------------------------------------------------
        asset_close = self._extract_close_prices(all_historical_data, price_col_asset)
        bench_close_df = self._extract_close_prices(benchmark_historical_data, price_col_bench)

        # Choose the benchmark column (first if multi-asset)
        benchmark_close = (
            bench_close_df.iloc[:, 0] if isinstance(bench_close_df, pd.DataFrame) else bench_close_df
        )

        # Restrict history to current date and keep a small buffer to ensure
        # we always have at least *beta_window* observations.
        asset_close = asset_close[asset_close.index <= current_date].tail(beta_window + 40)
        benchmark_close = benchmark_close[benchmark_close.index <= current_date].tail(beta_window + 40)

        # --------------------------------------------------------------
        # Compute *daily* returns
        # --------------------------------------------------------------
        asset_rets = asset_close.pct_change(fill_method=None)
        bench_rets = benchmark_close.pct_change(fill_method=None)

        # Drop assets with no data whatsoever
        asset_rets = asset_rets.dropna(axis=1, how="all")

        # --------------------------------------------------------------
        # Rolling beta – pick the latest value for each asset
        # --------------------------------------------------------------
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
        short_candidates: List[str] = []
        for asset in top_beta_assets:
            if asset not in asset_close.columns:
                continue
            rsi_value = self._calculate_rsi(asset_close[asset].dropna(), rsi_len)
            if rsi_value >= rsi_over:
                short_candidates.append(asset)

        # --------------------------------------------------------------
        # Maintain existing shorts until the cover condition is met
        # --------------------------------------------------------------
        shorts_to_keep: List[str] = []
        if self.w_prev is not None and not self.w_prev.empty:
            for asset, weight in self.w_prev[self.w_prev < 0].items():
                if asset not in asset_close.columns:
                    continue
                price_series = asset_close[asset].dropna()
                if price_series.empty:
                    continue

                # Current close
                current_close = price_series.iloc[-1]

                # Prior month low
                prev_month_date = current_date - pd.DateOffset(months=1)
                mask_prev_month = (
                    (price_series.index.month == prev_month_date.month)
                    & (price_series.index.year == prev_month_date.year)
                )
                prev_month_low = price_series[mask_prev_month].min()

                # Cover only if we *definitively* break below the prior month's low
                if (not np.isnan(prev_month_low)) and (current_close < prev_month_low):
                    # Do **not** append → position will be closed
                    continue
                shorts_to_keep.append(asset)

        # Final list of shorts for this rebalance
        self._short_assets = list({*shorts_to_keep, *short_candidates})

        # --------------------------------------------------------------
        # Delegate the heavy-lifting to the parent class
        # --------------------------------------------------------------
        return super().generate_signals(
            all_historical_data,
            benchmark_historical_data,
            current_date,
            start_date,
            end_date,
        )

    # ------------------------------------------------------------------
    # Custom candidate weight logic – invoked by *MomentumStrategy*
    # ------------------------------------------------------------------
    def _calculate_candidate_weights(self, look: pd.Series) -> pd.Series:  # type: ignore[override]
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