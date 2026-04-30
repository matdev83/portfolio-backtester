import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from ..interfaces.strategy_resolver import StrategyResolverFactory
from ..interfaces.enforcement import enforce_strategy_parameter

# Removed legacy position sizer imports - now using strategy provider interfaces

logger = logging.getLogger(__name__)


def _effective_global_config_for_universe(
    strategy: Any, global_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if global_config is not None:
        return dict(global_config)
    gc = getattr(strategy, "global_config", None)
    if isinstance(gc, dict):
        return dict(gc)
    return {}


def _strategy_supports_method_dynamic_universe(strategy: Any) -> bool:
    try:
        provider = strategy.get_universe_provider()
    except Exception:
        return False
    supports_fn = getattr(provider, "supports_dynamic_universe", None)
    if not callable(supports_fn):
        return False
    try:
        return bool(supports_fn())
    except Exception:
        return False


def _pit_symbols_for_rebalance_date(
    strategy: Any, global_config: Dict[str, Any], current_date: pd.Timestamp
) -> Optional[List[str]]:
    try:
        pairs = strategy.get_universe_method_with_date(global_config, current_date)
    except Exception:
        return None
    out = [str(sym) for sym, _ in pairs]
    return out if out else None


def _slice_asset_hist_by_tickers(
    asset_hist: pd.DataFrame, pit_tickers: List[str], is_multi_index: bool
) -> pd.DataFrame:
    if not pit_tickers:
        return asset_hist
    if is_multi_index:
        tick_level = asset_hist.columns.get_level_values("Ticker")
        mask = tick_level.isin(pit_tickers)
        sub = asset_hist.loc[:, mask]
        return sub if sub.shape[1] > 0 else asset_hist
    present = [t for t in pit_tickers if t in asset_hist.columns]
    if not present:
        return asset_hist
    return asset_hist[present]


def _scenario_timing_mode(canonical_config: Any) -> str:
    tc = getattr(canonical_config, "timing_config", None)
    if tc is None and isinstance(canonical_config, dict):
        tc = canonical_config.get("timing_config")
    if tc is None:
        return "time_based"
    if hasattr(tc, "get"):
        m = tc.get("mode")
        return str(m) if m else "time_based"
    return "time_based"


def generate_signals(
    strategy,
    scenario_config,
    price_data_daily_ohlc,
    universe_tickers,
    benchmark_ticker,
    has_timed_out,
    global_config: Optional[Dict[str, Any]] = None,
):
    from ..canonical_config import CanonicalScenarioConfig
    from ..scenario_normalizer import ScenarioNormalizer

    # Ensure we are working with a canonical config internally
    if not isinstance(scenario_config, CanonicalScenarioConfig):
        logger.warning(
            "ACCIDENTAL BYPASS: Raw scenario dictionary passed to strategy_logic.generate_signals. "
            "All scenarios should be canonicalized at the boundary. "
            "Scenario: %s",
            scenario_config.get("name", "unnamed"),
        )
        normalizer = ScenarioNormalizer()
        if global_config is None:
            global_config = getattr(strategy, "global_config", {})
        canonical_config = normalizer.normalize(
            scenario=scenario_config, global_config=global_config
        )
    else:
        canonical_config = scenario_config

    # Check if this is a meta strategy - if so, use trade-based approach
    strategy_resolver = StrategyResolverFactory.create()
    if strategy_resolver.is_meta_strategy(type(strategy)):
        return _generate_meta_strategy_signals(
            strategy,
            canonical_config,
            price_data_daily_ohlc,
            universe_tickers,
            benchmark_ticker,
            has_timed_out,
        )

    # Standard strategy signal generation
    timing_controller = strategy.get_timing_controller()
    timing_controller.reset_state()

    start_date = price_data_daily_ohlc.index.min()
    end_date = price_data_daily_ohlc.index.max()

    scenario_start_raw = canonical_config.get("start_date")
    scenario_end_raw = canonical_config.get("end_date")
    wfo_start_raw = canonical_config.get("wfo_start_date")
    wfo_end_raw = canonical_config.get("wfo_end_date")

    def _align_ts(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
        if ts is None:
            return None
        if getattr(start_date, "tzinfo", None) is not None:
            return (
                ts.tz_localize(start_date.tzinfo)
                if ts.tzinfo is None
                else ts.tz_convert(start_date.tzinfo)
            )
        if ts.tzinfo is not None:
            return ts.tz_convert(None)
        return ts

    scenario_start_date = _align_ts(
        pd.to_datetime(scenario_start_raw) if scenario_start_raw else None
    )
    scenario_end_date = _align_ts(pd.to_datetime(scenario_end_raw) if scenario_end_raw else None)
    wfo_start_date = _align_ts(pd.to_datetime(wfo_start_raw) if wfo_start_raw else None)
    wfo_end_date = _align_ts(pd.to_datetime(wfo_end_raw) if wfo_end_raw else None)

    if scenario_start_date is not None:
        start_date = max(start_date, scenario_start_date)
    if scenario_end_date is not None:
        end_date = min(end_date, scenario_end_date)

    if wfo_start_date is not None:
        start_date = max(start_date, wfo_start_date)
    if wfo_end_date is not None:
        end_date = min(end_date, wfo_end_date)

    rebalance_dates = timing_controller.get_rebalance_dates(
        start_date=start_date,
        end_date=end_date,
        available_dates=price_data_daily_ohlc.index,
        strategy_context=strategy,
    )

    timing_mode = _scenario_timing_mode(canonical_config)
    use_sparse_nan_for_inactive_rows = timing_mode == "signal_based"

    # Honor scenario-configured daily rebalance for meta strategies in single-path architecture
    try:
        configured_freq = scenario_config.get("timing_config", {}).get("rebalance_frequency")
        if configured_freq == "D":
            rebalance_dates = price_data_daily_ohlc.index
    except Exception:
        # Fall back silently to timing controller dates on any issue
        pass

    # OPTIMIZATION: Pre-calculate unique fields to avoid repeated computation in the loop.
    unique_fields = []
    if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
        unique_fields = list(price_data_daily_ohlc.columns.get_level_values("Field").unique())

    # --- PERFORMANCE OPTIMIZATION: Hoist data preparations out of the loop ---
    is_multi_index = (
        isinstance(price_data_daily_ohlc.columns, pd.MultiIndex)
        and "Ticker" in price_data_daily_ohlc.columns.names
    )

    asset_data_view = None
    benchmark_data_view = None
    non_universe_data_view = None
    non_universe_tickers = strategy.get_non_universe_data_requirements()

    if is_multi_index:
        asset_hist_data_cols = pd.MultiIndex.from_product(
            [universe_tickers, unique_fields], names=["Ticker", "Field"]
        ).intersection(price_data_daily_ohlc.columns)
        asset_data_view = price_data_daily_ohlc[asset_hist_data_cols]

        benchmark_hist_data_cols = pd.MultiIndex.from_product(
            [[benchmark_ticker], unique_fields], names=["Ticker", "Field"]
        ).intersection(price_data_daily_ohlc.columns)
        benchmark_data_view = price_data_daily_ohlc[benchmark_hist_data_cols]

        if non_universe_tickers:
            non_universe_hist_data_cols = pd.MultiIndex.from_product(
                [non_universe_tickers, unique_fields], names=["Ticker", "Field"]
            ).intersection(price_data_daily_ohlc.columns)
            non_universe_data_view = price_data_daily_ohlc[non_universe_hist_data_cols]

    else:
        asset_data_view = price_data_daily_ohlc[universe_tickers]
        benchmark_data_view = price_data_daily_ohlc[[benchmark_ticker]]
        if non_universe_tickers:
            non_universe_data_view = price_data_daily_ohlc[non_universe_tickers]

    effective_global_config = _effective_global_config_for_universe(strategy, global_config)
    use_pit_universe_slice = _strategy_supports_method_dynamic_universe(strategy)

    # PERFORMANCE: Pre-compute signature inspection outside the loop
    import inspect

    sig = inspect.signature(strategy.generate_signals)
    has_non_universe_param = "non_universe_historical_data" in sig.parameters

    # PERFORMANCE: Cache MultiIndex level checks
    has_close_field = False
    if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
        has_close_field = "Close" in price_data_daily_ohlc.columns.get_level_values("Field")

        # OPTIMIZATION: Pre-compute indices for faster MultiIndex access
        if not hasattr(strategy, "_close_field_indices"):
            strategy._close_field_indices = {}
            # Find Field level index
            field_level_idx = price_data_daily_ohlc.columns.names.index("Field")
            # Get all Close column indices
            field_values = price_data_daily_ohlc.columns.get_level_values(field_level_idx)
            close_indices = [i for i, val in enumerate(field_values) if val == "Close"]
            strategy._close_field_indices["Close"] = close_indices

            # Also find Ticker level for faster access
            ticker_level_idx = price_data_daily_ohlc.columns.names.index("Ticker")
            strategy._ticker_level_idx = ticker_level_idx

    # PERFORMANCE: Pre-compute date masks for slicing
    date_masks = {}
    for date in rebalance_dates:
        date_masks[date] = asset_data_view.index <= date

    # PERFORMANCE: Pre-compute close prices and universe prices caches
    if not hasattr(strategy, "_cached_close_prices"):
        strategy._cached_close_prices = {}
    if not hasattr(strategy, "_cached_universe_prices"):
        strategy._cached_universe_prices = {}
    # --- END PERFORMANCE OPTIMIZATION ---

    num_assets = len(universe_tickers)
    signals_arr = (
        np.full((len(rebalance_dates), num_assets), np.nan, dtype=float)
        if use_sparse_nan_for_inactive_rows
        else np.zeros((len(rebalance_dates), num_assets))
    )

    for i, current_rebalance_date in enumerate(rebalance_dates):
        if has_timed_out():
            logger.warning("Timeout reached during scenario run. Halting signal generation.")
            break

        should_generate = timing_controller.should_generate_signal(
            current_date=current_rebalance_date, strategy_context=strategy
        )

        if not should_generate:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Timing controller skipped signal generation for date: {current_rebalance_date}"
                )
            continue

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Generating signals for date: {current_rebalance_date}")

        # SLICING OPTIMIZATION: Use pre-computed boolean mask for better performance
        date_mask = date_masks[current_rebalance_date]

        # Apply boolean mask (faster than iloc with computed indices)
        all_historical_data_for_strat = asset_data_view.loc[date_mask]
        benchmark_historical_data_for_strat = benchmark_data_view.loc[date_mask]

        non_universe_historical_data_for_strat = pd.DataFrame()
        if non_universe_data_view is not None:
            non_universe_historical_data_for_strat = non_universe_data_view.loc[date_mask]

        strat_asset_hist = all_historical_data_for_strat
        if use_pit_universe_slice:
            pit_syms = _pit_symbols_for_rebalance_date(
                strategy, effective_global_config, current_rebalance_date
            )
            if pit_syms:
                strat_asset_hist = _slice_asset_hist_by_tickers(
                    all_historical_data_for_strat, pit_syms, is_multi_index
                )

        # PERFORMANCE: Use pre-computed signature check instead of inspect.signature in the loop
        if has_non_universe_param:
            current_weights_df = strategy.generate_signals(
                all_historical_data=strat_asset_hist,
                benchmark_historical_data=benchmark_historical_data_for_strat,
                non_universe_historical_data=non_universe_historical_data_for_strat,
                current_date=current_rebalance_date,
                start_date=wfo_start_date,
                end_date=wfo_end_date,
            )
        else:
            current_weights_df = strategy.generate_signals(
                all_historical_data=strat_asset_hist,
                benchmark_historical_data=benchmark_historical_data_for_strat,
                current_date=current_rebalance_date,
                start_date=wfo_start_date,
                end_date=wfo_end_date,
            )

        if current_weights_df is not None and not current_weights_df.empty:
            if len(current_weights_df) > 0:
                current_weights_series = current_weights_df.iloc[0]
                timing_controller.update_signal_state(
                    current_rebalance_date, current_weights_series
                )

                try:
                    # PERFORMANCE: Use cached close prices with optimized MultiIndex access
                    if current_rebalance_date not in strategy._cached_close_prices:
                        # Cache miss - compute prices
                        if has_close_field and hasattr(strategy, "_close_field_indices"):
                            # Fast path: Use pre-computed indices for direct column access
                            row = price_data_daily_ohlc.loc[current_rebalance_date]
                            close_indices = strategy._close_field_indices["Close"]

                            # Check if we have valid indices
                            if close_indices:
                                # Extract columns using pre-computed indices
                                # This avoids the expensive xs operation
                                cols = [row.iloc[i] for i in close_indices]
                                close_values = np.array(cols)

                                # Get ticker names from the MultiIndex
                                ticker_level = strategy._ticker_level_idx
                                tickers = [
                                    price_data_daily_ohlc.columns[i][ticker_level]
                                    for i in close_indices
                                ]

                                # Create Series with tickers as index
                                strategy._cached_close_prices[current_rebalance_date] = pd.Series(
                                    close_values, index=tickers
                                )
                            else:
                                # Fallback to xs if optimization not possible
                                strategy._cached_close_prices[current_rebalance_date] = (
                                    price_data_daily_ohlc.loc[current_rebalance_date].xs(
                                        "Close", level="Field"
                                    )
                                )
                        elif has_close_field:
                            # Standard xs operation if we don't have pre-computed indices
                            strategy._cached_close_prices[current_rebalance_date] = (
                                price_data_daily_ohlc.loc[current_rebalance_date].xs(
                                    "Close", level="Field"
                                )
                            )
                        elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
                            strategy._cached_close_prices[current_rebalance_date] = (
                                price_data_daily_ohlc.loc[current_rebalance_date]
                            )
                        else:
                            try:
                                strategy._cached_close_prices[current_rebalance_date] = (
                                    price_data_daily_ohlc.loc[current_rebalance_date].xs(
                                        "Close", level=-1
                                    )
                                )
                            except Exception:
                                strategy._cached_close_prices[current_rebalance_date] = (
                                    price_data_daily_ohlc.loc[current_rebalance_date].iloc[
                                        : len(universe_tickers)
                                    ]
                                )

                    # Use cached prices
                    current_prices = strategy._cached_close_prices[current_rebalance_date]

                    # PERFORMANCE: Use cached universe prices with optimized subset check
                    if current_rebalance_date not in strategy._cached_universe_prices:
                        # Cache miss - compute universe prices
                        # Fast path: check if all tickers are in current_prices using set operations
                        universe_set = set(universe_tickers)
                        prices_set = set(current_prices.index)

                        if universe_set.issubset(prices_set):
                            # All tickers present - direct indexing (fastest)
                            # Use numpy-based indexing for better performance
                            universe_prices = current_prices.loc[universe_tickers]
                        else:
                            # Some tickers missing - reindex needed
                            # Use optimized reindex with pre-allocation
                            universe_prices = pd.Series(
                                index=universe_tickers, dtype=current_prices.dtype
                            )

                            # Fill values for existing tickers (faster than reindex)
                            common_tickers = universe_set.intersection(prices_set)
                            for ticker in common_tickers:
                                universe_prices[ticker] = current_prices[ticker]

                            # Apply ffill only if there are NaN values
                            if universe_prices.isna().any():
                                universe_prices = universe_prices.ffill()

                        # Cache the result
                        strategy._cached_universe_prices[current_rebalance_date] = universe_prices

                    # Use cached universe prices
                    universe_prices = strategy._cached_universe_prices[current_rebalance_date]

                    timing_controller.update_position_state(
                        current_rebalance_date, current_weights_series, universe_prices
                    )

                except Exception as e:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Could not update position state for {current_rebalance_date}: {e}"
                        )

        if use_sparse_nan_for_inactive_rows:
            if current_weights_df is None or current_weights_df.empty:
                continue
            aligned_weights = current_weights_df.reindex(columns=universe_tickers)
            if aligned_weights.empty:
                continue
            signals_arr[i, :] = aligned_weights.iloc[0].astype(np.float64).to_numpy()
        else:
            if current_weights_df is not None:
                aligned_weights = current_weights_df.reindex(columns=universe_tickers).fillna(0.0)
                if not aligned_weights.empty:
                    signals_arr[i, :] = aligned_weights.iloc[0].values.astype(np.float64)
                else:
                    signals_arr[i, :] = 0.0
            else:
                signals_arr[i, :] = 0.0

    signals = pd.DataFrame(signals_arr, index=rebalance_dates, columns=universe_tickers)
    if not use_sparse_nan_for_inactive_rows:
        signals.fillna(0.0, inplace=True)

    # DEPRECATED: Framework-level trade direction filtering has been replaced
    # by strict enforcement at the strategy level. Strategies now throw
    # TradeDirectionViolationError if they violate trade_longs/trade_shorts constraints.
    # This prevents coding errors and ensures immediate failure rather than silent filtering.
    # The framework filtering is kept for backward compatibility but should not be needed.

    return signals


def _generate_meta_strategy_signals(
    strategy,
    scenario_config,
    price_data_daily_ohlc,
    universe_tickers,
    benchmark_ticker,
    has_timed_out,
):
    """
    Generate signals for meta strategies that aggregate child strategies.
    """
    # Initialize the meta strategy with the scenario config
    if hasattr(strategy, "initialize"):
        strategy.initialize(scenario_config)

    # Get all rebalance dates from the price data
    rebalance_dates = price_data_daily_ohlc.index

    # Honor scenario-configured rebalance frequency if present
    try:
        configured_freq = scenario_config.get("timing_config", {}).get("rebalance_frequency")
        if configured_freq:
            # Use the configured frequency to resample the price data index
            if configured_freq != "D":  # Daily is already the default
                rebalance_dates = pd.date_range(
                    start=rebalance_dates.min(),
                    end=rebalance_dates.max(),
                    freq="ME" if configured_freq == "M" else configured_freq,
                )
                # Ensure dates exist in the original price data
                rebalance_dates = rebalance_dates[rebalance_dates.isin(price_data_daily_ohlc.index)]
    except Exception as e:
        logger.warning(f"Error setting rebalance frequency: {e}. Using daily rebalance.")

    # OPTIMIZATION: Pre-calculate unique fields to avoid repeated computation in the loop.
    unique_fields = []
    if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
        unique_fields = list(price_data_daily_ohlc.columns.get_level_values("Field").unique())

    # --- PERFORMANCE OPTIMIZATION: Hoist data preparations out of the loop ---
    is_multi_index = (
        isinstance(price_data_daily_ohlc.columns, pd.MultiIndex)
        and "Ticker" in price_data_daily_ohlc.columns.names
    )

    asset_data_view = None
    benchmark_data_view = None
    non_universe_data_view = None
    non_universe_tickers = strategy.get_non_universe_data_requirements()

    if is_multi_index:
        asset_hist_data_cols = pd.MultiIndex.from_product(
            [universe_tickers, unique_fields], names=["Ticker", "Field"]
        ).intersection(price_data_daily_ohlc.columns)
        asset_data_view = price_data_daily_ohlc[asset_hist_data_cols]

        benchmark_hist_data_cols = pd.MultiIndex.from_product(
            [[benchmark_ticker], unique_fields], names=["Ticker", "Field"]
        ).intersection(price_data_daily_ohlc.columns)
        benchmark_data_view = price_data_daily_ohlc[benchmark_hist_data_cols]

        if non_universe_tickers:
            non_universe_hist_data_cols = pd.MultiIndex.from_product(
                [non_universe_tickers, unique_fields], names=["Ticker", "Field"]
            ).intersection(price_data_daily_ohlc.columns)
            non_universe_data_view = price_data_daily_ohlc[non_universe_hist_data_cols]

    else:
        asset_data_view = price_data_daily_ohlc[universe_tickers]
        benchmark_data_view = price_data_daily_ohlc[[benchmark_ticker]]
        if non_universe_tickers:
            non_universe_data_view = price_data_daily_ohlc[non_universe_tickers]

    # PERFORMANCE: Pre-compute date masks for slicing
    date_masks = {}
    for date in rebalance_dates:
        date_masks[date] = asset_data_view.index <= date

    # PERFORMANCE: Cache MultiIndex level checks
    has_close_field = False
    if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
        has_close_field = "Close" in price_data_daily_ohlc.columns.get_level_values("Field")

        # OPTIMIZATION: Pre-compute indices for faster MultiIndex access
        if not hasattr(strategy, "_close_field_indices"):
            strategy._close_field_indices = {}
            # Find Field level index
            field_level_idx = price_data_daily_ohlc.columns.names.index("Field")
            # Get all Close column indices
            field_values = price_data_daily_ohlc.columns.get_level_values(field_level_idx)
            close_indices = [i for i, val in enumerate(field_values) if val == "Close"]
            strategy._close_field_indices["Close"] = close_indices

            # Also find Ticker level for faster access
            ticker_level_idx = price_data_daily_ohlc.columns.names.index("Ticker")
            strategy._ticker_level_idx = ticker_level_idx

    # PERFORMANCE: Pre-compute close prices and universe prices caches
    if not hasattr(strategy, "_cached_close_prices"):
        strategy._cached_close_prices = {}
    if not hasattr(strategy, "_cached_universe_prices"):
        strategy._cached_universe_prices = {}
    # --- END PERFORMANCE OPTIMIZATION ---

    # OPTIMIZATION: Pre-allocate numpy array for signals
    num_assets = len(universe_tickers)
    signals_arr = np.zeros((len(rebalance_dates), num_assets))

    for i, current_rebalance_date in enumerate(rebalance_dates):
        if has_timed_out():
            logger.warning("Timeout reached during meta strategy run. Halting signal generation.")
            break

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Generating meta strategy signals for date: {current_rebalance_date}")

        # SLICING OPTIMIZATION: Use pre-computed boolean mask for better performance
        date_mask = date_masks[current_rebalance_date]

        # Apply boolean mask (faster than iloc with computed indices)
        all_historical_data_for_strat = asset_data_view.loc[date_mask]
        benchmark_historical_data_for_strat = benchmark_data_view.loc[date_mask]

        non_universe_historical_data_for_strat = pd.DataFrame()
        if non_universe_data_view is not None:
            non_universe_historical_data_for_strat = non_universe_data_view.loc[date_mask]

        # Generate signals for the current date
        current_weights_df = None
        try:
            # Call the meta strategy's generate_signals method

            current_weights = strategy.generate_signals(
                all_historical_data=all_historical_data_for_strat,
                benchmark_historical_data=benchmark_historical_data_for_strat,
                non_universe_historical_data=non_universe_historical_data_for_strat,
                current_date=current_rebalance_date,
            )

            if current_weights is not None and not current_weights.empty:
                # For meta strategies, weights can be a DataFrame or Series
                if isinstance(current_weights, pd.Series):
                    current_weights_df = pd.DataFrame([current_weights])
                    current_weights_df.index = pd.DatetimeIndex([current_rebalance_date])
                else:
                    current_weights_df = current_weights.copy()
                    if len(
                        current_weights_df.index
                    ) == 1 and not pd.api.types.is_datetime64_any_dtype(current_weights_df.index):
                        current_weights_df.index = pd.DatetimeIndex([current_rebalance_date])

                # Update the meta strategy with the current weights and prices
                try:
                    # PERFORMANCE: Use cached close prices with optimized MultiIndex access
                    if current_rebalance_date not in strategy._cached_close_prices:
                        # Cache miss - compute prices
                        if has_close_field and hasattr(strategy, "_close_field_indices"):
                            # Fast path: Use pre-computed indices for direct column access
                            row = price_data_daily_ohlc.loc[current_rebalance_date]
                            close_indices = strategy._close_field_indices["Close"]

                            # Check if we have valid indices
                            if close_indices:
                                # Extract columns using pre-computed indices
                                # This avoids the expensive xs operation
                                cols = [row.iloc[i] for i in close_indices]
                                close_values = np.array(cols)

                                # Get ticker names from the MultiIndex
                                ticker_level = strategy._ticker_level_idx
                                tickers = [
                                    price_data_daily_ohlc.columns[i][ticker_level]
                                    for i in close_indices
                                ]

                                # Create Series with tickers as index
                                strategy._cached_close_prices[current_rebalance_date] = pd.Series(
                                    close_values, index=tickers
                                )
                            else:
                                # Fallback to xs if optimization not possible
                                strategy._cached_close_prices[current_rebalance_date] = (
                                    price_data_daily_ohlc.loc[current_rebalance_date].xs(
                                        "Close", level="Field"
                                    )
                                )
                        elif has_close_field:
                            # Standard xs operation if we don't have pre-computed indices
                            strategy._cached_close_prices[current_rebalance_date] = (
                                price_data_daily_ohlc.loc[current_rebalance_date].xs(
                                    "Close", level="Field"
                                )
                            )
                        elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
                            strategy._cached_close_prices[current_rebalance_date] = (
                                price_data_daily_ohlc.loc[current_rebalance_date]
                            )
                        else:
                            try:
                                strategy._cached_close_prices[current_rebalance_date] = (
                                    price_data_daily_ohlc.loc[current_rebalance_date].xs(
                                        "Close", level=-1
                                    )
                                )
                            except Exception:
                                strategy._cached_close_prices[current_rebalance_date] = (
                                    price_data_daily_ohlc.loc[current_rebalance_date].iloc[
                                        : len(universe_tickers)
                                    ]
                                )

                    # Use cached prices
                    current_prices = strategy._cached_close_prices[current_rebalance_date]

                    # PERFORMANCE: Use cached universe prices with optimized subset check
                    if current_rebalance_date not in strategy._cached_universe_prices:
                        # Cache miss - compute universe prices
                        # Fast path: check if all tickers are in current_prices using set operations
                        universe_set = set(universe_tickers)
                        prices_set = set(current_prices.index)

                        if universe_set.issubset(prices_set):
                            # All tickers present - direct indexing (fastest)
                            # Use numpy-based indexing for better performance
                            universe_prices = current_prices.loc[universe_tickers]
                        else:
                            # Some tickers missing - reindex needed
                            # Use optimized reindex with pre-allocation
                            universe_prices = pd.Series(
                                index=universe_tickers, dtype=current_prices.dtype
                            )

                            # Fill values for existing tickers (faster than reindex)
                            common_tickers = universe_set.intersection(prices_set)
                            for ticker in common_tickers:
                                universe_prices[ticker] = current_prices[ticker]

                            # Apply ffill only if there are NaN values
                            if universe_prices.isna().any():
                                universe_prices = universe_prices.ffill()

                        # Cache the result
                        strategy._cached_universe_prices[current_rebalance_date] = universe_prices

                    # Use cached universe prices
                    universe_prices = strategy._cached_universe_prices[current_rebalance_date]

                    # Get the weights from the DataFrame (first row)
                    weights_series = (
                        current_weights_df.iloc[0]
                        if len(current_weights_df) > 0
                        else pd.Series(0, index=universe_tickers)
                    )

                    # Update the meta strategy with the current weights and prices
                    strategy.update(current_rebalance_date, weights_series, universe_prices)

                except Exception as e:
                    logger.warning(
                        f"Error updating meta strategy for date {current_rebalance_date}: {e}"
                    )

            if current_weights_df is not None:
                aligned_weights = current_weights_df.reindex(columns=universe_tickers).fillna(0.0)
                if not aligned_weights.empty:
                    signals_arr[i, :] = aligned_weights.iloc[0].values.astype(np.float64)
                else:
                    signals_arr[i, :] = 0.0
            else:
                signals_arr[i, :] = 0.0

        except Exception as e:
            logger.warning(
                f"Error generating signals for meta strategy on date {current_rebalance_date}: {e}"
            )
            signals_arr[i, :] = 0.0

    # Create a single DataFrame from the numpy array at the end
    signals = pd.DataFrame(signals_arr, index=rebalance_dates, columns=universe_tickers)
    signals.fillna(0.0, inplace=True)

    return signals


@enforce_strategy_parameter
def rebalance(signals, frequency="M"):
    """
    Rebalance signals to the specified frequency.
    """
    if signals.empty:
        return signals

    if frequency == "D":
        return signals  # Daily signals don't need rebalancing

    # For monthly/other frequencies, resample and forward-fill
    # This ensures we have a weight for each rebalance date
    resampled = signals.resample("ME" if frequency == "M" else frequency).last()
    return resampled.ffill()


@enforce_strategy_parameter
def size_positions(
    signals,
    scenario_config,
    price_data_daily_ohlc=None,
    universe_tickers=None,
    benchmark_ticker=None,
    global_config=None,
    strategy=None,
):
    """
    Apply position sizing to signals.

    This function is kept for backward compatibility.
    All sizing is now handled directly by the strategy provider interfaces.
    """
    # No additional sizing needed; strategies now provide properly sized signals
    return signals
