import inspect
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, cast

from ..interfaces.enforcement import enforce_strategy_parameter
from ..interfaces.strategy_resolver import StrategyResolverFactory
from .meta_execution import MetaExecutionMode, portfolio_execution_mode_for_strategy
from ..optimization.feature_store import FeatureStore
from ..optimization.market_data_panel import MarketDataPanel
from ..optimization.signal_cache import (
    SignalCache,
    compute_signal_matrix_cache_digest,
    default_never_timed_out,
    index_fingerprint,
    signal_affecting_param_subset,
    strategy_allows_signal_matrix_cache,
)
from ..optimization.strategy_data_context import StrategyDataContext
from ..strategies._core.target_generation import StrategyContext


# Removed legacy position sizer imports - now using strategy provider interfaces

logger = logging.getLogger(__name__)


class StrategyTargetGenerationError(TypeError):
    """Raised when a standard (non-meta) strategy cannot produce a deterministic target-weight matrix.

    Meta strategies bypass target-weight generation here; see ``MetaExecutionMode`` in
    ``portfolio_backtester.backtester_logic.meta_execution``.
    """


def _effective_global_config_for_universe(
    strategy: Any, global_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if global_config is not None:
        return dict(global_config)
    gc = getattr(strategy, "global_config", None)
    if isinstance(gc, dict):
        return dict(gc)
    return {}


def _cache_attachment_target(
    global_config: Optional[Dict[str, Any]], strategy: Any
) -> Dict[str, Any]:
    if global_config is not None:
        return global_config
    gc = getattr(strategy, "global_config", None)
    return gc if isinstance(gc, dict) else {}


def _strategy_supports_method_dynamic_universe(strategy: Any) -> bool:
    try:
        provider = strategy.get_universe_provider()
    except Exception:
        return False
    supports_fn = getattr(provider, "supports_dynamic_universe", None)
    if not callable(supports_fn):
        return False
    try:
        # Use `is True` (not just truthy) so MagicMock auto-returns don't
        # accidentally activate the dynamic path in tests.
        return supports_fn() is True
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


def _resolve_signal_scan_window(
    canonical_config: Any,
    price_index: pd.DatetimeIndex,
) -> tuple[
    pd.Timestamp,
    pd.Timestamp,
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
]:
    """Clip OHLC bounds by scenario/WFO dates (same semantics as standard ``generate_signals``)."""

    start_date = price_index.min()
    end_date = price_index.max()

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

    return (
        start_date,
        end_date,
        scenario_start_date,
        scenario_end_date,
        wfo_start_date,
        wfo_end_date,
    )


def _expanding_iloc_ends(
    idx: pd.Index, rebalance_dates: Any
) -> tuple[Optional[np.ndarray], Optional[dict[Any, Any]]]:
    """Compute ``iloc[:end]`` slice ends for expanding windows, or legacy boolean masks.

    Returns:
        ``(ends, None)`` when the index supports ``searchsorted``; otherwise
        ``(None, date_masks)`` with boolean masks keyed by rebalance date.
    """
    ends: Optional[np.ndarray] = None
    if isinstance(idx, pd.DatetimeIndex) and idx.is_monotonic_increasing and idx.is_unique:
        try:
            ends = np.asarray(idx.searchsorted(rebalance_dates, side="right"), dtype=np.int64)
        except Exception:  # noqa: BLE001
            ends = None

    if ends is not None:
        return ends, None

    masks: dict[Any, Any] = {d: idx <= d for d in rebalance_dates}
    return None, masks


def _try_build_strategy_context_panel(
    price_data_daily_ohlc: pd.DataFrame,
) -> Optional[MarketDataPanel]:
    try:
        seed = pd.DataFrame(index=price_data_daily_ohlc.index)
        return MarketDataPanel.from_daily_ohlc_and_returns(price_data_daily_ohlc, seed)
    except Exception:  # noqa: BLE001
        logger.debug(
            "Could not build MarketDataPanel for strategy data context",
            exc_info=True,
        )
        return None


def _legacy_per_date_signal_generation_loop(
    *,
    strategy: Any,
    timing_controller: Any,
    has_timed_out: Any,
    rebalance_dates: pd.DatetimeIndex,
    universe_tickers: List[str],
    benchmark_ticker: str,
    wfo_start_date: Optional[pd.Timestamp],
    wfo_end_date: Optional[pd.Timestamp],
    asset_data_view: pd.DataFrame,
    benchmark_data_view: pd.DataFrame,
    non_universe_data_view: Optional[pd.DataFrame],
    use_pit_universe_slice: bool,
    effective_global_config: Dict[str, Any],
    is_multi_index: bool,
    has_non_universe_param: bool,
    inject_data_context: bool,
    full_strategy_panel: Optional[MarketDataPanel],
    strategy_feature_store: Optional[FeatureStore],
    close_panel_full: Optional[pd.DataFrame],
    has_close_field: bool,
    price_data_daily_ohlc: pd.DataFrame,
    use_sparse_nan_for_inactive_rows: bool,
    expanding_ends: Optional[np.ndarray],
    date_masks: Optional[dict[Any, Any]],
) -> pd.DataFrame:
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
                    "Timing controller skipped signal generation for date: %s",
                    current_rebalance_date,
                )
            continue

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Generating signals for date: %s", current_rebalance_date)

        if expanding_ends is not None:
            end = int(expanding_ends[i])
            all_historical_data_for_strat = asset_data_view.iloc[:end]
            benchmark_historical_data_for_strat = benchmark_data_view.iloc[:end]
            non_universe_historical_data_for_strat = (
                non_universe_data_view.iloc[:end]
                if non_universe_data_view is not None
                else pd.DataFrame()
            )
        else:
            assert date_masks is not None
            date_mask = date_masks[current_rebalance_date]
            all_historical_data_for_strat = asset_data_view.loc[date_mask]
            benchmark_historical_data_for_strat = benchmark_data_view.loc[date_mask]
            non_universe_historical_data_for_strat = pd.DataFrame()
            if non_universe_data_view is not None:
                non_universe_historical_data_for_strat = non_universe_data_view.loc[date_mask]
            end = int(date_mask.sum())

        strat_asset_hist = all_historical_data_for_strat
        if use_pit_universe_slice:
            pit_syms = _pit_symbols_for_rebalance_date(
                strategy, effective_global_config, current_rebalance_date
            )
            if pit_syms:
                strat_asset_hist = _slice_asset_hist_by_tickers(
                    all_historical_data_for_strat, pit_syms, is_multi_index
                )

        dc_args: Dict[str, Any] = {}
        if inject_data_context:
            assert full_strategy_panel is not None
            assert strategy_feature_store is not None
            row_ix = max(0, end - 1) if end > 0 else 0
            dc_args["data_context"] = StrategyDataContext(
                panel=full_strategy_panel,
                feature_store=strategy_feature_store,
                window_bounds=None,
                current_row_ix=row_ix,
                current_date=current_rebalance_date,
                universe_tickers=tuple(universe_tickers),
                benchmark_ticker=str(benchmark_ticker),
            )

        if has_non_universe_param:
            current_weights_df = strategy.generate_signals(
                all_historical_data=strat_asset_hist,
                benchmark_historical_data=benchmark_historical_data_for_strat,
                non_universe_historical_data=non_universe_historical_data_for_strat,
                current_date=current_rebalance_date,
                start_date=wfo_start_date,
                end_date=wfo_end_date,
                **dc_args,
            )
        else:
            current_weights_df = strategy.generate_signals(
                all_historical_data=strat_asset_hist,
                benchmark_historical_data=benchmark_historical_data_for_strat,
                current_date=current_rebalance_date,
                start_date=wfo_start_date,
                end_date=wfo_end_date,
                **dc_args,
            )

        if current_weights_df is not None and not current_weights_df.empty:
            if len(current_weights_df) > 0:
                current_weights_series = current_weights_df.iloc[0]
                timing_controller.update_signal_state(
                    current_rebalance_date, current_weights_series
                )

                try:
                    if current_rebalance_date not in strategy._cached_close_prices:
                        if close_panel_full is not None and end > 0:
                            strategy._cached_close_prices[current_rebalance_date] = (
                                close_panel_full.iloc[end - 1]
                            )
                        elif has_close_field and hasattr(strategy, "_close_field_indices"):
                            row = price_data_daily_ohlc.loc[current_rebalance_date]
                            close_indices = strategy._close_field_indices["Close"]

                            if close_indices:
                                cols = [row.iloc[i] for i in close_indices]
                                close_values = np.array(cols)

                                ticker_level = strategy._ticker_level_idx
                                tickers = [
                                    price_data_daily_ohlc.columns[i][ticker_level]
                                    for i in close_indices
                                ]

                                strategy._cached_close_prices[current_rebalance_date] = pd.Series(
                                    close_values, index=tickers
                                )
                            else:
                                strategy._cached_close_prices[current_rebalance_date] = (
                                    price_data_daily_ohlc.loc[current_rebalance_date].xs(
                                        "Close", level="Field"
                                    )
                                )
                        elif has_close_field:
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

                    current_prices = strategy._cached_close_prices[current_rebalance_date]

                    if current_rebalance_date not in strategy._cached_universe_prices:
                        universe_set = set(universe_tickers)
                        prices_set = set(current_prices.index)

                        if universe_set.issubset(prices_set):
                            universe_prices = current_prices.loc[universe_tickers]
                        else:
                            universe_prices = pd.Series(
                                index=universe_tickers, dtype=current_prices.dtype
                            )

                            common_tickers = universe_set.intersection(prices_set)
                            for ticker in common_tickers:
                                universe_prices[ticker] = current_prices[ticker]

                            if universe_prices.isna().any():
                                universe_prices = universe_prices.ffill()

                        strategy._cached_universe_prices[current_rebalance_date] = universe_prices

                    universe_prices = strategy._cached_universe_prices[current_rebalance_date]

                    timing_controller.update_position_state(
                        current_rebalance_date, current_weights_series, universe_prices
                    )

                except Exception as e:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Could not update position state for %s: %s",
                            current_rebalance_date,
                            e,
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
    return signals


class LegacyGenerateSignalsAdapter:
    """Delegates ``generate_target_weights`` to sequential legacy ``generate_signals`` scans."""

    def __init__(self, wrapped: Any, *, global_config: Optional[Dict[str, Any]] = None) -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_global_config", global_config)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def generate_target_weights(self, context: StrategyContext) -> pd.DataFrame:
        strategy = object.__getattribute__(self, "_wrapped")
        raw_gc = object.__getattribute__(self, "_global_config")
        global_cfg = dict(raw_gc) if isinstance(raw_gc, dict) else {}
        price_data_daily_ohlc = context.full_price_panel
        if price_data_daily_ohlc is None:
            price_data_daily_ohlc = context.asset_data

        universe_tickers = list(context.universe_tickers)
        benchmark_ticker = context.benchmark_ticker
        rebalance_dates = context.rebalance_dates
        asset_data_view = context.asset_data
        benchmark_data_view = context.benchmark_data
        nu_df = context.non_universe_data
        non_universe_data_view = nu_df if len(nu_df.columns) > 0 else None
        wfo_start_date = context.wfo_start_date
        wfo_end_date = context.wfo_end_date
        use_sparse_nan_for_inactive_rows = context.use_sparse_nan_for_inactive_rows

        timing_controller = strategy.get_timing_controller()
        timing_controller.reset_state()

        effective_global_config = _effective_global_config_for_universe(strategy, global_cfg)
        use_strategy_data_context = bool(
            (effective_global_config.get("feature_flags") or {}).get("strategy_data_context", False)
        )
        full_strategy_panel: Optional[MarketDataPanel] = None
        strategy_feature_store: Optional[FeatureStore] = None
        if use_strategy_data_context:
            full_strategy_panel = _try_build_strategy_context_panel(price_data_daily_ohlc)
            if full_strategy_panel is not None:
                strategy_feature_store = FeatureStore(full_strategy_panel)

        use_pit_universe_slice = _strategy_supports_method_dynamic_universe(strategy)

        sig = inspect.signature(strategy.generate_signals)
        has_non_universe_param = "non_universe_historical_data" in sig.parameters
        has_explicit_data_context = "data_context" in sig.parameters
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        inject_data_context = bool(
            use_strategy_data_context
            and full_strategy_panel is not None
            and strategy_feature_store is not None
            and (has_explicit_data_context or has_var_keyword)
        )

        has_close_field = False
        if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
            has_close_field = "Close" in price_data_daily_ohlc.columns.get_level_values("Field")

            if not hasattr(strategy, "_close_field_indices"):
                strategy._close_field_indices = {}
                field_level_idx = price_data_daily_ohlc.columns.names.index("Field")
                field_values = price_data_daily_ohlc.columns.get_level_values(field_level_idx)
                close_indices = [i for i, val in enumerate(field_values) if val == "Close"]
                strategy._close_field_indices["Close"] = close_indices

                ticker_level_idx = price_data_daily_ohlc.columns.names.index("Ticker")
                strategy._ticker_level_idx = ticker_level_idx

        expanding_ends, date_masks = _expanding_iloc_ends(asset_data_view.index, rebalance_dates)

        close_panel_full: Optional[pd.DataFrame] = None
        if has_close_field and isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
            try:
                close_panel_full = cast(
                    pd.DataFrame, price_data_daily_ohlc.xs("Close", level="Field", axis=1)
                )
            except Exception:  # noqa: BLE001
                close_panel_full = None

        if not hasattr(strategy, "_cached_close_prices"):
            strategy._cached_close_prices = {}
        if not hasattr(strategy, "_cached_universe_prices"):
            strategy._cached_universe_prices = {}

        is_multi_index = (
            isinstance(price_data_daily_ohlc.columns, pd.MultiIndex)
            and "Ticker" in price_data_daily_ohlc.columns.names
        )

        return _legacy_per_date_signal_generation_loop(
            strategy=strategy,
            timing_controller=timing_controller,
            has_timed_out=default_never_timed_out,
            rebalance_dates=rebalance_dates,
            universe_tickers=universe_tickers,
            benchmark_ticker=str(benchmark_ticker),
            wfo_start_date=wfo_start_date,
            wfo_end_date=wfo_end_date,
            asset_data_view=asset_data_view,
            benchmark_data_view=benchmark_data_view,
            non_universe_data_view=non_universe_data_view,
            use_pit_universe_slice=use_pit_universe_slice,
            effective_global_config=effective_global_config,
            is_multi_index=is_multi_index,
            has_non_universe_param=has_non_universe_param,
            inject_data_context=inject_data_context,
            full_strategy_panel=full_strategy_panel,
            strategy_feature_store=strategy_feature_store,
            close_panel_full=close_panel_full,
            has_close_field=has_close_field,
            price_data_daily_ohlc=price_data_daily_ohlc,
            use_sparse_nan_for_inactive_rows=use_sparse_nan_for_inactive_rows,
            expanding_ends=expanding_ends,
            date_masks=date_masks,
        )


def _call_generate_target_weights(
    strategy: Any,
    *,
    asset_data_view: pd.DataFrame,
    benchmark_data_view: pd.DataFrame,
    non_universe_data_view: Optional[pd.DataFrame],
    rebalance_dates: pd.DatetimeIndex,
    universe_tickers: List[str],
    benchmark_ticker: str,
    wfo_start_date: Optional[pd.Timestamp],
    wfo_end_date: Optional[pd.Timestamp],
    use_sparse_nan_for_inactive_rows: bool,
    price_data_daily_ohlc: pd.DataFrame,
) -> pd.DataFrame:
    """Invoke ``strategy.generate_target_weights`` for deterministic full-scan targets."""
    target_fn = getattr(strategy, "generate_target_weights", None)
    if not callable(target_fn):
        raise StrategyTargetGenerationError(
            f"{type(strategy).__name__} must implement callable generate_target_weights; "
            "wrap legacy per-date strategies with LegacyGenerateSignalsAdapter."
        )
    nu = non_universe_data_view if non_universe_data_view is not None else pd.DataFrame()
    ctx = StrategyContext.from_standard_inputs(
        asset_data=asset_data_view,
        benchmark_data=benchmark_data_view,
        non_universe_data=nu,
        rebalance_dates=rebalance_dates,
        universe_tickers=universe_tickers,
        benchmark_ticker=str(benchmark_ticker),
        wfo_start_date=wfo_start_date,
        wfo_end_date=wfo_end_date,
        use_sparse_nan_for_inactive_rows=use_sparse_nan_for_inactive_rows,
        full_price_panel=price_data_daily_ohlc,
    )
    try:
        weights = target_fn(ctx)
    except StrategyTargetGenerationError:
        raise
    except Exception as exc:
        raise StrategyTargetGenerationError(
            f"{type(strategy).__name__}.generate_target_weights failed"
        ) from exc
    if weights is None:
        raise StrategyTargetGenerationError(
            f"{type(strategy).__name__}.generate_target_weights returned None; "
            "production strategies must return a DataFrame."
        )
    if not isinstance(weights, pd.DataFrame):
        raise StrategyTargetGenerationError(
            f"{type(strategy).__name__}.generate_target_weights must return a pandas.DataFrame, "
            f"got {type(weights).__name__}"
        )
    return weights.reindex(index=rebalance_dates, columns=universe_tickers)


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
        effective_global = (
            global_config
            if global_config is not None
            else (getattr(strategy, "global_config", None) or {})
        )
        canonical_config = normalizer.normalize(
            scenario=scenario_config, global_config=effective_global
        )
    else:
        canonical_config = scenario_config

    strategy_resolver = StrategyResolverFactory.create()
    if (
        portfolio_execution_mode_for_strategy(strategy, strategy_resolver=strategy_resolver)
        is MetaExecutionMode.TRADE_AGGREGATION
    ):
        return _generate_meta_strategy_signals(
            strategy,
            canonical_config,
            price_data_daily_ohlc,
            universe_tickers,
            benchmark_ticker,
            has_timed_out,
            global_config=global_config,
        )

    # Standard strategy signal generation
    timing_controller = strategy.get_timing_controller()
    timing_controller.reset_state()

    (
        start_date,
        end_date,
        scenario_start_date,
        scenario_end_date,
        wfo_start_date,
        wfo_end_date,
    ) = _resolve_signal_scan_window(canonical_config, price_data_daily_ohlc.index)

    rebalance_dates = timing_controller.get_rebalance_dates(
        start_date=start_date,
        end_date=end_date,
        available_dates=price_data_daily_ohlc.index,
        strategy_context=strategy,
    )

    timing_mode = _scenario_timing_mode(canonical_config)
    use_sparse_nan_for_inactive_rows = timing_mode == "signal_based"

    # Honor scenario-configured daily rebalance when frequency is D (meta strategies already returned)
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

    use_pit_universe_slice = _strategy_supports_method_dynamic_universe(strategy)

    cache_attachment = _cache_attachment_target(global_config, strategy)
    feature_flags_all = (
        (cache_attachment.get("feature_flags") or {}) if isinstance(cache_attachment, dict) else {}
    )
    use_signal_cache = bool(feature_flags_all.get("signal_cache", False))
    idx_for_cache = price_data_daily_ohlc.index
    idx_ok = (
        isinstance(idx_for_cache, pd.DatetimeIndex)
        and idx_for_cache.is_monotonic_increasing
        and idx_for_cache.is_unique
    )
    cache_store: Optional[SignalCache] = None
    cache_digest: Optional[str] = None
    if (
        use_signal_cache
        and idx_ok
        and has_timed_out is default_never_timed_out
        and not use_pit_universe_slice
        and strategy_allows_signal_matrix_cache(strategy)
    ):
        existing = cache_attachment.get("_signal_matrix_cache")
        if not isinstance(existing, SignalCache):
            existing = SignalCache()
            cache_attachment["_signal_matrix_cache"] = existing
        cache_store = existing
        non_uni_key = tuple(sorted(non_universe_tickers)) if non_universe_tickers else tuple()
        rd_ns = tuple(int(pd.Timestamp(d).value) for d in rebalance_dates)
        params_slice = signal_affecting_param_subset(strategy, canonical_config.strategy_params)
        scenario_bounds_key = {
            "start_date": str(scenario_start_date) if scenario_start_date is not None else None,
            "end_date": str(scenario_end_date) if scenario_end_date is not None else None,
            "wfo_start_date": str(wfo_start_date) if wfo_start_date is not None else None,
            "wfo_end_date": str(wfo_end_date) if wfo_end_date is not None else None,
        }
        strat_qual = f"{type(strategy).__module__}.{type(strategy).__qualname__}"
        ff_key = dict(feature_flags_all) if isinstance(feature_flags_all, dict) else {}
        cache_digest = compute_signal_matrix_cache_digest(
            strategy_module_qualname=strat_qual,
            universe_tickers=tuple(str(t) for t in universe_tickers),
            benchmark_ticker=str(benchmark_ticker),
            non_universe_tickers=non_uni_key,
            rebalance_dates_ns=rd_ns,
            use_sparse_nan_for_inactive_rows=use_sparse_nan_for_inactive_rows,
            timing_mode=timing_mode,
            timing_config=canonical_config.timing_config,
            scenario_bounds=scenario_bounds_key,
            strategy_params_slice=params_slice,
            feature_flags=ff_key,
            index_fp=index_fingerprint(price_data_daily_ohlc.index),
        )
        cached_df = cache_store.get(cache_digest)
        if cached_df is not None:
            return cached_df.copy()

    dense_targets = _call_generate_target_weights(
        strategy,
        asset_data_view=asset_data_view,
        benchmark_data_view=benchmark_data_view,
        non_universe_data_view=non_universe_data_view,
        rebalance_dates=rebalance_dates,
        universe_tickers=list(universe_tickers),
        benchmark_ticker=str(benchmark_ticker),
        wfo_start_date=wfo_start_date,
        wfo_end_date=wfo_end_date,
        use_sparse_nan_for_inactive_rows=use_sparse_nan_for_inactive_rows,
        price_data_daily_ohlc=price_data_daily_ohlc,
    )
    if not use_sparse_nan_for_inactive_rows:
        dense_targets = dense_targets.fillna(0.0)

    if cache_store is not None and cache_digest is not None:
        cache_store.put(cache_digest, dense_targets.copy(deep=True))

    return dense_targets


def _generate_meta_strategy_signals(
    strategy,
    scenario_config,
    price_data_daily_ohlc,
    universe_tickers,
    benchmark_ticker,
    has_timed_out,
    global_config: Optional[Dict[str, Any]] = None,
):
    """
    Generate signals for meta strategies that aggregate child strategies.
    """
    # Initialize the meta strategy with the scenario config
    if hasattr(strategy, "initialize"):
        strategy.initialize(scenario_config)

    timing_controller = strategy.get_timing_controller()
    timing_controller.reset_state()

    (
        start_date,
        end_date,
        _scenario_start_date,
        _scenario_end_date,
        wfo_start_date,
        wfo_end_date,
    ) = _resolve_signal_scan_window(scenario_config, price_data_daily_ohlc.index)

    rebalance_dates = timing_controller.get_rebalance_dates(
        start_date=start_date,
        end_date=end_date,
        available_dates=price_data_daily_ohlc.index,
        strategy_context=strategy,
    )

    try:
        configured_freq = scenario_config.get("timing_config", {}).get("rebalance_frequency")
        if configured_freq == "D":
            rebalance_dates = price_data_daily_ohlc.index
    except Exception:
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

    effective_global_config_meta = _effective_global_config_for_universe(strategy, global_config)
    use_strategy_data_context_meta = bool(
        (effective_global_config_meta.get("feature_flags") or {}).get(
            "strategy_data_context", False
        )
    )
    full_strategy_panel_meta: Optional[MarketDataPanel] = None
    strategy_feature_store_meta: Optional[FeatureStore] = None
    if use_strategy_data_context_meta:
        full_strategy_panel_meta = _try_build_strategy_context_panel(price_data_daily_ohlc)
        if full_strategy_panel_meta is not None:
            strategy_feature_store_meta = FeatureStore(full_strategy_panel_meta)

    sig_meta = inspect.signature(strategy.generate_signals)
    has_non_universe_meta = "non_universe_historical_data" in sig_meta.parameters
    has_explicit_data_context_meta = "data_context" in sig_meta.parameters
    has_var_keyword_meta = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_meta.parameters.values()
    )
    inject_data_context_meta = bool(
        use_strategy_data_context_meta
        and full_strategy_panel_meta is not None
        and strategy_feature_store_meta is not None
        and (has_explicit_data_context_meta or has_var_keyword_meta)
    )

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

    expanding_ends, date_masks = _expanding_iloc_ends(asset_data_view.index, rebalance_dates)

    close_panel_full: Optional[pd.DataFrame] = None
    if has_close_field and isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
        try:
            close_panel_full = price_data_daily_ohlc.xs("Close", level="Field", axis=1)
        except Exception:  # noqa: BLE001
            close_panel_full = None

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

        should_generate = timing_controller.should_generate_signal(
            current_date=current_rebalance_date, strategy_context=strategy
        )
        if not should_generate:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Timing controller skipped meta signal generation for date: %s",
                    current_rebalance_date,
                )
            continue

        if expanding_ends is not None:
            end = int(expanding_ends[i])
            all_historical_data_for_strat = asset_data_view.iloc[:end]
            benchmark_historical_data_for_strat = benchmark_data_view.iloc[:end]
            non_universe_historical_data_for_strat = (
                non_universe_data_view.iloc[:end]
                if non_universe_data_view is not None
                else pd.DataFrame()
            )
        else:
            assert date_masks is not None
            date_mask = date_masks[current_rebalance_date]
            all_historical_data_for_strat = asset_data_view.loc[date_mask]
            benchmark_historical_data_for_strat = benchmark_data_view.loc[date_mask]
            non_universe_historical_data_for_strat = pd.DataFrame()
            if non_universe_data_view is not None:
                non_universe_historical_data_for_strat = non_universe_data_view.loc[date_mask]
            end = int(date_mask.sum())

        # Generate signals for the current date
        current_weights_df = None
        try:
            # Call the meta strategy's generate_signals method

            dc_args_meta: Dict[str, Any] = {}
            if inject_data_context_meta:
                assert full_strategy_panel_meta is not None
                assert strategy_feature_store_meta is not None
                row_ix_meta = max(0, end - 1) if end > 0 else 0
                dc_args_meta["data_context"] = StrategyDataContext(
                    panel=full_strategy_panel_meta,
                    feature_store=strategy_feature_store_meta,
                    window_bounds=None,
                    current_row_ix=row_ix_meta,
                    current_date=current_rebalance_date,
                    universe_tickers=tuple(universe_tickers),
                    benchmark_ticker=benchmark_ticker,
                )

            gen_kw: Dict[str, Any] = {
                **dc_args_meta,
                "all_historical_data": all_historical_data_for_strat,
                "benchmark_historical_data": benchmark_historical_data_for_strat,
                "current_date": current_rebalance_date,
                "start_date": wfo_start_date,
                "end_date": wfo_end_date,
            }
            if has_non_universe_meta:
                gen_kw["non_universe_historical_data"] = non_universe_historical_data_for_strat

            current_weights = strategy.generate_signals(**gen_kw)

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
                        if close_panel_full is not None and end > 0:
                            strategy._cached_close_prices[current_rebalance_date] = (
                                close_panel_full.iloc[end - 1]
                            )
                        elif has_close_field and hasattr(strategy, "_close_field_indices"):
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
