import logging
from typing import Any, Mapping, cast

import pandas as pd
import numpy as np

from ..interfaces.strategy_resolver import StrategyResolverFactory
from ..portfolio.rebalancing import rebalance_to_first_event_per_period
from ..timing.config_validator import TimingConfigValidator
from ..timing.trade_execution_timing import (
    TRADE_EXECUTION_TIMING_DEFAULT,
    map_sparse_target_weights_to_execution_dates,
)
from ..trading.trade_tracker import TradeTracker
from ..trading.unified_commission_calculator import get_unified_commission_calculator
from ..simulation.kernel import simulate_portfolio
from .meta_execution import (
    MetaExecutionMode,
    attach_portfolio_execution_model,
    portfolio_execution_mode_for_strategy,
)
from .portfolio_simulation_input import (
    build_portfolio_simulation_input,
    extract_open_frame_from_ohlc,
    market_panel_aligns_with_ohlc,
    prepare_close_arrays_for_simulation,
    prepare_open_arrays_for_simulation,
)


logger = logging.getLogger(__name__)


def _timing_config_as_mapping(scenario_config: object) -> Mapping[str, Any] | None:
    if scenario_config is None:
        return None
    if isinstance(scenario_config, dict):
        if "timing_config" not in scenario_config:
            return None
        tc = scenario_config["timing_config"]
    else:
        tc = getattr(scenario_config, "timing_config", None)
    if tc is None:
        return None
    if isinstance(tc, dict):
        return tc
    if hasattr(tc, "get"):
        return cast(Mapping[str, Any], tc)
    return None


def _sparse_targets_after_time_based_rebalance(
    sized_signals: pd.DataFrame,
    scenario_config: object,
) -> pd.DataFrame:
    tc = _timing_config_as_mapping(scenario_config)
    if tc is None:
        return sized_signals
    mode_raw = tc.get("mode") if hasattr(tc, "get") else None
    mode = str(mode_raw) if mode_raw else "time_based"
    if mode != "time_based":
        return sized_signals
    freq_raw = tc.get("rebalance_frequency") if hasattr(tc, "get") else None
    frequency = str(freq_raw) if freq_raw else "M"
    return rebalance_to_first_event_per_period(sized_signals, frequency)


def _resolve_trade_execution_timing_for_portfolio(scenario_config: object, strategy: Any) -> str:
    if strategy is not None:
        return str(strategy.get_trade_execution_timing())
    if scenario_config is None:
        return TRADE_EXECUTION_TIMING_DEFAULT
    tc = None
    if isinstance(scenario_config, dict):
        tc = scenario_config.get("timing_config")
    else:
        tc = getattr(scenario_config, "timing_config", None)
    raw = None
    if isinstance(tc, dict):
        raw = tc.get("trade_execution_timing")
    elif tc is not None and hasattr(tc, "get"):
        raw = tc.get("trade_execution_timing")
    if raw is None:
        return TRADE_EXECUTION_TIMING_DEFAULT
    errs = TimingConfigValidator.validate_trade_execution_timing(raw)
    if errs:
        raise ValueError(errs[0])
    return str(raw)


def _sized_signals_to_weights_daily(
    sized_signals: pd.DataFrame,
    universe_tickers: list[str],
    daily_index: pd.Index,
) -> pd.DataFrame:
    """Expand sparse event targets to a daily index.

    Column-wise ``ffill`` before ``fillna(0)`` preserves the last explicit targets across
    rows that are entirely NaN (e.g. skipped signal_based scans). Leading NaNs become
    zero exposure before reindexing to the full session calendar.
    """
    weights_monthly = sized_signals.copy().reindex(columns=universe_tickers).ffill().fillna(0.0)
    return weights_monthly.reindex(daily_index, method="ffill")


def calculate_portfolio_returns(
    sized_signals,
    scenario_config,
    price_data_daily_ohlc,
    rets_daily,
    universe_tickers,
    global_config,
    track_trades=False,
    strategy=None,
    market_data_panel=None,
    *,
    include_signed_weights: bool = False,
):
    """Portfolio net returns for standard strategies via the canonical share/cash simulator.

    The returned ``pandas.Series`` includes ``attrs["portfolio_backtester.execution_model"]``
    set to ``MetaExecutionMode.CANONICAL_SHARE_CASH_SIMULATION.value`` for standard strategies,
    or ``MetaExecutionMode.TRADE_AGGREGATION.value`` for meta strategies (see ``meta_execution``).

    Meta strategies (``MetaExecutionMode.TRADE_AGGREGATION``) bypass that path and
    rebuild returns from aggregated sub-strategy trades; see ``meta_execution.py``.
    """
    strategy_resolver = StrategyResolverFactory.create()
    if (
        strategy is not None
        and portfolio_execution_mode_for_strategy(strategy, strategy_resolver=strategy_resolver)
        == MetaExecutionMode.TRADE_AGGREGATION
    ):
        return _calculate_meta_strategy_portfolio_returns(
            strategy,
            scenario_config,
            price_data_daily_ohlc,
            rets_daily,
            universe_tickers,
            global_config,
            track_trades,
            include_signed_weights=include_signed_weights,
        )

    logger.debug("sized_signals shape: %s", sized_signals.shape)
    logger.debug("sized_signals head:\n%s", sized_signals.head())

    if logger.isEnabledFor(logging.DEBUG):
        wm = sized_signals.copy().reindex(columns=universe_tickers).fillna(0.0)
        logger.debug("sized weights (pre-daily ffill) shape: %s", wm.shape)
        logger.debug("sized weights (pre-daily ffill) head:\n%s", wm.head())

    sized_for_timing = _sparse_targets_after_time_based_rebalance(sized_signals, scenario_config)
    tet = _resolve_trade_execution_timing_for_portfolio(scenario_config, strategy)
    execution_calendar = pd.DatetimeIndex(price_data_daily_ohlc.index)
    sized_for_daily = map_sparse_target_weights_to_execution_dates(
        sized_for_timing,
        trade_execution_timing=tet,
        calendar=execution_calendar,
        logger=logger,
    )

    weights_daily = _sized_signals_to_weights_daily(
        sized_for_daily, universe_tickers, price_data_daily_ohlc.index
    )

    # NOTE: weights_daily are targets; turnover is driven by ``rebalance_mask`` from sparse
    # execution rows (plus required day-0 entry when the mask is otherwise flat False).

    if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and (
        "Close" in price_data_daily_ohlc.columns.get_level_values(-1)
    ):
        close_prices_df = price_data_daily_ohlc.xs("Close", level="Field", axis=1)
        price_index = close_prices_df.index
    else:
        close_prices_df = price_data_daily_ohlc
        price_index = price_data_daily_ohlc.index

    valid_cols = [t for t in universe_tickers if t in close_prices_df.columns]

    resolved_panel = market_data_panel
    if resolved_panel is None and isinstance(global_config, dict):
        resolved_panel = global_config.get("market_data_panel")

    price_ix = pd.DatetimeIndex(price_index)
    close_arr, close_mask_arr = prepare_close_arrays_for_simulation(
        market_data_panel=resolved_panel,
        close_prices_df=close_prices_df,
        price_index=price_ix,
        valid_cols=valid_cols,
    )
    open_arr_np = None
    open_mask_np = None
    if tet == "next_bar_open":
        open_frame = extract_open_frame_from_ohlc(price_data_daily_ohlc)
        panel_has_opens = (
            resolved_panel is not None
            and market_panel_aligns_with_ohlc(resolved_panel, price_ix, valid_cols)
            and resolved_panel.open_np is not None
        )
        if open_frame is None and not panel_has_opens:
            raise ValueError(
                "trade_execution_timing=next_bar_open requires OHLC with MultiIndex Field "
                "including Open, or an aligned MarketDataPanel providing opens."
            )
        open_arr_np, open_mask_np = prepare_open_arrays_for_simulation(
            market_data_panel=resolved_panel,
            open_prices_df=open_frame,
            price_index=price_ix,
            valid_cols=valid_cols,
        )

    sparse_exec_targets = sized_for_daily.reindex(columns=valid_cols)
    sim_input = build_portfolio_simulation_input(
        weights_daily=weights_daily,
        price_index=price_ix,
        valid_cols=valid_cols,
        close_arr=close_arr,
        close_price_mask_arr=close_mask_arr,
        open_arr=open_arr_np,
        open_price_mask_arr=open_mask_np,
        sparse_execution_targets=sparse_exec_targets,
        trade_execution_timing=tet,
    )

    sim_result = simulate_portfolio(
        sim_input,
        global_config=global_config,
        scenario_config=scenario_config,
    )

    signed_weights_out: pd.DataFrame | None = None
    if include_signed_weights:
        close_aligned = close_prices_df.reindex(index=price_ix, columns=valid_cols).astype(float)
        pos_aligned = sim_result.positions.reindex(index=price_ix, columns=valid_cols).fillna(0.0)
        pv_aligned = sim_result.portfolio_values.reindex(price_ix).astype(float)
        pv_safe = pv_aligned.replace(0.0, np.nan)
        dollar_positions = pos_aligned.multiply(close_aligned, axis=0)
        signed_weights_out = dollar_positions.div(pv_safe, axis=0).fillna(0.0)

    returns_net_of_costs = attach_portfolio_execution_model(
        sim_result.daily_returns.astype(float),
        MetaExecutionMode.CANONICAL_SHARE_CASH_SIMULATION,
    )
    transaction_costs = pd.Series(
        sim_result.total_daily_transaction_cost_frac_of_reference_pv,
        index=price_ix,
        dtype=float,
    )

    merged_flags: dict = {}
    if isinstance(global_config, dict):
        merged_flags.update(global_config.get("feature_flags", {}) or {})
    if isinstance(scenario_config, dict):
        merged_flags.update(scenario_config.get("feature_flags", {}) or {})
    if bool(merged_flags.get("pnl_sanity", False)):
        w_sums = weights_daily.reindex(index=price_ix, columns=valid_cols).fillna(0.0).sum(axis=1)
        logger.info(
            "[PnL sanity] tickers=%d weights_sum[min=%.4f max=%.4f] ret[min=%.4f max=%.4f] tc_max=%.6f",
            len(valid_cols),
            float(w_sums.min()) if len(w_sums) else float("nan"),
            float(w_sums.max()) if len(w_sums) else float("nan"),
            float(returns_net_of_costs.min()) if len(returns_net_of_costs) else float("nan"),
            float(returns_net_of_costs.max()) if len(returns_net_of_costs) else float("nan"),
            (
                float(transaction_costs.max())
                if transaction_costs is not None and len(transaction_costs)
                else float("nan")
            ),
        )

    trade_tracker = None
    if track_trades:
        initial_portfolio_value = (
            global_config.get("portfolio_value", 100000.0)
            if isinstance(global_config, dict)
            else 100000.0
        )

        allocation_mode = (
            scenario_config.get("allocation_mode", "reinvestment")
            if isinstance(scenario_config, dict)
            else "reinvestment"
        )

        trade_tracker = TradeTracker(initial_portfolio_value, allocation_mode)

        common_index = price_ix
        valid_cols_tt = valid_cols
        close_for_tracker = close_prices_df.reindex(
            index=common_index, columns=valid_cols_tt
        ).astype(float)
        trade_tracker.populate_from_execution_ledger(
            sim_result.execution_ledger,
            sim_result.portfolio_values.reindex(common_index),
            sim_result.positions.reindex(index=common_index, columns=valid_cols_tt),
            close_for_tracker,
        )

    if include_signed_weights:
        return returns_net_of_costs, trade_tracker, signed_weights_out
    return returns_net_of_costs, trade_tracker


def _calculate_meta_strategy_portfolio_returns(
    strategy,
    scenario_config,
    price_data_daily_ohlc,
    rets_daily,
    universe_tickers,
    global_config,
    track_trades=False,
    *,
    include_signed_weights: bool = False,
):
    """Meta strategies use ``MetaExecutionMode.TRADE_AGGREGATION`` (trade ledger), not ``simulate_portfolio``."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Calculating meta strategy portfolio returns for {strategy.__class__.__name__}"
        )

    # Get the trade aggregator from the meta strategy
    trade_aggregator = strategy.get_trade_aggregator()

    # Get all trades from sub-strategies
    all_trades = trade_aggregator.get_aggregated_trades()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Meta strategy has {len(all_trades)} aggregated trades")

    if not all_trades:
        if logger.isEnabledFor(logging.WARNING):
            logger.warning("Meta strategy has no trades - returning zero returns")
        out_z = attach_portfolio_execution_model(
            pd.Series(0.0, index=price_data_daily_ohlc.index),
            MetaExecutionMode.TRADE_AGGREGATION,
        )
        return (out_z, None, None) if include_signed_weights else (out_z, None)

    # Extract market data for portfolio valuation
    if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
        # Extract close prices from MultiIndex columns
        market_data = price_data_daily_ohlc.xs("Close", level="Field", axis=1)
    else:
        # Single level columns
        market_data = price_data_daily_ohlc

    # Update portfolio values with market data
    trade_aggregator.update_portfolio_values_with_market_data(market_data)

    # Calculate returns based on actual trades and market movements
    portfolio_timeline = trade_aggregator.get_portfolio_timeline()

    if portfolio_timeline.empty:
        if logger.isEnabledFor(logging.WARNING):
            logger.warning("Meta strategy has no portfolio timeline - returning zero returns")
        out_z = attach_portfolio_execution_model(
            pd.Series(0.0, index=price_data_daily_ohlc.index),
            MetaExecutionMode.TRADE_AGGREGATION,
        )
        return (out_z, None, None) if include_signed_weights else (out_z, None)

    # Align returns with the price data index
    portfolio_returns = attach_portfolio_execution_model(
        portfolio_timeline["returns"]
        .reindex(price_data_daily_ohlc.index)
        .fillna(0.0)
        .astype(float),
        MetaExecutionMode.TRADE_AGGREGATION,
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Meta strategy returns calculated: {len(portfolio_returns)} days")
        logger.debug(
            f"Returns range: {portfolio_returns.min():.6f} to {portfolio_returns.max():.6f}"
        )
        tr_np = portfolio_returns.to_numpy(dtype=np.float64, copy=False)
        total_rt = float(np.prod(1.0 + tr_np) - 1.0)
        logger.debug("Total return: %.6f", total_rt)

    # Create trade tracker if requested
    trade_tracker = None
    if track_trades:
        trade_tracker = _create_meta_strategy_trade_tracker(
            strategy, global_config, scenario_config
        )

    signed_weights_out = None
    if include_signed_weights:
        signed_weights_out = trade_aggregator.get_signed_weights_dataframe(
            pd.DatetimeIndex(price_data_daily_ohlc.index)
        )
        return portfolio_returns, trade_tracker, signed_weights_out
    return portfolio_returns, trade_tracker


def _create_meta_strategy_trade_tracker(strategy, global_config, scenario_config=None):
    """
    Create a trade tracker for meta strategies using their aggregated trades.
    Behavior preserved; complexity reduced via small helpers.
    """

    def _allocation_mode(s_cfg):
        return s_cfg.get("allocation_mode", "reinvestment") if s_cfg else "reinvestment"

    def _compute_base_portfolio_value(tt: TradeTracker) -> float:
        # Select capital base according to allocation mode
        if tt.allocation_mode in ["reinvestment", "compound"]:
            return float(tt.get_current_portfolio_value())
        return float(tt.initial_portfolio_value)

    def _positions_to_weights(positions: dict, prices: dict, base_value: float) -> pd.Series:
        # Convert current positions to weights; ignore assets without prices
        assets = list(prices.keys())
        weights = pd.Series(0.0, index=assets)
        total_val = 0.0
        for asset, qty in positions.items():
            if abs(qty) > 1e-6 and asset in prices:
                total_val += qty * prices[asset]
        denom = total_val if total_val > 0 else base_value
        if denom <= 0:
            return weights
        for asset, qty in positions.items():
            if abs(qty) > 1e-6 and asset in prices:
                weights[asset] = (qty * prices[asset]) / denom
        return weights

    def _distribute_commissions(
        total_cost: float, traded_assets: list[str], current_positions: dict
    ) -> dict:
        if total_cost <= 0:
            return {}
        targets = (
            traded_assets[:]
            if traded_assets
            else [a for a, q in current_positions.items() if abs(q) > 1e-6]
        )
        if not targets:
            return {}
        per_asset = total_cost / len(targets)
        return {a: per_asset for a in targets}

    portfolio_value = global_config.get("portfolio_value", 100000.0)
    allocation_mode = _allocation_mode(scenario_config)
    trade_tracker = TradeTracker(portfolio_value, allocation_mode)

    # Collect and validate trades
    all_trades = strategy.get_aggregated_trades()
    if not all_trades:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("No trades to populate in trade tracker")
        return trade_tracker

    sorted_trades = sorted(all_trades, key=lambda t: t.date)
    current_positions: dict[str, float] = {}

    all_dates = sorted({t.date for t in sorted_trades})
    all_assets = {t.asset for t in sorted_trades}

    # Use unified model for commissions within meta-tracking
    tx_cost_model = get_unified_commission_calculator(global_config)

    for date in all_dates:
        date_trades = [t for t in sorted_trades if t.date == date]

        # Build prices and position change map
        prices: dict[str, float] = {}
        position_changes: dict[str, float] = {}
        for tr in date_trades:
            prices[tr.asset] = tr.price
            position_changes[tr.asset] = position_changes.get(tr.asset, 0.0) + abs(tr.quantity)
            current_positions.setdefault(tr.asset, 0.0)
            if tr.side.value == "buy":
                current_positions[tr.asset] += tr.quantity
            else:
                current_positions[tr.asset] -= tr.quantity

        # Commission base
        base_portfolio_value = _compute_base_portfolio_value(trade_tracker)

        # Compute weights for tx cost calc using total portfolio value
        weights = _positions_to_weights(current_positions, prices, base_portfolio_value)

        # Turnover proxy (sum of absolute changes) normalized by portfolio value
        total_turnover = sum(position_changes.values()) if position_changes else 0.0

        commissions: dict[str, float] = {}
        if total_turnover > 1e-8:
            turnover_series = pd.Series(
                [float(total_turnover) / max(float(base_portfolio_value), 1e-12)],
                index=[date],
                dtype=float,
            )
            weights_df = pd.DataFrame([weights], index=[date])

            # Construct minimal price frame
            price_row = {a: prices.get(a, 0.0) for a in all_assets}
            dummy_price_data = pd.DataFrame([price_row], index=[date])

            tx_series, _, _ = tx_cost_model.calculate(
                turnover=turnover_series,
                weights_daily=weights_df,
                price_data=dummy_price_data,
                portfolio_value=float(base_portfolio_value),
            )
            total_cost = (
                float(tx_series.iloc[0])
                if hasattr(tx_series, "iloc") and len(tx_series)
                else float(tx_series)
            )
            traded_assets = [t.asset for t in date_trades]
            commissions = _distribute_commissions(total_cost, traded_assets, current_positions)

        # Build price series limited to assets with prices
        price_assets = [a for a in all_assets if a in prices]
        price_series = pd.Series({a: prices[a] for a in price_assets}, dtype=float).dropna()

        if not price_series.empty:
            weights = weights.reindex(price_series.index).fillna(0.0)
            # Normalize zero commissions to a tiny epsilon if all zero (to satisfy downstream expectations)
            if commissions and all(v == 0 for v in commissions.values()):
                commissions = {k: 0.0001 for k in price_series.index}
            commission_series = pd.Series(0.0, index=price_series.index)
            for a, c in commissions.items():
                if a in commission_series.index:
                    commission_series[a] = c

            trade_tracker.update_positions(date, weights, price_series, commission_series.to_dict())
            trade_tracker.update_mfe_mae(date, price_series)

    # Final liquidation
    if all_dates:
        final_date = max(all_dates)
        # Use last known non-empty price_series if available
        # Reconstruct minimal price series using last seen prices for assets traded
        last_prices = {t.asset: t.price for t in sorted_trades if t.date == final_date}
        price_series = pd.Series(last_prices, dtype=float).dropna()
        if not price_series.empty:
            final_weights = pd.Series(0.0, index=price_series.index)
            final_commissions = {asset: 0.0 for asset in price_series.index}
            trade_tracker.update_positions(
                final_date, final_weights, price_series, final_commissions
            )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Created trade tracker with {len(all_trades)} meta strategy trades across {len(all_dates)} dates"
        )
        trade_stats = trade_tracker.get_trade_statistics()
        logger.debug(f"Framework trade tracker shows {trade_stats.get('all_num_trades', 0)} trades")

    return trade_tracker
