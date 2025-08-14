import logging
import pandas as pd
import numpy as np

from ..interfaces.strategy_resolver import StrategyResolverFactory
from ..portfolio.rebalancing import rebalance
from ..trading.trade_tracker import TradeTracker
from ..trading.unified_commission_calculator import get_unified_commission_calculator
from ..numba_kernels import (
    position_and_pnl_kernel,
    detailed_commission_slippage_kernel,
    trade_tracking_kernel,
    trade_lifecycle_kernel,
)


logger = logging.getLogger(__name__)


def calculate_portfolio_returns(
    sized_signals,
    scenario_config,
    price_data_daily_ohlc,
    rets_daily,
    universe_tickers,
    global_config,
    track_trades=False,
    strategy=None,
):
    # Check if this is a meta strategy - if so, use trade-based returns
    strategy_resolver = StrategyResolverFactory.create()
    if strategy is not None and strategy_resolver.is_meta_strategy(type(strategy)):
        return _calculate_meta_strategy_portfolio_returns(
            strategy,
            scenario_config,
            price_data_daily_ohlc,
            rets_daily,
            universe_tickers,
            global_config,
            track_trades,
        )

    logger.debug("sized_signals shape: %s", sized_signals.shape)
    logger.debug("sized_signals head:\n%s", sized_signals.head())

    # Standard portfolio return calculation
    rebalance_frequency = scenario_config.get("timing_config", {}).get("rebalance_frequency", "M")
    weights_monthly = rebalance(sized_signals, rebalance_frequency)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("weights_monthly shape: %s", weights_monthly.shape)
        logger.debug("weights_monthly head:\n%s", weights_monthly.head())

    weights_monthly = weights_monthly.reindex(columns=universe_tickers).fillna(0.0)

    weights_daily = weights_monthly.reindex(price_data_daily_ohlc.index, method="ffill")
    # Use previous-day weights for returns so first day's gross return is 0
    weights_for_returns = weights_daily.shift(1).fillna(0.0)

    if rets_daily is None:
        logger.error("rets_daily is None before reindexing in run_scenario.")
        return pd.Series(0.0, index=price_data_daily_ohlc.index)

    aligned_rets_daily = rets_daily.reindex(price_data_daily_ohlc.index).fillna(0.0)

    # Fast path: ndarray/Numba kernel if enabled
    feature_flags = (
        (global_config or {}).get("feature_flags", {}) if isinstance(global_config, dict) else {}
    )
    # Enable ndarray/Numba fast path by default; allow users to disable via config
    use_ndarray_sim = bool(feature_flags.get("ndarray_simulation", True))

    # Ensure variable is defined for all branches
    transaction_costs: pd.Series | None = None

    if use_ndarray_sim:
        import numpy as np

        # The ndarray simulation is now the only path. If it fails, the system will raise an exception.
        if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and (
            "Close" in price_data_daily_ohlc.columns.get_level_values(-1)
        ):
            close_prices_df = price_data_daily_ohlc.xs("Close", level="Field", axis=1)
            price_index = close_prices_df.index
        else:
            close_prices_df = price_data_daily_ohlc
            price_index = price_data_daily_ohlc.index

        valid_cols = [t for t in universe_tickers if t in aligned_rets_daily.columns]
        r = (
            aligned_rets_daily.reindex(index=price_index, columns=valid_cols)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        m = ~np.isnan(r)
        w = (
            weights_for_returns.reindex(index=price_index, columns=valid_cols)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        # Use single optimized kernel implementation
        _, _, _ = position_and_pnl_kernel(w, r, m)

        # Compute detailed IBKR-style commission+slippage in ndarray path
        # Build close price matrix aligned to valid_cols and price_index
        close_prices_use = close_prices_df.reindex(index=price_index, columns=valid_cols).astype(
            float
        )
        price_mask = close_prices_use.notna() & (close_prices_use > 0)
        close_arr = close_prices_use.fillna(0.0).to_numpy(copy=True)
        price_mask_arr = price_mask.to_numpy(copy=True)

        # Read commission params from global_config
        commission_per_share = (
            float(global_config.get("commission_per_share", 0.005))
            if isinstance(global_config, dict)
            else 0.005
        )
        commission_min_per_order = (
            float(global_config.get("commission_min_per_order", 1.0))
            if isinstance(global_config, dict)
            else 1.0
        )
        commission_max_percent = (
            float(global_config.get("commission_max_percent_of_trade", 0.005))
            if isinstance(global_config, dict)
            else 0.005
        )
        slippage_bps = (
            float(global_config.get("slippage_bps", 2.5))
            if isinstance(global_config, dict)
            else 2.5
        )
        portfolio_value = (
            float(global_config.get("portfolio_value", 100000.0))
            if isinstance(global_config, dict)
            else 100000.0
        )

        # Use current weights (not shifted) for turnover/commissions
        weights_current = (
            weights_daily.reindex(index=price_index, columns=valid_cols).fillna(0.0).to_numpy()
        )

        # Use single optimized kernel implementation
        tc_frac, tc_frac_detailed = detailed_commission_slippage_kernel(
            weights_current=weights_current,
            close_prices=close_arr,
            portfolio_value=portfolio_value,
            commission_per_share=commission_per_share,
            commission_min_per_order=commission_min_per_order,
            commission_max_percent=commission_max_percent,
            slippage_bps=slippage_bps,
            price_mask=price_mask_arr,
        )

        # transaction_costs: per-day total fraction of portfolio value
        # per_asset_transaction_costs: per-day, per-asset fraction matrix
        transaction_costs = pd.Series(tc_frac, index=price_index, dtype=float)
        per_asset_transaction_costs = pd.DataFrame(
            tc_frac_detailed, index=price_index, columns=valid_cols
        )

        gross_returns_series = (weights_for_returns * aligned_rets_daily).sum(axis=1)

        returns_net_of_costs = pd.Series(
            gross_returns_series - transaction_costs, index=price_index
        )

    else:
        # The fallback path has been intentionally removed.
        # If use_ndarray_sim is False, this will now raise an error.
        raise RuntimeError(
            "The legacy Pandas-based portfolio simulation has been removed. "
            "Please enable 'ndarray_simulation' in the feature flags."
        )

    # Initialize trade tracker if requested
    trade_tracker = None
    if track_trades:
        initial_portfolio_value = global_config.get("portfolio_value", 100000.0)

        # Get allocation mode from scenario config (strategy-level setting)
        allocation_mode = scenario_config.get("allocation_mode", "reinvestment")

        trade_tracker = TradeTracker(initial_portfolio_value, allocation_mode)

        # PREPARE DATA FOR TRADE TRACKING (NUMPY FAST PATH)
        # Align all dataframes to a common index and columns
        valid_cols = list(weights_daily.columns.intersection(close_prices_df.columns))
        common_index = weights_daily.index.intersection(close_prices_df.index)

        weights_arr = (
            weights_daily.reindex(index=common_index, columns=valid_cols).fillna(0.0).to_numpy()
        )
        prices_arr = (
            close_prices_df.reindex(index=common_index, columns=valid_cols).fillna(0.0).to_numpy()
        )
        commissions_arr = (
            per_asset_transaction_costs.reindex(index=common_index, columns=valid_cols)
            .fillna(0.0)
            .to_numpy()
        )

        _track_trades_and_populate(
            trade_tracker=trade_tracker,
            weights_arr=weights_arr,
            prices_arr=prices_arr,
            commissions_arr=commissions_arr,
            dates=common_index.to_numpy(),
            tickers=np.array(valid_cols),
        )

        # If trade tracking was used, the portfolio value series is the source of truth for returns
        pv_series = trade_tracker.portfolio_value_tracker.daily_portfolio_value
        if isinstance(pv_series, pd.Series) and not pv_series.empty:
            returns_net_of_costs = pv_series.pct_change(fill_method=None).fillna(0.0).astype(float)

    return returns_net_of_costs, trade_tracker


def _calculate_meta_strategy_portfolio_returns(
    strategy,
    scenario_config,
    price_data_daily_ohlc,
    rets_daily,
    universe_tickers,
    global_config,
    track_trades=False,
):
    """
    Calculate portfolio returns for meta strategies using their aggregated trade history.

    Meta strategies track actual trades from sub-strategies, so we use their
    trade aggregator to calculate returns instead of the standard signal-based approach.
    """
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
        return pd.Series(0.0, index=price_data_daily_ohlc.index), None

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
        return pd.Series(0.0, index=price_data_daily_ohlc.index), None

    # Align returns with the price data index
    portfolio_returns = (
        portfolio_timeline["returns"].reindex(price_data_daily_ohlc.index).fillna(0.0)
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Meta strategy returns calculated: {len(portfolio_returns)} days")
        logger.debug(
            f"Returns range: {portfolio_returns.min():.6f} to {portfolio_returns.max():.6f}"
        )
        logger.debug(f"Total return: {(1 + portfolio_returns).prod() - 1:.6f}")

    # Create trade tracker if requested
    trade_tracker = None
    if track_trades:
        trade_tracker = _create_meta_strategy_trade_tracker(
            strategy, global_config, scenario_config
        )

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


def _track_trades_and_populate(
    trade_tracker: TradeTracker,
    weights_arr: np.ndarray,
    prices_arr: np.ndarray,
    commissions_arr: np.ndarray,
    dates: np.ndarray,
    tickers: np.ndarray,
):
    """
    Orchestrates the trade tracking process by calling the high-performance
    Numba kernel and then populating the TradeTracker object with the results.
    Args:
        trade_tracker: The TradeTracker object to populate.
        weights_arr: NumPy array of asset weights.
        prices_arr: NumPy array of asset prices.
        commissions_arr: NumPy array of commissions.
        dates: NumPy array of dates for the backtest period.
        tickers: NumPy array of ticker symbols.
    """
    price_mask = prices_arr > 0
    allocation_mode_map = {"reinvestment": 0, "fixed": 1}
    allocation_mode = allocation_mode_map.get(trade_tracker.allocation_mode, 0)

    # Call the Numba kernel to get portfolio values and positions
    portfolio_values, _, positions = trade_tracking_kernel(
        initial_portfolio_value=trade_tracker.initial_portfolio_value,
        allocation_mode=allocation_mode,
        weights=weights_arr,
        prices=prices_arr,
        price_mask=price_mask,
        commissions=commissions_arr,
    )

    # Call the new kernel to get completed trade data
    completed_trades_arr = trade_lifecycle_kernel(
        positions=positions,
        prices=prices_arr,
        dates=dates,
        commissions=commissions_arr,
        initial_capital=trade_tracker.initial_portfolio_value,
    )

    # Convert kernel output to DataFrames for the TradeTracker
    portfolio_values_series = pd.Series(portfolio_values, index=pd.to_datetime(dates))
    positions_df = pd.DataFrame(positions, index=pd.to_datetime(dates), columns=tickers)
    prices_df = pd.DataFrame(prices_arr, index=pd.to_datetime(dates), columns=tickers)

    # Populate the trade tracker with the results from the kernels
    trade_tracker.populate_from_kernel_results(
        portfolio_values=portfolio_values_series,
        positions=positions_df,
        completed_trades=completed_trades_arr,
        tickers=tickers,
        prices=prices_df,
    )

    logger.info(
        "[TradeTracking] completed_trades=%d",
        len(trade_tracker.trade_lifecycle_manager.get_completed_trades()),
    )
