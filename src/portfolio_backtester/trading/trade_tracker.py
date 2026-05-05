"""
TradeTracker facade system.

This module unifies various trading components into a cohesive interface
for managing trade tracking, statistics, and portfolio evaluation.
"""

import logging
import pandas as pd
from typing import Dict, Optional, Any
import numpy as np

from .trade_lifecycle_manager import TradeLifecycleManager, Trade
from .trade_statistics_calculator import TradeStatisticsCalculator
from .portfolio_value_tracker import PortfolioValueTracker
from .trade_table_formatter import TradeTableFormatter
from ..interfaces.commission_parameter_handler_interface import (
    CommissionParameterHandlerFactory,
)

logger = logging.getLogger(__name__)


def _ledger_sign_nonzero(x: float, eps: float) -> int:
    if abs(x) <= eps:
        return 0
    return 1 if x > 0 else -1


def _populate_mfe_mae_before_close(
    trade: Trade, prices: pd.DataFrame, exit_date: pd.Timestamp
) -> None:
    """Set per-share MFE/MAE on ``trade`` from ``prices`` between entry and exit (inclusive)."""
    ticker = trade.ticker
    if ticker not in prices.columns:
        trade.mfe = 0.0
        trade.mae = 0.0
        return
    px = prices[ticker]
    entry = pd.Timestamp(trade.entry_date)
    exit_ts = pd.Timestamp(exit_date)
    try:
        window = px.loc[entry:exit_ts]
    except (TypeError, KeyError):
        mask = (px.index >= entry) & (px.index <= exit_ts)
        window = px.loc[mask]
    if window.empty:
        trade.mfe = 0.0
        trade.mae = 0.0
        return
    ep = float(trade.entry_price)
    q = float(trade.quantity)
    mfe_ps = 0.0
    mae_ps = 0.0
    for cur in window.to_numpy(dtype=float, copy=False):
        if np.isnan(cur):
            continue
        cur_f = float(cur)
        if q > 0.0:
            pnl_ps = cur_f - ep
        else:
            pnl_ps = ep - cur_f
        if pnl_ps > mfe_ps:
            mfe_ps = pnl_ps
        if pnl_ps < mae_ps:
            mae_ps = pnl_ps
    trade.mfe = float(mfe_ps)
    trade.mae = float(mae_ps)


def _split_flip_transaction_cost(
    cost: float,
    exec_price: float,
    abs_shares_closed: float,
    abs_shares_opened: float,
) -> tuple[float, float]:
    n_close = abs_shares_closed * exec_price
    n_open = abs_shares_opened * exec_price
    tot = n_close + n_open
    if tot <= 1e-18:
        return cost, 0.0
    return cost * (n_close / tot), cost * (n_open / tot)


def _partial_close_open_trade(
    mgr: TradeLifecycleManager,
    ticker: str,
    date: pd.Timestamp,
    exec_price: float,
    exit_commission: float,
    close_qty_abs: float,
    prices: pd.DataFrame,
) -> None:
    t = mgr.open_positions[ticker]
    q_total = abs(t.quantity)
    if q_total <= 0.0 or close_qty_abs <= 0.0:
        return
    inv = 1.0 if t.quantity > 0 else -1.0
    frac = close_qty_abs / q_total
    closed = Trade(
        ticker=ticker,
        entry_date=t.entry_date,
        entry_price=t.entry_price,
        quantity=inv * close_qty_abs,
        entry_value=t.entry_price * close_qty_abs,
        commission_entry=t.commission_entry * frac,
        exit_date=date,
        exit_price=exec_price,
        commission_exit=exit_commission,
        detailed_commission_entry=t.detailed_commission_entry,
        mfe=0.0,
        mae=0.0,
    )
    _populate_mfe_mae_before_close(closed, prices, date)
    closed.mfe = closed.mfe * abs(closed.quantity)
    closed.mae = closed.mae * abs(closed.quantity)
    closed.finalize()
    mgr.trades.append(closed)
    rem = q_total - close_qty_abs
    if rem <= 1e-9:
        del mgr.open_positions[ticker]
    else:
        t.quantity = inv * rem
        t.entry_value = abs(t.quantity) * t.entry_price
        t.commission_entry = t.commission_entry * (rem / q_total)


class TradeTracker:
    """
    Comprehensive trade tracking and analysis system.

    This facade coordinates the lifecycle management of trades,
    portfolio value tracking, and detailed statistics calculation
    for complete portfolio management.
    """

    def __init__(
        self,
        initial_portfolio_value: float = 100000.0,
        allocation_mode: str = "reinvestment",
    ) -> None:
        """
        Initialize the TradeTracker facade.

        Args:
            initial_portfolio_value: Starting capital amount
            allocation_mode: Capital allocation mode
        """
        self.trade_lifecycle_manager = TradeLifecycleManager()
        self.trade_statistics_calculator = TradeStatisticsCalculator()
        self.portfolio_value_tracker = PortfolioValueTracker(
            initial_portfolio_value, allocation_mode
        )
        self.table_formatter = TradeTableFormatter()

    @property
    def allocation_mode(self) -> str:
        """Get the allocation mode."""
        return self.portfolio_value_tracker.allocation_mode

    @property
    def initial_portfolio_value(self) -> float:
        """Get the initial portfolio value."""
        return self.portfolio_value_tracker.initial_portfolio_value

    @property
    def current_portfolio_value(self) -> float:
        """Get the current portfolio value."""
        return self.portfolio_value_tracker.current_portfolio_value

    def update_positions(
        self,
        date: pd.Timestamp,
        new_weights: pd.Series,
        prices: pd.Series,
        commissions,  # Union[Dict[str, float], float]
        detailed_commission_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update positions based on new target weights with dynamic capital tracking.

        Args:
            date: Current date
            new_weights: Target portfolio weights
            prices: Asset prices
            commissions: Commission costs per asset
            detailed_commission_info: Detailed commission information from unified calculator
        """
        # Handle commissions parameter using polymorphic interface
        commission_handler = CommissionParameterHandlerFactory.create_handler(commissions)
        commissions_dict = commission_handler.normalize_commissions(
            commissions, new_weights, self.trade_lifecycle_manager.get_open_positions()
        )

        # Calculate target quantities based on allocation mode
        base_capital = self.portfolio_value_tracker.get_base_capital_for_allocation()

        target_quantities = {}
        for ticker, weight in new_weights.items():
            price = prices.get(ticker)
            if price is not None and price > 0:
                target_value = weight * base_capital
                target_quantities[ticker] = target_value / price

        # Process position changes
        current_tickers = set(self.trade_lifecycle_manager.get_open_positions().keys())
        target_tickers = set(
            str(ticker) for ticker, qty in target_quantities.items() if abs(qty) > 1e-6
        )

        # Close positions not in target
        for ticker in current_tickers - target_tickers:
            if ticker in prices:
                commission = commissions_dict.get(ticker, 0.0)

                # Extract detailed commission info for this ticker if available
                ticker_commission_details = None
                if detailed_commission_info and ticker in detailed_commission_info:
                    ticker_commission_details = detailed_commission_info[ticker]

                closed_trade = self.trade_lifecycle_manager.close_position(
                    date, ticker, prices[ticker], commission, ticker_commission_details
                )
                if closed_trade:
                    self.portfolio_value_tracker.update_portfolio_value(closed_trade.pnl_net or 0.0)

        # Open new positions
        for ticker in target_tickers - current_tickers:
            if ticker in prices:
                commission = commissions_dict.get(str(ticker), 0.0)

                # Extract detailed commission info for this ticker if available
                ticker_commission_details = None
                if detailed_commission_info and ticker in detailed_commission_info:
                    ticker_commission_details = detailed_commission_info[ticker]

                self.trade_lifecycle_manager.open_position(
                    date,
                    str(ticker),
                    target_quantities[ticker],
                    prices[ticker],
                    commission,
                    ticker_commission_details,
                )

        # Update daily metrics
        self.portfolio_value_tracker.update_daily_metrics(
            date,
            target_quantities,
            prices,
            self.trade_lifecycle_manager.get_open_positions(),
        )

    def update_mfe_mae(self, date: pd.Timestamp, prices: pd.Series) -> None:
        """Update Maximum Favorable/Adverse Excursion for open positions."""
        self.trade_lifecycle_manager.update_mfe_mae(date, prices)

    def populate_from_kernel_results(
        self,
        portfolio_values: pd.Series,
        positions: pd.DataFrame,
        completed_trades: np.ndarray,
        tickers: np.ndarray,
        prices: pd.DataFrame,
        *,
        allow_legacy_close_trade_inference: bool = False,
    ) -> None:
        """Populate from ``trade_lifecycle_kernel`` structured trades (legacy / non-canonical).

        **Not** used by ``calculate_portfolio_returns`` for standard strategies, which must use
        :meth:`populate_from_execution_ledger` with the canonical simulator ledger. This path
        infers lifecycle from positions plus a single **close** price panel and remains for
        unit tests and the :mod:`portfolio_backtester.trading.numba_trade_tracker` adapter only.

        Args:
            portfolio_values: A Series of daily portfolio values.
            positions: A DataFrame of daily asset positions.
            completed_trades: A structured NumPy array of completed trades.
            tickers: An array of ticker symbols corresponding to the ticker_idx in the trades.
            prices: A DataFrame of daily asset prices.
            allow_legacy_close_trade_inference: Must be ``True`` for intentional calls; otherwise
                raises :class:`RuntimeError` so production code cannot silently use close-based
                trade reconstruction.
        """
        if not allow_legacy_close_trade_inference:
            raise RuntimeError(
                "TradeTracker.populate_from_kernel_results is legacy close-price position "
                "inference for tests and numba_trade_tracker only; pass "
                "allow_legacy_close_trade_inference=True if intentional. Production portfolio "
                "paths must use populate_from_execution_ledger with the canonical execution_ledger."
            )
        # Directly set the portfolio value timeline and positions
        self.portfolio_value_tracker.set_state_from_kernel(portfolio_values, positions, prices)

        # Convert the structured array of trades into Trade objects
        for raw_trade in completed_trades:
            pnl_gross = raw_trade["pnl"]
            commission = raw_trade["commission"]
            pnl_net = pnl_gross - commission

            trade = Trade(
                ticker=tickers[raw_trade["ticker_idx"]],
                entry_date=pd.to_datetime(raw_trade["entry_date"], unit="ns"),
                exit_date=pd.to_datetime(raw_trade["exit_date"], unit="ns"),
                entry_price=raw_trade["entry_price"],
                exit_price=raw_trade["exit_price"],
                quantity=raw_trade["quantity"],
                entry_value=raw_trade["entry_price"] * raw_trade["quantity"],
                pnl_net=pnl_net,
                pnl_gross=pnl_gross,
                commission_entry=0.0,
                commission_exit=commission,
                # MFE/MAE are not calculated in the kernel for performance reasons
                mfe=0.0,
                mae=0.0,
            )
            self.trade_lifecycle_manager.trades.append(trade)

        logger.info("[TradeTracker] Populated %d trades from kernel.", len(completed_trades))

    def populate_from_execution_ledger(
        self,
        execution_ledger: pd.DataFrame,
        portfolio_values: pd.Series,
        positions: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> None:
        """Build completed trades from canonical simulator ``execution_ledger`` rows.

        Rows are replayed in stable order: ``execution_date_idx``, ``ticker``, ``execution_price``,
        ``quantity`` (``mergesort``) so same-calendar multi-asset fills are deterministic.
        Increasing same-direction exposure averages ``entry_price`` and recomputes open-leg
        per-share MFE/MAE through the add execution date so excursion matches the new basis.
        """
        self.trade_lifecycle_manager.trades.clear()
        self.trade_lifecycle_manager.open_positions.clear()

        eps = 1e-9
        if execution_ledger is not None and not execution_ledger.empty:
            sort_cols = ["execution_date_idx", "ticker", "execution_price", "quantity"]
            missing = [c for c in sort_cols if c not in execution_ledger.columns]
            if missing:
                raise ValueError(
                    f"execution_ledger missing columns required for stable sort: {missing}"
                )
            ledger_sorted = execution_ledger.sort_values(sort_cols, kind="mergesort")
            mgr = self.trade_lifecycle_manager
            for _, row in ledger_sorted.iterrows():
                dqty = float(row["quantity"])
                if abs(dqty) <= eps:
                    continue
                date = pd.Timestamp(row["execution_date"])
                ticker = str(row["ticker"])
                exec_price = float(row["execution_price"])
                cost = float(row["cost"])
                pos_b = float(row["position_before"])
                pos_a = float(row["position_after"])
                sb = _ledger_sign_nonzero(pos_b, eps)
                sa = _ledger_sign_nonzero(pos_a, eps)

                if sb == 0 and sa != 0:
                    mgr.open_position(date, ticker, pos_a, exec_price, cost)
                    continue
                if sa == 0 and sb != 0:
                    open_trade = mgr.open_positions.get(ticker)
                    if open_trade is not None:
                        _populate_mfe_mae_before_close(open_trade, prices, date)
                    mgr.close_position(date, ticker, exec_price, cost)
                    continue

                if sb == sa:
                    if abs(pos_a) > abs(pos_b) + eps:
                        t = mgr.open_positions[ticker]
                        old_abs = abs(t.quantity)
                        new_abs = abs(pos_a)
                        add_abs = new_abs - old_abs
                        inv = 1.0 if t.quantity > 0 else -1.0
                        e_new = (t.entry_price * old_abs + exec_price * add_abs) / new_abs
                        t.entry_price = e_new
                        t.quantity = inv * new_abs
                        t.entry_value = abs(t.quantity) * t.entry_price
                        t.commission_entry += cost
                        _populate_mfe_mae_before_close(t, prices, date)
                    elif abs(pos_b) > abs(pos_a) + eps:
                        _partial_close_open_trade(
                            mgr,
                            ticker,
                            date,
                            exec_price,
                            cost,
                            abs(pos_b) - abs(pos_a),
                            prices,
                        )
                    continue

                c_exit, c_entry = _split_flip_transaction_cost(
                    cost,
                    exec_price,
                    abs(pos_b),
                    abs(pos_a),
                )
                flip_trade = mgr.open_positions.get(ticker)
                if flip_trade is not None:
                    _populate_mfe_mae_before_close(flip_trade, prices, date)
                mgr.close_position(date, ticker, exec_price, c_exit)
                mgr.open_position(date, ticker, pos_a, exec_price, c_entry)

        self.portfolio_value_tracker.set_state_from_kernel(portfolio_values, positions, prices)
        n_done = len(self.trade_lifecycle_manager.get_completed_trades())
        logger.info("[TradeTracker] Populated %d trades from execution ledger.", n_done)

    def get_trade_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive trade statistics."""
        completed_trades = self.trade_lifecycle_manager.get_completed_trades()
        portfolio_stats = self.portfolio_value_tracker.get_portfolio_level_stats()
        trade_stats = self.trade_statistics_calculator.calculate_statistics(
            completed_trades,
            self.portfolio_value_tracker.initial_portfolio_value,
            self.portfolio_value_tracker.allocation_mode,
        )
        trade_stats.update(portfolio_stats)
        return trade_stats

    def get_trade_statistics_table(self) -> pd.DataFrame:
        """Get trade statistics formatted as a table with All/Long/Short columns."""
        stats = self.get_trade_statistics()
        return self.table_formatter.format_statistics_table(stats)

    def get_directional_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of key metrics by direction for easy comparison."""
        stats = self.get_trade_statistics()
        return self.table_formatter.format_directional_summary(stats)

    def close_all_positions(
        self,
        date: pd.Timestamp,
        prices: pd.Series,
        commissions: Optional[Dict[str, float]] = None,
    ) -> None:
        """Close all open positions at the end of backtesting."""
        closed_trades = self.trade_lifecycle_manager.close_all_positions(date, prices, commissions)
        for trade in closed_trades:
            self.portfolio_value_tracker.update_portfolio_value(trade.pnl_net or 0.0)

    def get_current_portfolio_value(self) -> float:
        """Get the current portfolio value after all trades."""
        return self.portfolio_value_tracker.get_current_portfolio_value()

    def get_total_return(self) -> float:
        """Get the total return as a percentage."""
        return self.portfolio_value_tracker.get_total_return()

    def get_capital_timeline(self) -> pd.DataFrame:
        """Get a timeline of portfolio value, cash balance, and position values."""
        return self.portfolio_value_tracker.get_capital_timeline()


# Export Trade class
__all__ = ["TradeTracker", "Trade"]
