"""Canonical trade analytics built from the execution ledger."""

from __future__ import annotations

import pandas as pd

from .trade_lifecycle_manager import Trade
from .trade_tracker import TradeTracker


def _completed_trade_row(trade: Trade) -> dict[str, object]:
    side = "long" if trade.quantity > 0 else "short"
    return {
        "ticker": trade.ticker,
        "side": side,
        "entry_date": trade.entry_date,
        "exit_date": trade.exit_date,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "quantity": trade.quantity,
        "entry_value": trade.entry_value,
        "entry_cost": trade.commission_entry,
        "exit_cost": trade.commission_exit,
        "duration_days": trade.duration_days,
        "pnl_gross": float(trade.pnl_gross) if trade.pnl_gross is not None else float("nan"),
        "pnl_net": float(trade.pnl_net) if trade.pnl_net is not None else float("nan"),
        "mfe": trade.mfe,
        "mae": trade.mae,
    }


def _open_trade_row(trade: Trade) -> dict[str, object]:
    side = "long" if trade.quantity > 0 else "short"
    return {
        "ticker": trade.ticker,
        "side": side,
        "entry_date": trade.entry_date,
        "entry_price": trade.entry_price,
        "quantity": trade.quantity,
        "entry_value": trade.entry_value,
        "entry_cost": trade.commission_entry,
        "current_quantity": trade.quantity,
    }


_COMPLETED_COLUMNS: tuple[str, ...] = (
    "ticker",
    "side",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "quantity",
    "entry_value",
    "entry_cost",
    "exit_cost",
    "duration_days",
    "pnl_gross",
    "pnl_net",
    "mfe",
    "mae",
)

_OPEN_COLUMNS: tuple[str, ...] = (
    "ticker",
    "side",
    "entry_date",
    "entry_price",
    "quantity",
    "entry_value",
    "entry_cost",
    "current_quantity",
)

_SUMMARY_COLUMNS: tuple[str, ...] = (
    "num_completed",
    "num_open",
    "gross_pnl",
    "net_pnl",
    "long_count",
    "short_count",
)


def build_canonical_trade_analytics(
    execution_ledger: pd.DataFrame,
    portfolio_values: pd.Series,
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    initial_portfolio_value: float = 100000.0,
    allocation_mode: str = "reinvestment",
) -> dict[str, pd.DataFrame]:
    """Replays ``execution_ledger`` via :class:`TradeTracker` and returns tabular analytics.

    Args:
        execution_ledger: Canonical simulator ledger rows.
        portfolio_values: Daily portfolio value series (kernel-aligned index).
        positions: Daily positions DataFrame (kernel-aligned).
        prices: Daily close(or execution reference) prices for MFE/MAE.
        initial_portfolio_value: Starting capital for the tracker facade.
        allocation_mode: Tracker allocation mode (passed through only).

    Returns:
        Mapping with keys ``completed_trades``, ``open_trades``, and ``summary``.
    """
    tracker = TradeTracker(
        initial_portfolio_value=initial_portfolio_value,
        allocation_mode=allocation_mode,
    )
    tracker.populate_from_execution_ledger(
        execution_ledger,
        portfolio_values,
        positions,
        prices,
    )
    mgr = tracker.trade_lifecycle_manager
    completed = mgr.get_completed_trades()
    open_map = mgr.get_open_positions()

    completed_rows = [_completed_trade_row(t) for t in completed]
    completed_df = (
        pd.DataFrame(completed_rows, columns=list(_COMPLETED_COLUMNS))
        if completed_rows
        else pd.DataFrame(columns=list(_COMPLETED_COLUMNS))
    )

    open_items = sorted(open_map.items(), key=lambda kv: kv[0])
    open_rows = [_open_trade_row(t) for _, t in open_items]
    open_df = (
        pd.DataFrame(open_rows, columns=list(_OPEN_COLUMNS))
        if open_rows
        else pd.DataFrame(columns=list(_OPEN_COLUMNS))
    )

    long_count = sum(1 for t in completed if t.quantity > 0)
    short_count = sum(1 for t in completed if t.quantity < 0)
    gross = sum(float(t.pnl_gross or 0.0) for t in completed)
    net = sum(float(t.pnl_net or 0.0) for t in completed)

    summary_df = pd.DataFrame(
        [
            {
                "num_completed": len(completed),
                "num_open": len(open_map),
                "gross_pnl": gross,
                "net_pnl": net,
                "long_count": long_count,
                "short_count": short_count,
            }
        ],
        columns=list(_SUMMARY_COLUMNS),
    )

    return {
        "completed_trades": completed_df,
        "open_trades": open_df,
        "summary": summary_df,
    }
