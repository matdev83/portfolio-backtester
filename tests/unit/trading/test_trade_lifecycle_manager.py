"""Tests for trade lifecycle Trade.finalize duration logic."""

from unittest.mock import patch

import pandas as pd
import pytest

from portfolio_backtester.trading.trade_lifecycle_manager import Trade


def test_finalize_business_day_duration_happy_path() -> None:
    entry = pd.Timestamp("2024-01-02")
    exit_ = pd.Timestamp("2024-01-05")
    trade = Trade(
        ticker="SPY",
        entry_date=entry,
        entry_price=100.0,
        quantity=10.0,
        entry_value=1000.0,
        commission_entry=0.0,
        exit_date=exit_,
        exit_price=101.0,
        commission_exit=0.0,
    )
    trade.finalize()
    assert trade.duration_days is not None
    assert trade.duration_days >= 0
    assert trade.pnl_net is not None


def test_finalize_calendar_fallback_on_value_error() -> None:
    entry = pd.Timestamp("2024-01-02")
    exit_ = pd.Timestamp("2024-01-10")
    trade = Trade(
        ticker="SPY",
        entry_date=entry,
        entry_price=100.0,
        quantity=10.0,
        entry_value=1000.0,
        commission_entry=0.0,
        exit_date=exit_,
        exit_price=101.0,
        commission_exit=0.0,
    )
    with patch("portfolio_backtester.trading.trade_lifecycle_manager.pd.bdate_range") as mock_br:
        mock_br.side_effect = ValueError("invalid range")
        trade.finalize()
    expected_calendar = (exit_ - entry).days
    assert trade.duration_days == expected_calendar


def test_finalize_bdate_range_runtime_error_propagates() -> None:
    entry = pd.Timestamp("2024-01-02")
    exit_ = pd.Timestamp("2024-01-10")
    trade = Trade(
        ticker="SPY",
        entry_date=entry,
        entry_price=100.0,
        quantity=10.0,
        entry_value=1000.0,
        commission_entry=0.0,
        exit_date=exit_,
        exit_price=101.0,
        commission_exit=0.0,
    )
    with patch("portfolio_backtester.trading.trade_lifecycle_manager.pd.bdate_range") as mock_br:
        mock_br.side_effect = RuntimeError("unexpected")
        with pytest.raises(RuntimeError, match="unexpected"):
            trade.finalize()
