"""Tests for trade-marker row materialization in ``plot_generator``."""

from __future__ import annotations

import pandas as pd

from portfolio_backtester.reporting.plot_generator import _trade_marker_frame


def test_trade_marker_frame_prefers_quantity_over_position() -> None:
    d0 = pd.Timestamp("2020-01-01")
    d1 = pd.Timestamp("2020-02-01")
    df = pd.DataFrame(
        {
            "quantity": [1.0],
            "position": [99.0],
            "entry_date": [d0],
            "exit_date": [d1],
            "entry_price": [10.0],
            "exit_price": [11.0],
        }
    )
    out = _trade_marker_frame(df)
    assert float(out.iloc[0]["qty"]) == 1.0


def test_trade_marker_frame_uses_position_when_no_quantity() -> None:
    d0 = pd.Timestamp("2020-01-01")
    d1 = pd.Timestamp("2020-02-01")
    df = pd.DataFrame(
        {
            "position": [-2.0],
            "entry_date": [d0],
            "exit_date": [d1],
            "entry_price": [10.0],
            "exit_price": [11.0],
        }
    )
    out = _trade_marker_frame(df)
    assert float(out.iloc[0]["qty"]) == -2.0


def test_entry_date_nat_does_not_fall_back_to_date_column() -> None:
    """Match Python ``entry or date`` when ``NaT`` is truthy (pandas 2.x)."""
    d1 = pd.Timestamp("2020-01-01")
    df = pd.DataFrame(
        {
            "quantity": [1.0],
            "entry_date": [pd.NaT],
            "date": [d1],
            "exit_date": [pd.NaT],
            "entry_price": [1.0],
            "exit_price": [2.0],
        }
    )
    out = _trade_marker_frame(df)
    assert pd.isna(out.iloc[0]["entry_date"])


def test_entry_date_none_falls_back_to_date() -> None:
    d1 = pd.Timestamp("2020-01-01")
    df = pd.DataFrame(
        {
            "quantity": [1.0],
            "date": [d1],
            "exit_date": [pd.NaT],
            "entry_price": [1.0],
            "exit_price": [2.0],
        }
    )
    df["entry_date"] = pd.Series([None], dtype=object)
    out = _trade_marker_frame(df)
    assert out.iloc[0]["entry_date"] == d1


def test_trade_marker_frame_empty() -> None:
    df = pd.DataFrame()
    out = _trade_marker_frame(df)
    assert out.empty
