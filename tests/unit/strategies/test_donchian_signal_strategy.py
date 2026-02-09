from __future__ import annotations

import pandas as pd

from portfolio_backtester.strategies.builtins.signal.donchian_asri_signal_strategy import (
    donchian_position_close_breakout,
    donchian_position_with_risk_filter,
)


def test_donchian_position_close_breakout_enters_and_exits() -> None:
    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    close = pd.Series([100.0] * 20 + [105.0] * 10 + [95.0] * 10, index=idx)

    pos = donchian_position_close_breakout(close, entry_lookback=20, exit_lookback=10)
    assert pos.name == "donchian_pos"
    assert float(pos.iloc[0]) == 0.0
    assert float(pos.loc["2026-01-22"]) == 1.0
    assert float(pos.iloc[-1]) == 0.0


def test_donchian_position_with_risk_filter_blocks_entry_and_forces_exit() -> None:
    idx = pd.date_range("2026-01-01", periods=35, freq="D")
    close = pd.Series([100.0] * 20 + [105.0] * 15, index=idx)
    risk_on = pd.Series(True, index=idx)

    risk_on.loc["2026-01-21":"2026-01-25"] = False
    pos = donchian_position_with_risk_filter(
        close, risk_on=risk_on, entry_lookback=20, exit_lookback=10
    )
    assert pos.max() == 0.0

    risk_on2 = pd.Series(True, index=idx)
    pos2 = donchian_position_with_risk_filter(
        close, risk_on=risk_on2, entry_lookback=20, exit_lookback=10
    )
    assert pos2.max() == 1.0
    risk_on2.loc["2026-01-28":"2026-01-30"] = False
    pos3 = donchian_position_with_risk_filter(
        close, risk_on=risk_on2, entry_lookback=20, exit_lookback=10
    )
    assert float(pos3.loc["2026-01-29"]) == 0.0
