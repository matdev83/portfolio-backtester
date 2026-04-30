from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy import (
    MmmQsSwingNasdaqSignalStrategy,
    _compute_adx_series,
)


def _idx_calendar_days_end_on(end: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """``n`` calendar days ending on ``end`` (inclusive)."""
    return pd.date_range(end - pd.Timedelta(days=n - 1), end, freq="D")


def _ohlc_frame(
    idx: pd.DatetimeIndex,
    close: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
    open_: list[float] | None = None,
    ticker: str = "QQQ",
) -> pd.DataFrame:
    hi = list(high) if high is not None else [c * 1.01 for c in close]
    lo = list(low) if low is not None else [c * 0.99 for c in close]
    op = list(open_) if open_ is not None else list(close)
    columns = pd.MultiIndex.from_product(
        [[ticker], ["Open", "High", "Low", "Close"]],
        names=["Ticker", "Field"],
    )
    return pd.DataFrame(
        np.column_stack([op, hi, lo, close]),
        index=idx,
        columns=columns,
        dtype=float,
    )


def test_warmup_returns_zero_and_resets_state() -> None:
    idx = pd.date_range("2024-06-03", periods=15, freq="D")
    close = [100.0 + i * 0.1 for i in range(len(idx))]
    ohlc = _ohlc_frame(idx, close)
    strat = MmmQsSwingNasdaqSignalStrategy({"strategy_params": {}})
    bench = pd.DataFrame(index=idx)
    out = strat.generate_signals(ohlc, bench, None, current_date=idx[-1])
    assert float(out.iloc[0]["QQQ"]) == 0.0


def test_entry_when_adx_and_atr_patched() -> None:
    """Long when calendar OK, IBS low, close below prior HLC3 mean, ADX/ATR patched."""
    # End on a Monday so default day-of-week filters allow trading.
    idx = _idx_calendar_days_end_on(pd.Timestamp("2024-07-08"), 60)
    # Uptrend then dip on last day so close < mean(hlc3[1..10])
    close = [100.0 + i * 0.12 for i in range(len(idx) - 1)] + [100.0]
    low = list(close)
    high = [c + 2.0 for c in close]
    # Last bar: push IBS low (close near low of bar)
    low[-1] = close[-1] - 0.05
    high[-1] = close[-1] + 5.0
    ohlc = _ohlc_frame(idx, close, high=high, low=low)
    adx_const = pd.Series(35.0, index=idx)
    strat = MmmQsSwingNasdaqSignalStrategy({"strategy_params": {}})
    bench = pd.DataFrame(index=idx)
    with (
        patch(
            "portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy._compute_adx_series",
            return_value=adx_const,
        ),
        patch(
            "portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy.calculate_atr_fast",
            return_value=pd.Series({"QQQ": 2.0}),
        ),
    ):
        out = strat.generate_signals(ohlc, bench, None, current_date=idx[-1])
    assert float(out.at[idx[-1], "QQQ"]) == pytest.approx(1.0)


def test_february_disabled() -> None:
    # Entire window in Jan–Feb so the evaluation date is in February (month off).
    idx = pd.date_range("2024-01-02", "2024-02-27", freq="D")
    close = [100.0 + i * 0.05 for i in range(len(idx) - 1)] + [100.0]
    low = list(close)
    high = [c + 2.0 for c in close]
    low[-1] = close[-1] - 0.05
    high[-1] = close[-1] + 5.0
    ohlc = _ohlc_frame(idx, close, high=high, low=low)
    strat = MmmQsSwingNasdaqSignalStrategy({"strategy_params": {}})
    bench = pd.DataFrame(index=idx)
    with (
        patch(
            "portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy._compute_adx_series",
            return_value=pd.Series(35.0, index=idx),
        ),
        patch(
            "portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy.calculate_atr_fast",
            return_value=pd.Series({"QQQ": 2.0}),
        ),
    ):
        out = strat.generate_signals(ohlc, bench, None, current_date=idx[-1])
    assert float(out.at[idx[-1], "QQQ"]) == 0.0


def test_wednesday_disabled_default() -> None:
    idx = pd.date_range("2024-06-03", periods=50, freq="D")
    # 2024-06-05 is Wednesday
    wed = pd.Timestamp("2024-06-05")
    assert int(wed.weekday()) == 2
    close = [100.0 + i * 0.05 for i in range(len(idx))]
    ohlc = _ohlc_frame(idx, close)
    strat = MmmQsSwingNasdaqSignalStrategy({"strategy_params": {}})
    bench = pd.DataFrame(index=idx)
    with (
        patch(
            "portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy._compute_adx_series",
            return_value=pd.Series(35.0, index=idx),
        ),
        patch(
            "portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy.calculate_atr_fast",
            return_value=pd.Series({"QQQ": 2.0}),
        ),
    ):
        out = strat.generate_signals(ohlc, bench, None, current_date=wed)
    assert float(out.at[wed, "QQQ"]) == 0.0


def test_close_above_prior_high_exits_same_instance() -> None:
    """After a synthetic long, ``close > high[1]`` forces flat on the next bar."""
    idx = pd.date_range("2024-05-20", "2024-07-10", freq="D")
    close = [100.0] * len(idx)
    high = [c + 1.0 for c in close]
    low = [c - 1.0 for c in close]
    i_exit = int(idx.get_indexer([pd.Timestamp("2024-07-08")])[0])
    high[i_exit - 1] = 110.0
    close[i_exit] = 115.0
    high[i_exit] = 116.0
    low[i_exit] = 114.0
    ohlc = _ohlc_frame(idx, close, high=high, low=low)
    strat = MmmQsSwingNasdaqSignalStrategy({"strategy_params": {}})
    bench = pd.DataFrame(index=idx)
    strat._bracket.in_long = True
    strat._bracket.stop_price = 1.0
    strat._bracket.take_profit_price = 1000.0
    exit_ts = pd.Timestamp(idx[i_exit])
    out = strat.generate_signals(ohlc, bench, None, current_date=exit_ts)
    assert float(out.at[exit_ts, "QQQ"]) == 0.0
    assert strat._bracket.in_long is False


def test_stop_loss_hit_before_take_profit() -> None:
    d_enter = pd.Timestamp("2024-07-02")
    d_stop = pd.Timestamp("2024-07-08")
    idx = pd.date_range("2024-05-20", d_stop, freq="D")
    close = [100.0 + i * 0.05 for i in range(len(idx) - 1)] + [100.0]
    low = list(close)
    high = [c + 2.0 for c in close]
    low[-2] = close[-2] - 0.05
    high[-2] = close[-2] + 5.0
    ohlc = _ohlc_frame(idx, close, high=high, low=low)
    strat = MmmQsSwingNasdaqSignalStrategy(
        {"strategy_params": {"sl_atr_mult": 1.0, "tp_atr_mult": 5.0}}
    )
    bench = pd.DataFrame(index=idx)
    d_enter = idx[-2]
    d_stop = idx[-1]
    adx_calls: dict[str, int] = {"n": 0}

    def _adx_side_effect(
        high: pd.Series, low: pd.Series, close: pd.Series, di_len: int, adx_s: int
    ) -> pd.Series:
        adx_calls["n"] += 1
        if adx_calls["n"] >= 2:
            return pd.Series(5.0, index=close.index)
        return pd.Series(35.0, index=close.index)

    with (
        patch(
            "portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy._compute_adx_series",
            side_effect=_adx_side_effect,
        ),
        patch(
            "portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy.calculate_atr_fast",
            return_value=pd.Series({"QQQ": 2.0}),
        ),
    ):
        strat.generate_signals(ohlc, bench, None, current_date=d_enter)
        entry_c = float(ohlc.at[d_enter, ("QQQ", "Close")])
        stop_level = entry_c - 2.0 * 1.0
        close2: list[float] = ohlc[("QQQ", "Close")].astype(float).tolist()
        low2: list[float] = ohlc[("QQQ", "Low")].astype(float).tolist()
        high2: list[float] = ohlc[("QQQ", "High")].astype(float).tolist()
        close2[-1] = stop_level + 5.0
        low2[-1] = stop_level - 0.5
        high2[-1] = stop_level + 8.0
        ohlc2 = _ohlc_frame(idx, close2, high=high2, low=low2)
        o2 = strat.generate_signals(ohlc2, bench, None, current_date=d_stop)
    assert float(o2.at[d_stop, "QQQ"]) == 0.0


def test_take_profit_hit() -> None:
    d_enter = pd.Timestamp("2024-07-02")
    d_tp = pd.Timestamp("2024-07-08")
    idx = pd.date_range("2024-05-20", d_tp, freq="D")
    close = [100.0 + i * 0.05 for i in range(len(idx) - 1)] + [100.0]
    low = list(close)
    high = [c + 2.0 for c in close]
    low[-2] = close[-2] - 0.05
    high[-2] = close[-2] + 5.0
    ohlc = _ohlc_frame(idx, close, high=high, low=low)
    strat = MmmQsSwingNasdaqSignalStrategy(
        {"strategy_params": {"sl_atr_mult": 5.0, "tp_atr_mult": 1.0}}
    )
    bench = pd.DataFrame(index=idx)
    d_enter = idx[-2]
    d_tp = idx[-1]
    adx_calls: dict[str, int] = {"n": 0}

    def _adx_side_effect(
        high: pd.Series, low: pd.Series, close: pd.Series, di_len: int, adx_s: int
    ) -> pd.Series:
        adx_calls["n"] += 1
        if adx_calls["n"] >= 2:
            return pd.Series(5.0, index=close.index)
        return pd.Series(35.0, index=close.index)

    with (
        patch(
            "portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy._compute_adx_series",
            side_effect=_adx_side_effect,
        ),
        patch(
            "portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy.calculate_atr_fast",
            return_value=pd.Series({"QQQ": 2.0}),
        ),
    ):
        strat.generate_signals(ohlc, bench, None, current_date=d_enter)
        entry_c = float(ohlc.at[d_enter, ("QQQ", "Close")])
        tp_level = entry_c + 2.0 * 1.0
        close2: list[float] = ohlc[("QQQ", "Close")].astype(float).tolist()
        low2: list[float] = ohlc[("QQQ", "Low")].astype(float).tolist()
        high2: list[float] = ohlc[("QQQ", "High")].astype(float).tolist()
        high2[-1] = tp_level + 0.5
        close2[-1] = tp_level + 0.1
        low2[-1] = tp_level - 0.5
        ohlc2 = _ohlc_frame(idx, close2, high=high2, low=low2)
        o2 = strat.generate_signals(ohlc2, bench, None, current_date=d_tp)
    assert float(o2.at[d_tp, "QQQ"]) == 0.0


def test_compute_adx_series_finite_after_warmup() -> None:
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = pd.Series(range(100), index=idx, dtype=float)
    high = 100 + rng
    low = 99 + rng
    close = 99.5 + rng
    adx = _compute_adx_series(high, low, close, di_length=7, adx_smoothing=14)
    assert pd.notna(adx.iloc[-1])
