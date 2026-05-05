from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.portfolio_simulation_input import (
    EXECUTION_TIMING_BAR_CLOSE,
    EXECUTION_TIMING_NEXT_BAR_OPEN,
    build_portfolio_simulation_input,
    extract_field_frame_from_ohlc,
    extract_open_frame_from_ohlc,
    prepare_close_arrays_for_simulation,
    prepare_open_arrays_for_simulation,
    propagate_rebalance_mask_for_invalid_next_bar_opens,
    propagate_rebalance_mask_for_invalid_next_bar_opens_with_decision_idx,
    sparse_execution_rebalance_event_mask,
)
from portfolio_backtester.simulation.kernel import simulate_portfolio
from portfolio_backtester.timing.trade_execution_timing import (
    map_sparse_target_weights_to_execution_dates,
)


def test_rebalance_mask_sparse_only_where_explicit_active_rows_exist():
    cal = pd.date_range("2023-01-01", periods=5, freq="D")
    sparse = pd.DataFrame(
        {
            "A": [1.0, 1.0],
            "B": [0.0, 0.0],
        },
        index=pd.DatetimeIndex([cal[0], cal[3]]),
    )
    m = sparse_execution_rebalance_event_mask(sparse, cal, ["A", "B"])
    assert m.dtype == np.bool_
    assert bool(m[0]) and bool(m[3])
    assert int(np.sum(m)) == 2


def test_ffill_dense_weights_have_no_implicit_rebalance_from_builder_mask():
    cal = pd.date_range("2023-01-01", periods=4, freq="D")
    sparse = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[cal[0]])
    wd = sparse.reindex(cal).ffill().fillna(0.0)
    close = np.ones((len(cal), 2), dtype=np.float64)
    close[:, 0] = 100.0
    close[:, 1] = 50.0
    mk = np.ones_like(close, dtype=bool)

    inp = build_portfolio_simulation_input(
        weights_daily=wd,
        price_index=cal,
        valid_cols=["A", "B"],
        close_arr=close,
        close_price_mask_arr=mk,
        sparse_execution_targets=sparse,
        trade_execution_timing="bar_close",
    )
    assert inp.rebalance_mask[0]
    assert not bool(inp.rebalance_mask[1:].any())


def test_adjacent_duplicate_sparse_targets_both_rebalance_events():
    cal = pd.date_range("2023-01-01", periods=3, freq="D")
    sparse = pd.DataFrame(
        {"A": [0.6, 0.6], "B": [0.4, 0.4]},
        index=pd.DatetimeIndex([cal[0], cal[1]]),
    )
    m = sparse_execution_rebalance_event_mask(sparse, cal, ["A", "B"])
    assert bool(m[0]) and bool(m[1])
    assert not bool(m[2])


def test_repeated_sparse_rows_keep_multiple_rebalance_events():
    cal = pd.date_range("2023-01-01", periods=3, freq="D")
    sparse = pd.DataFrame({"A": [1.0, 1.0]}, index=pd.DatetimeIndex([cal[0], cal[2]]))
    wd = sparse.reindex(cal).ffill().fillna(0.0)
    close = np.full((len(cal), 1), 100.0, dtype=np.float64)
    mk = np.ones_like(close, dtype=bool)
    inp = build_portfolio_simulation_input(
        weights_daily=wd,
        price_index=cal,
        valid_cols=["A"],
        close_arr=close,
        close_price_mask_arr=mk,
        sparse_execution_targets=sparse,
    )
    assert inp.rebalance_mask[0] and inp.rebalance_mask[2] and not inp.rebalance_mask[1]


def test_all_nan_skipped_row_does_not_create_rebalance_event():
    cal = pd.date_range("2023-01-01", periods=3, freq="D")
    sparse = pd.DataFrame(
        {"A": [1.0, np.nan], "B": [0.0, np.nan]},
        index=pd.DatetimeIndex([cal[0], cal[1]]),
    )
    m = sparse_execution_rebalance_event_mask(sparse, cal, ["A", "B"])
    assert m[0] and not m[1]


def test_next_bar_open_sparse_remapped_shifts_mask_to_execution_calendar_day():
    cal = pd.DatetimeIndex(pd.date_range("2023-01-03", periods=4, freq="D"))
    decision = pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex([cal[0]]))
    exec_sparse = map_sparse_target_weights_to_execution_dates(
        decision, trade_execution_timing="next_bar_open", calendar=cal
    )
    m = sparse_execution_rebalance_event_mask(exec_sparse, cal, ["A"])
    assert not m[0]
    assert m[1]
    assert not bool(m[2:].any())


def test_explicit_rebalance_mask_overrides_sparse_derivation():
    cal = pd.date_range("2023-01-01", periods=3, freq="D")
    sparse = pd.DataFrame({"A": [1.0]}, index=[cal[0]])
    wd = sparse.reindex(cal).ffill().fillna(0.0)
    close = np.full((3, 1), 100.0)
    mk = np.ones_like(close, dtype=bool)
    override = np.array([False, True, False], dtype=np.bool_)
    inp = build_portfolio_simulation_input(
        weights_daily=wd,
        price_index=cal,
        valid_cols=["A"],
        close_arr=close,
        close_price_mask_arr=mk,
        sparse_execution_targets=sparse,
        rebalance_mask_arr=override,
    )
    np.testing.assert_array_equal(inp.rebalance_mask, override)


def test_build_sets_execution_equal_to_close_for_bar_close():
    cal = pd.date_range("2023-01-01", periods=2, freq="D")
    wd = pd.DataFrame({"A": [1.0, 1.0]}, index=cal)
    close = np.array([[100.0], [110.0]], dtype=np.float64)
    mk = np.ones_like(close, dtype=bool)
    inp = build_portfolio_simulation_input(
        weights_daily=wd,
        price_index=cal,
        valid_cols=["A"],
        close_arr=close,
        close_price_mask_arr=mk,
        rebalance_mask_arr=np.ones(len(cal), dtype=np.bool_),
        trade_execution_timing="bar_close",
    )
    assert inp.execution_timing == EXECUTION_TIMING_BAR_CLOSE
    np.testing.assert_array_equal(inp.execution_prices, inp.close_prices)
    np.testing.assert_array_equal(inp.execution_price_mask, inp.close_price_mask)


def test_build_uses_open_prices_for_execution_when_next_bar_open():
    cal = pd.date_range("2023-01-01", periods=2, freq="D")
    wd = pd.DataFrame({"A": [1.0, 1.0]}, index=cal)
    close = np.array([[100.0], [100.0]], dtype=np.float64)
    open_ = np.array([[99.0], [101.0]], dtype=np.float64)
    mk = np.ones_like(close, dtype=bool)
    inp = build_portfolio_simulation_input(
        weights_daily=wd,
        price_index=cal,
        valid_cols=["A"],
        close_arr=close,
        close_price_mask_arr=mk,
        open_arr=open_,
        open_price_mask_arr=mk,
        rebalance_mask_arr=np.ones(len(cal), dtype=np.bool_),
        trade_execution_timing="next_bar_open",
    )
    assert inp.execution_timing == EXECUTION_TIMING_NEXT_BAR_OPEN
    np.testing.assert_array_equal(inp.execution_prices, open_)
    np.testing.assert_array_equal(inp.close_prices, close)


def test_multiindex_ohlc_open_used_for_execution_and_close_for_mark_to_market():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    cols = pd.MultiIndex.from_tuples(
        [
            ("A", "Open"),
            ("A", "Close"),
            ("B", "Open"),
            ("B", "Close"),
        ],
        names=["Ticker", "Field"],
    )
    daily = pd.DataFrame(
        [
            [100.0, 100.0, 50.0, 50.0],
            [90.0, 100.0, 55.0, 50.0],
        ],
        index=dates,
        columns=cols,
    )
    close_df = daily.xs("Close", level="Field", axis=1)
    valid_cols = ["A", "B"]
    price_ix = pd.DatetimeIndex(close_df.index)
    close_arr, close_mask = prepare_close_arrays_for_simulation(
        market_data_panel=None,
        close_prices_df=close_df,
        price_index=price_ix,
        valid_cols=valid_cols,
    )
    open_frame = extract_open_frame_from_ohlc(daily)
    assert open_frame is not None
    open_arr, open_mask = prepare_open_arrays_for_simulation(
        market_data_panel=None,
        open_prices_df=open_frame,
        price_index=price_ix,
        valid_cols=valid_cols,
    )
    wd = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=dates, columns=valid_cols)
    inp = build_portfolio_simulation_input(
        weights_daily=wd,
        price_index=price_ix,
        valid_cols=valid_cols,
        close_arr=close_arr,
        close_price_mask_arr=close_mask,
        open_arr=open_arr,
        open_price_mask_arr=open_mask,
        rebalance_mask_arr=np.ones(len(dates), dtype=np.bool_),
        trade_execution_timing="next_bar_open",
    )
    assert inp.execution_prices[1, 1] == pytest.approx(55.0)
    assert inp.close_prices[1, 1] == pytest.approx(50.0)
    g = {
        "portfolio_value": 10_000.0,
        "commission_per_share": 0.0,
        "commission_min_per_order": 0.0,
        "commission_max_percent_of_trade": 0.0,
        "slippage_bps": 0.0,
    }
    out = simulate_portfolio(
        inp,
        global_config=g,
        scenario_config={"allocation_mode": "reinvestment"},
    )
    assert float(out.positions.iloc[1, 1]) == pytest.approx(9000.0 / 55.0, rel=0.0, abs=1e-6)


def test_next_bar_open_extends_rebalance_when_open_invalid_for_nonzero_targets():
    cal = pd.date_range("2023-01-01", periods=3, freq="D")
    wd = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=cal)
    close = np.full((3, 1), 100.0)
    close_mk = np.ones_like(close, dtype=bool)
    open_ = np.full((3, 1), 100.0)
    open_mk = np.array([[False], [True], [True]], dtype=bool)
    rb = np.array([True, False, False], dtype=bool)
    inp = build_portfolio_simulation_input(
        weights_daily=wd,
        price_index=cal,
        valid_cols=["A"],
        close_arr=close,
        close_price_mask_arr=close_mk,
        open_arr=open_,
        open_price_mask_arr=open_mk,
        rebalance_mask_arr=rb,
        trade_execution_timing="next_bar_open",
    )
    assert inp.rebalance_mask[0] and inp.rebalance_mask[1]
    assert not bool(inp.rebalance_mask[2])


def test_next_bar_open_propagates_across_several_invalid_open_sessions():
    n = 6
    cal = pd.date_range("2023-01-01", periods=n, freq="D")
    wd = pd.DataFrame(np.ones((n, 1)), index=cal, columns=["A"])
    close = np.full((n, 1), 100.0)
    close_mk = np.ones_like(close, dtype=bool)
    open_ = np.full((n, 1), 100.0)
    open_mk = np.array([[False], [False], [False], [True], [True], [True]], dtype=bool)
    rb = np.array([True] + [False] * (n - 1), dtype=bool)
    inp = build_portfolio_simulation_input(
        weights_daily=wd,
        price_index=cal,
        valid_cols=["A"],
        close_arr=close,
        close_price_mask_arr=close_mk,
        open_arr=open_,
        open_price_mask_arr=open_mk,
        rebalance_mask_arr=rb,
        trade_execution_timing="next_bar_open",
    )
    assert (
        inp.rebalance_mask[0]
        and inp.rebalance_mask[1]
        and inp.rebalance_mask[2]
        and inp.rebalance_mask[3]
    )
    assert not bool(inp.rebalance_mask[4:].any())


def test_next_bar_open_last_session_invalid_does_not_extend_past_end():
    cal = pd.date_range("2023-01-01", periods=3, freq="D")
    wd = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=cal)
    close = np.full((3, 1), 100.0)
    close_mk = np.ones_like(close, dtype=bool)
    open_ = np.full((3, 1), 100.0)
    open_mk = np.zeros((3, 1), dtype=bool)
    rb = np.array([False, False, True], dtype=bool)
    inp = build_portfolio_simulation_input(
        weights_daily=wd,
        price_index=cal,
        valid_cols=["A"],
        close_arr=close,
        close_price_mask_arr=close_mk,
        open_arr=open_,
        open_price_mask_arr=open_mk,
        rebalance_mask_arr=rb,
        trade_execution_timing="next_bar_open",
    )
    np.testing.assert_array_equal(inp.rebalance_mask, rb)


def test_bar_close_does_not_propagate_rebalance_for_invalid_opens():
    cal = pd.date_range("2023-01-01", periods=3, freq="D")
    wd = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=cal)
    close = np.full((3, 1), 100.0)
    close_mk = np.ones_like(close, dtype=bool)
    open_ = np.full((3, 1), 100.0)
    open_mk = np.array([[False], [True], [True]], dtype=bool)
    rb = np.array([True, False, False], dtype=bool)
    inp = build_portfolio_simulation_input(
        weights_daily=wd,
        price_index=cal,
        valid_cols=["A"],
        close_arr=close,
        close_price_mask_arr=close_mk,
        open_arr=open_,
        open_price_mask_arr=open_mk,
        rebalance_mask_arr=rb,
        trade_execution_timing="bar_close",
    )
    np.testing.assert_array_equal(inp.rebalance_mask, rb)


def test_propagate_rebalance_mask_misaligned_shapes_raise():
    rb = np.array([True, False], dtype=bool)
    w = np.ones((1, 1), dtype=np.float64)
    m = np.ones((1, 1), dtype=bool)
    with pytest.raises(ValueError, match="align on time"):
        propagate_rebalance_mask_for_invalid_next_bar_opens(rb, w, m)


def test_next_bar_open_ledger_decision_idx_pins_across_invalid_open_retries():
    rb = np.array([True, True, False, False], dtype=np.bool_)
    weights = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float64)
    open_mk = np.array([[True], [False], [False], [True]], dtype=bool)
    rb_out, dec = propagate_rebalance_mask_for_invalid_next_bar_opens_with_decision_idx(
        rb, weights, open_mk
    )
    assert bool(rb_out[1]) and bool(rb_out[2]) and bool(rb_out[3])
    assert int(dec[1, 0]) == 0
    assert int(dec[2, 0]) == 0
    assert int(dec[3, 0]) == 0


def test_extract_field_frame_unnamed_multiindex_last_level():
    dates = pd.date_range("2023-01-01", periods=1, freq="D")
    cols = pd.MultiIndex.from_tuples([("A", "Open"), ("A", "Close")], names=[None, None])
    daily = pd.DataFrame([[1.0, 2.0]], index=dates, columns=cols)
    close_df = extract_field_frame_from_ohlc(daily, "Close")
    assert list(close_df.columns) == ["A"]
    assert float(close_df.iloc[0, 0]) == pytest.approx(2.0)
    open_frame = extract_open_frame_from_ohlc(daily)
    assert open_frame is not None
    assert float(open_frame.iloc[0, 0]) == pytest.approx(1.0)


def test_next_bar_open_without_open_raises():
    cal = pd.date_range("2023-01-01", periods=2, freq="D")
    wd = pd.DataFrame({"A": [1.0, 1.0]}, index=cal)
    close = np.full((2, 1), 100.0)
    mk = np.ones_like(close, dtype=bool)
    with pytest.raises(ValueError):
        build_portfolio_simulation_input(
            weights_daily=wd,
            price_index=cal,
            valid_cols=["A"],
            close_arr=close,
            close_price_mask_arr=mk,
            rebalance_mask_arr=np.ones(len(cal), dtype=np.bool_),
            trade_execution_timing="next_bar_open",
        )
