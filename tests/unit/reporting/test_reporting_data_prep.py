"""Unit tests for reporting data preparation.

Focus:
- JSON serialization for report artifacts
- Stable output schemas for report data files
- Graceful handling of missing/empty data
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_backtester.reporting.optimizer_report_generator import OptimizerReportGenerator
from portfolio_backtester.reporting.table_generator import generate_transaction_history_csv


def test_make_json_serializable_handles_nested_numpy_and_pandas() -> None:
    gen = OptimizerReportGenerator(base_reports_dir="unused")

    ts = pd.Timestamp("2024-01-01")
    idx = pd.date_range("2024-01-01", periods=2)

    obj = {
        "int": np.int64(7),
        "float": np.float64(1.25),
        "arr": np.array([1, 2, 3], dtype=np.int64),
        "series": pd.Series([1.0, np.nan], index=["a", "b"]),
        "df": pd.DataFrame({"x": [1, 2]}),
        "ts": ts,
        "idx": idx,
        "nested": [np.float32(2.5), {"n": np.int32(3)}],
        "none_like": np.nan,
    }

    converted = gen._make_json_serializable(obj)

    # Must be JSON dumpable
    dumped = json.dumps(converted)
    loaded = json.loads(dumped)

    assert loaded["int"] == 7
    assert loaded["float"] == 1.25
    assert loaded["arr"] == [1, 2, 3]
    assert loaded["nested"][0] == 2.5
    assert loaded["nested"][1]["n"] == 3
    assert loaded["none_like"] is None

    # pandas objects become dicts
    assert isinstance(loaded["series"], dict)
    assert isinstance(loaded["df"], dict)


def test_save_optimization_data_writes_expected_files(tmp_path: Path) -> None:
    gen = OptimizerReportGenerator(base_reports_dir=str(tmp_path))

    run_dir = tmp_path / "run"
    (run_dir / "data").mkdir(parents=True)

    optimization_data = {
        "strategy_name": "S",
        "optimal_parameters": {"p": np.int64(2)},
        "performance_metrics": {"Sharpe": np.float64(1.1)},
        "parameter_importance": {"p": 0.9},
        "trials_data": [
            {"number": 0, "value": 1.0, "params": {"p": 1}, "state": "COMPLETE"},
            {"number": 1, "value": 2.0, "params": {"p": 2}, "state": "COMPLETE"},
        ],
    }

    gen.save_optimization_data(run_dir=run_dir, optimization_data=optimization_data)

    main_json = run_dir / "data" / "optimization_results.json"
    importance_json = run_dir / "data" / "parameter_importance.json"
    trials_csv = run_dir / "data" / "trials_data.csv"

    assert main_json.exists()
    assert importance_json.exists()
    assert trials_csv.exists()

    # Verify JSON content is parseable
    parsed = json.loads(main_json.read_text(encoding="utf-8"))
    assert parsed["strategy_name"] == "S"
    assert parsed["optimal_parameters"]["p"] == 2

    # Verify CSV has rows
    df = pd.read_csv(trials_csv)
    assert len(df) == 2
    assert set(df.columns) >= {"number", "value", "params", "state"}


def test_generate_transaction_history_csv_creates_two_rows_per_trade(tmp_path: Path) -> None:
    trade_history = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "quantity": 10,
                "entry_date": pd.Timestamp("2024-01-01"),
                "exit_date": pd.Timestamp("2024-01-05"),
                "entry_price": 100.0,
                "exit_price": 110.0,
                "commission_entry": 1.0,
                "commission_exit": 1.0,
                "pnl_net": 98.0,
            },
            {
                "ticker": "BBB",
                "quantity": -5,  # short
                "entry_date": pd.Timestamp("2024-02-01"),
                "exit_date": pd.Timestamp("2024-02-03"),
                "entry_price": 50.0,
                "exit_price": 45.0,
                "commission_entry": 0.5,
                "commission_exit": 0.5,
                "pnl_net": 24.0,
            },
        ]
    )

    backtest_results = {"strat": {"trade_history": trade_history}}

    generate_transaction_history_csv(backtest_results=backtest_results, report_dir=str(tmp_path))

    out = tmp_path / "transaction_history_strat.csv"
    assert out.exists()

    tx = pd.read_csv(out)
    # Each trade becomes entry + exit
    assert len(tx) == 4
    assert set(tx.columns) == {
        "datetime",
        "symbol",
        "trade_type",
        "amount_of_shares",
        "trade_price",
        "calculated_commissions",
        "profit_loss",
    }

    # One exit row per trade should have profit_loss populated
    assert tx["profit_loss"].notna().sum() == 2


def test_generate_transaction_history_csv_skips_when_required_columns_missing(
    tmp_path: Path,
) -> None:
    # Missing required columns -> should not create file
    trade_history = pd.DataFrame([{"ticker": "AAA", "quantity": 1}])
    backtest_results = {"strat": {"trade_history": trade_history}}

    generate_transaction_history_csv(backtest_results=backtest_results, report_dir=str(tmp_path))

    out = tmp_path / "transaction_history_strat.csv"
    assert not out.exists()
