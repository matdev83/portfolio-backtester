"""Tests for research protocol static matplotlib outputs (bootstrap diagnostics, cost chart)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")

from portfolio_backtester.research.static_charts import (
    write_bootstrap_distribution_chart_pngs,
    write_cost_sensitivity_line_chart_png,
)


def test_write_bootstrap_distribution_chart_pngs_creates_histogram_and_qq(tmp_path: Path) -> None:
    df = pd.DataFrame({"value": np.linspace(0.01, 0.99, 25)})
    path_csv = tmp_path / "bootstrap_distribution_random_wfo_architecture.csv"
    df.to_csv(path_csv, index=False)
    outputs = write_bootstrap_distribution_chart_pngs(tmp_path)
    assert outputs
    names = [p.name for p in outputs]
    assert any(n.endswith("_histogram.png") for n in names)
    assert any(n.endswith("_qq.png") for n in names)
    for p in outputs:
        assert p.is_file()
        assert p.stat().st_size > 0


def test_write_bootstrap_distribution_chart_pngs_skips_empty_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "bootstrap_distribution_block_shuffled_returns.csv"
    csv_path.write_text("value\n", encoding="utf-8")
    assert write_bootstrap_distribution_chart_pngs(tmp_path) == []


def test_write_bootstrap_distribution_chart_pngs_skips_csv_without_value_column(
    tmp_path: Path,
) -> None:
    path_csv = tmp_path / "bootstrap_distribution_random_wfo_architecture.csv"
    pd.DataFrame({"stat": [0.1, 0.2]}).to_csv(path_csv, index=False)
    assert write_bootstrap_distribution_chart_pngs(tmp_path) == []


def test_write_bootstrap_distribution_chart_pngs_histogram_only_when_single_sample(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame({"value": [0.5]})
    path_csv = tmp_path / "bootstrap_distribution_random_wfo_architecture.csv"
    df.to_csv(path_csv, index=False)
    outputs = write_bootstrap_distribution_chart_pngs(tmp_path)
    assert len(outputs) == 1
    assert outputs[0].name.endswith("_histogram.png")


def test_write_cost_sensitivity_line_chart_png_reads_survival_metric_from_summary(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "slippage_bps": 0.0,
            "commission_multiplier": 1.0,
            "survives": True,
            "Sharpe": 1.5,
        },
        {
            "slippage_bps": 10.0,
            "commission_multiplier": 1.0,
            "survives": True,
            "Sharpe": 0.5,
        },
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "cost_sensitivity.csv", index=False)
    (tmp_path / "cost_sensitivity_summary.yaml").write_text(
        yaml.safe_dump({"survival_metric": "Sharpe"}),
        encoding="utf-8",
    )
    out = write_cost_sensitivity_line_chart_png(tmp_path)
    assert out is not None
    assert out.name == "cost_sensitivity_survival_curve.png"
    assert out.stat().st_size > 0
