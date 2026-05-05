"""Unit tests for ``scripts/benchmark_optimizer_objective``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_benchmark_module():
    path = _REPO_ROOT / "scripts" / "benchmark_optimizer_objective.py"
    name = "_benchmark_optimizer_objective"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_execute_timed_evaluations_both_counts_two_calls() -> None:
    m = _load_benchmark_module()
    evaluator = MagicMock()
    evaluator.evaluate_parameters.side_effect = [
        MagicMock(objective_value=1.0),
        MagicMock(objective_value=2.0),
    ]
    canonical = MagicMock()
    data = MagicMock()
    backtester = MagicMock()
    timings, meta = m.execute_timed_evaluations(
        evaluator,
        parameters={"a": 1},
        scenario_config=canonical,
        data=data,
        backtester=backtester,
        timing_mode="both",
    )
    assert evaluator.evaluate_parameters.call_count == 2
    assert "evaluate_parameters_cold_s" in timings
    assert "evaluate_parameters_warm_s" in timings
    assert timings["evaluate_parameters_cold_s"] >= 0.0
    assert timings["evaluate_parameters_warm_s"] >= 0.0
    assert meta["objective_value_cold"] == 1.0
    assert meta["objective_value_warm"] == 2.0


def test_execute_timed_evaluations_cold_single_call() -> None:
    m = _load_benchmark_module()
    evaluator = MagicMock()
    evaluator.evaluate_parameters.return_value = MagicMock(objective_value=3.5)
    timings, meta = m.execute_timed_evaluations(
        evaluator,
        parameters={},
        scenario_config=MagicMock(),
        data=MagicMock(),
        backtester=MagicMock(),
        timing_mode="cold",
    )
    assert evaluator.evaluate_parameters.call_count == 1
    assert "evaluate_parameters_cold_s" in timings
    assert "evaluate_parameters_warm_s" not in timings
    assert meta["objective_value_cold"] == 3.5
    assert meta.get("objective_value_warm") is None


def test_execute_timed_evaluations_warm_is_second_timed_call() -> None:
    m = _load_benchmark_module()
    evaluator = MagicMock()
    evaluator.evaluate_parameters.side_effect = [
        MagicMock(objective_value=-1.0),
        MagicMock(objective_value=99.0),
    ]
    timings, meta = m.execute_timed_evaluations(
        evaluator,
        parameters={},
        scenario_config=MagicMock(),
        data=MagicMock(),
        backtester=MagicMock(),
        timing_mode="warm",
    )
    assert evaluator.evaluate_parameters.call_count == 2
    assert "evaluate_parameters_warm_s" in timings
    assert "evaluate_parameters_cold_s" not in timings
    assert meta["objective_value_warm"] == 99.0


def test_optimizer_objective_result_to_dict_includes_timing_sections() -> None:
    m = _load_benchmark_module()
    rec = m.OptimizerObjectiveBenchmarkResult(
        scenario_file="sc.yaml",
        timing_mode="both",
        train_window_months=6,
        test_window_months=3,
        trial_start_date="2019-01-01",
        trial_end_date="2020-01-01",
        data_fetch_s=0.1,
        evaluate_parameters_cold_s=0.2,
        evaluate_parameters_warm_s=0.05,
        n_windows=2,
        metrics=["Sharpe"],
        trial_params={"x": 1},
        objective_value_cold=1.0,
        objective_value_warm=1.1,
    )
    d = m.optimizer_objective_benchmark_result_to_dict(rec)
    assert d["timing_mode"] == "both"
    assert d["sections"]["data_fetch_s"] == 0.1
    assert d["sections"]["evaluate_parameters_cold_s"] == 0.2
    assert d["sections"]["evaluate_parameters_warm_s"] == 0.05


def test_main_writes_json_with_explicit_timing_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = _load_benchmark_module()
    out = tmp_path / "bench.json"
    fake = m.OptimizerObjectiveBenchmarkResult(
        scenario_file="x.yaml",
        timing_mode="both",
        train_window_months=12,
        test_window_months=6,
        trial_start_date=None,
        trial_end_date=None,
        data_fetch_s=0.01,
        evaluate_parameters_cold_s=0.02,
        evaluate_parameters_warm_s=0.03,
        n_windows=1,
        metrics=["Calmar"],
        trial_params={},
        objective_value_cold=0.5,
        objective_value_warm=0.5,
    )
    monkeypatch.setattr(m, "run_optimizer_objective_benchmark", MagicMock(return_value=fake))
    code = m.main(["--scenario-filename", str(tmp_path / "missing.yaml"), "--output", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["sections"]["evaluate_parameters_cold_s"] == 0.02
    assert payload["sections"]["evaluate_parameters_warm_s"] == 0.03
    assert payload["timing_mode"] == "both"


def test_main_json_cold_mode_omits_warm_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = _load_benchmark_module()
    out = tmp_path / "cold.json"
    fake = m.OptimizerObjectiveBenchmarkResult(
        scenario_file="y.yaml",
        timing_mode="cold",
        train_window_months=12,
        test_window_months=6,
        trial_start_date=None,
        trial_end_date=None,
        data_fetch_s=0.01,
        evaluate_parameters_cold_s=0.02,
        evaluate_parameters_warm_s=None,
        n_windows=1,
        metrics=["Sharpe"],
        trial_params={},
        objective_value_cold=1.0,
        objective_value_warm=None,
    )
    monkeypatch.setattr(m, "run_optimizer_objective_benchmark", MagicMock(return_value=fake))
    code = m.main(
        [
            "--scenario-filename",
            str(tmp_path / "s.yaml"),
            "--timing-mode",
            "cold",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["timing_mode"] == "cold"
    assert payload["sections"]["evaluate_parameters_cold_s"] == 0.02
    assert payload["sections"].get("evaluate_parameters_warm_s") is None


def test_cli_help_smoke() -> None:
    m = _load_benchmark_module()
    with pytest.raises(SystemExit) as exc:
        m.main(["--help"])
    assert exc.value.code == 0


def test_cli_subprocess_help_smoke() -> None:
    import subprocess

    py = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)
    script = _REPO_ROOT / "scripts" / "benchmark_optimizer_objective.py"
    proc = subprocess.run(
        [str(py), str(script), "--help"],
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "timing-mode" in proc.stdout
