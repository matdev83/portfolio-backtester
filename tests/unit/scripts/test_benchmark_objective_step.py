"""Unit tests for ``scripts/benchmark_objective_step``."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_benchmark_module():
    path = _REPO_ROOT / "scripts" / "benchmark_objective_step.py"
    name = "_benchmark_objective_step"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_default_small_scenarios_constant() -> None:
    m = _load_benchmark_module()
    assert m.DEFAULT_SMALL_SCENARIOS == ("single_asset", "fixed_10", "momentum_50")


def test_objective_step_record_schema() -> None:
    m = _load_benchmark_module()
    records = m.run_benchmarks(["single_asset"])
    assert len(records) == 1
    r = records[0]
    d = m.benchmark_record_to_dict(r)
    for key in (
        "scenario",
        "target_generation_ms",
        "simulation_ms",
        "metrics_ms",
        "total_ms",
        "num_assets",
        "num_days",
        "num_rebalances",
    ):
        assert key in d
    assert d["scenario"] == "single_asset"
    assert d["num_assets"] >= 1
    assert d["num_days"] >= 1
    assert d["num_rebalances"] >= 1
    assert d["total_ms"] == pytest.approx(
        d["target_generation_ms"] + d["simulation_ms"] + d["metrics_ms"], rel=0, abs=1e-9
    )


def test_objective_step_all_small_scenarios() -> None:
    m = _load_benchmark_module()
    records = m.run_benchmarks(None)
    names = {r.scenario for r in records}
    assert names == set(m.DEFAULT_SMALL_SCENARIOS)


def test_main_writes_json_file(tmp_path: Path) -> None:
    m = _load_benchmark_module()
    out = tmp_path / "bench.json"
    code = m.main(["--scenario", "single_asset", "--output", str(out)])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "records" in data
    assert len(data["records"]) == 1
    assert data["records"][0]["scenario"] == "single_asset"


def test_main_stdout_json(capsys: pytest.CaptureFixture[str]) -> None:
    m = _load_benchmark_module()
    code = m.main(["--scenario", "fixed_10"])
    assert code == 0
    captured = capsys.readouterr().out.strip()
    data = json.loads(captured)
    assert len(data["records"]) == 1
    assert data["records"][0]["scenario"] == "fixed_10"


def test_enforce_fails_when_over_threshold(tmp_path: Path) -> None:
    m = _load_benchmark_module()
    baseline = tmp_path / "base.json"
    baseline.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "scenario": "single_asset",
                        "total_ms": 0.0,
                        "target_generation_ms": 0.0,
                        "simulation_ms": 0.0,
                        "metrics_ms": 0.0,
                        "num_assets": 1,
                        "num_days": 1,
                        "num_rebalances": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    code = m.main(
        [
            "--scenario",
            "single_asset",
            "--baseline",
            str(baseline),
            "--enforce",
            "--threshold",
            "1.0",
        ]
    )
    assert code == 1


def test_enforce_passes_with_loose_baseline(tmp_path: Path) -> None:
    m = _load_benchmark_module()
    baseline = tmp_path / "base.json"
    baseline.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "scenario": "single_asset",
                        "total_ms": 1.0e15,
                        "target_generation_ms": 0.0,
                        "simulation_ms": 0.0,
                        "metrics_ms": 0.0,
                        "num_assets": 1,
                        "num_days": 1,
                        "num_rebalances": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    code = m.main(
        [
            "--scenario",
            "single_asset",
            "--baseline",
            str(baseline),
            "--enforce",
            "--threshold",
            "1.0",
        ]
    )
    assert code == 0


def test_violations_for_enforce_unit() -> None:
    m = _load_benchmark_module()
    rec = m.BenchmarkRecord(
        scenario="x",
        target_generation_ms=0.0,
        simulation_ms=0.0,
        metrics_ms=0.0,
        total_ms=300.0,
        num_assets=1,
        num_days=10,
        num_rebalances=2,
    )
    baseline_map = {"x": 100.0}
    viol = m.violations_for_enforce([rec], baseline_map, threshold=2.0)
    assert viol == [("x", 200.0, 300.0)]
    assert m.violations_for_enforce([rec], baseline_map, threshold=3.0) == []


def test_cli_subprocess_smoke() -> None:
    py = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)
    script = _REPO_ROOT / "scripts" / "benchmark_objective_step.py"
    proc = subprocess.run(
        [str(py), str(script), "--scenario", "single_asset"],
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout.strip())
    assert data["records"][0]["scenario"] == "single_asset"
