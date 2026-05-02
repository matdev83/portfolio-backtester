"""Filesystem artifacts for research protocol runs."""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from portfolio_backtester.canonical_config import CanonicalScenarioConfig
from portfolio_backtester.research.hashing import stable_hash
from portfolio_backtester.research.protocol_config import (
    RESEARCH_PROTOCOL_ARTIFACT_VERSION,
    DoubleOOSWFOProtocolConfig,
)
from portfolio_backtester.research.results import (
    SelectedProtocol,
    UnseenValidationResult,
    WFOArchitectureResult,
)


class ResearchArtifactExistsError(RuntimeError):
    """Raised when a research artifact path already exists and overwrite is refused."""


def sanitize_scenario_name(name: str) -> str:
    """Return a filesystem-safe segment derived from a scenario name."""

    trimmed = name.strip()
    if not trimmed:
        return "_"
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", trimmed)


def protocol_config_to_plain(config: DoubleOOSWFOProtocolConfig) -> dict[str, Any]:
    """Serialize protocol configuration for YAML hashing and dumps."""

    out: dict[str, Any] = {
        "enabled": config.enabled,
        "global_train_period": {
            "start": pd.Timestamp(config.global_train_period.start).isoformat(),
            "end": pd.Timestamp(config.global_train_period.end).isoformat(),
        },
        "unseen_test_period": {
            "start": pd.Timestamp(config.unseen_test_period.start).isoformat(),
            "end": pd.Timestamp(config.unseen_test_period.end).isoformat(),
        },
        "wfo_window_grid": {
            "train_window_months": list(config.wfo_window_grid.train_window_months),
            "test_window_months": list(config.wfo_window_grid.test_window_months),
            "wfo_step_months": list(config.wfo_window_grid.wfo_step_months),
            "walk_forward_type": list(config.wfo_window_grid.walk_forward_type),
        },
        "selection": {
            "top_n": config.selection.top_n,
            "metric": config.selection.metric,
        },
        "final_unseen_mode": config.final_unseen_mode.value,
        "lock": {
            "enabled": config.lock.enabled,
            "refuse_overwrite": config.lock.refuse_overwrite,
        },
        "reporting": {
            "enabled": config.reporting.enabled,
            "generate_heatmaps": config.reporting.generate_heatmaps,
            "heatmap_metrics": list(config.reporting.heatmap_metrics),
            "generate_html": config.reporting.generate_html,
        },
    }
    if config.constraints:
        out["constraints"] = [
            {"metric": c.display_key, "min_value": c.min_value, "max_value": c.max_value}
            for c in config.constraints
        ]
    if config.composite_scoring is not None:
        cs = config.composite_scoring
        out["scoring"] = {
            "type": "composite_rank",
            "weights": {k: float(w) for k, w in cs.weights},
            "directions": dict(cs.directions),
        }
    rs = config.robust_selection
    out["robust_selection"] = {
        "enabled": rs.enabled,
        "weights": {
            "cell": float(rs.cell_weight),
            "neighbor_median": float(rs.neighbor_median_weight),
            "neighbor_min": float(rs.neighbor_min_weight),
        },
    }
    cost_sens = config.cost_sensitivity
    out["cost_sensitivity"] = {
        "enabled": cost_sens.enabled,
        "slippage_bps_grid": list(cost_sens.slippage_bps_grid),
        "commission_multiplier_grid": list(cost_sens.commission_multiplier_grid),
        "run_on": cost_sens.run_on.value,
    }
    bs = config.bootstrap
    out["bootstrap"] = {
        "enabled": bs.enabled,
        "n_samples": int(bs.n_samples),
        "random_seed": int(bs.random_seed),
        "random_wfo_architecture": {"enabled": bool(bs.random_wfo_architecture.enabled)},
        "block_shuffled_returns": {
            "enabled": bool(bs.block_shuffled_returns.enabled),
            "block_size_days": int(bs.block_shuffled_returns.block_size_days),
        },
        "block_shuffled_positions": {
            "enabled": bool(bs.block_shuffled_positions.enabled),
            "block_size_days": int(bs.block_shuffled_positions.block_size_days),
        },
        "random_strategy_parameters": {
            "enabled": bool(bs.random_strategy_parameters.enabled),
            "sample_size": int(bs.random_strategy_parameters.sample_size),
        },
    }
    ex = config.execution
    out["execution"] = {
        "max_grid_cells": int(ex.max_grid_cells),
        "fail_fast": bool(ex.fail_fast),
    }
    return out


class ResearchArtifactWriter:
    """Creates per-run directories under a shared research artifact root."""

    def __init__(self, base_dir: Path | str) -> None:
        self._base = Path(base_dir)

    def scenario_protocol_root(self, scenario_name: str) -> Path:
        """Return ``<base>/<safe_scenario>/research_protocol`` (per-scenario protocol root)."""

        safe = sanitize_scenario_name(scenario_name)
        return self._base / safe / "research_protocol"

    def create_run_directory(self, scenario_name: str, *, run_id: str | None = None) -> Path:
        """Materialize ``<base>/<safe_scenario>/research_protocol/<run_id>``.

        Args:
            scenario_name: Raw scenario label from configuration.
            run_id: Optional stable id; defaults to a UTC timestamp label with a
                short random suffix for uniqueness on fast successive calls.

        Returns:
            Absolute path to the created directory.
        """

        safe = sanitize_scenario_name(scenario_name)
        if run_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            run_id = f"{ts}_{secrets.token_hex(4)}"
        path = self._base / safe / "research_protocol" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def write_grid_results(run_dir: Path | str, grid_results: Sequence[WFOArchitectureResult]) -> None:
    """Write ``wfo_architecture_grid.csv`` summarizing grid search rows."""

    root = Path(run_dir)
    metric_keys: set[str] = set()
    for row in grid_results:
        metric_keys.update(row.metrics.keys())
    ordered_metrics = sorted(metric_keys)
    rows_out: list[dict[str, Any]] = []
    for r in grid_results:
        base = {
            **r.architecture.to_dict(),
            "score": r.score,
            "robust_score": r.robust_score,
            "n_evaluations": r.n_evaluations,
            "constraint_passed": r.constraint_passed,
            "constraint_failures": (
                "; ".join(r.constraint_failures) if r.constraint_failures else ""
            ),
            "best_parameters_json": json.dumps(dict(r.best_parameters), sort_keys=True),
        }
        for mk in ordered_metrics:
            base[mk] = r.metrics.get(mk)
        rows_out.append(base)
    cols = [
        "train_window_months",
        "test_window_months",
        "wfo_step_months",
        "walk_forward_type",
        "score",
        "robust_score",
        "n_evaluations",
        "constraint_passed",
        "constraint_failures",
        "best_parameters_json",
        *ordered_metrics,
    ]
    df = pd.DataFrame(rows_out, columns=cols)
    df.to_csv(root / "wfo_architecture_grid.csv", index=False)


def write_selected_protocols(run_dir: Path | str, protocols: Sequence[SelectedProtocol]) -> None:
    """Write ``selected_protocols.yaml`` with ranked protocol summaries."""

    root = Path(run_dir)
    payload: list[dict[str, Any]] = []
    for sp in protocols:
        payload.append(
            {
                "rank": sp.rank,
                "architecture": sp.architecture.to_dict(),
                "score": sp.score,
                "robust_score": sp.robust_score,
                "selected_parameters": dict(sp.selected_parameters),
                "metrics": dict(sp.metrics),
                "constraint_passed": sp.constraint_passed,
                "constraint_failures": list(sp.constraint_failures),
            }
        )
    path = root / "selected_protocols.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def write_lock_file(
    run_dir: Path | str,
    scenario_config: CanonicalScenarioConfig | Mapping[str, Any],
    global_config: Any,
    protocol_config: DoubleOOSWFOProtocolConfig,
    selected_protocol: SelectedProtocol,
    *,
    refuse_overwrite: bool = True,
) -> None:
    """Write ``protocol_lock.yaml`` pinning hashes and the winning architecture."""

    root = Path(run_dir)
    lock_path = root / "protocol_lock.yaml"
    if lock_path.is_file() and refuse_overwrite:
        raise ResearchArtifactExistsError(str(lock_path))

    scenario_hash = stable_hash(
        scenario_config.to_dict()
        if isinstance(scenario_config, CanonicalScenarioConfig)
        else scenario_config
    )
    to_dict_fn = getattr(global_config, "to_dict", None)
    global_payload: Any = to_dict_fn() if callable(to_dict_fn) else global_config
    global_hash = stable_hash(global_payload)
    proto_plain = protocol_config_to_plain(protocol_config)
    protocol_hash = stable_hash(proto_plain)

    document = {
        "protocol_version": RESEARCH_PROTOCOL_ARTIFACT_VERSION,
        "final_unseen_mode": protocol_config.final_unseen_mode.value,
        "hashes": {
            "scenario_hash": scenario_hash,
            "global_config_hash": global_hash,
            "protocol_config_hash": protocol_hash,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_wfo_architecture": selected_protocol.architecture.to_dict(),
        "selected_parameters": dict(selected_protocol.selected_parameters),
    }
    lock_path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )


def write_unseen_results(run_dir: Path | str, unseen: UnseenValidationResult) -> None:
    """Persist unseen holdout returns and summary metrics."""

    root = Path(run_dir)
    unseen.returns.to_csv(root / "unseen_test_returns.csv", index_label="date")
    metrics_doc = {"metrics": dict(unseen.metrics), "mode": unseen.mode}
    (root / "unseen_test_metrics.yaml").write_text(
        yaml.safe_dump(metrics_doc, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
