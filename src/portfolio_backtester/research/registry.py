"""Persistent registry of research protocol runs and unseen completion status."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from portfolio_backtester.canonical_config import CanonicalScenarioConfig
from portfolio_backtester.research.artifacts import protocol_config_to_plain
from portfolio_backtester.research.hashing import stable_hash
from portfolio_backtester.research.protocol_config import DoubleOOSWFOProtocolConfig


class ResearchRegistryError(RuntimeError):
    """Raised when a research registry invariant is violated (e.g. duplicate unseen)."""


def unseen_period_plain(protocol_config: DoubleOOSWFOProtocolConfig) -> dict[str, Any]:
    """Serializable unseen holdout range for hashing."""

    ut = protocol_config.unseen_test_period
    return {
        "start": pd.Timestamp(ut.start).isoformat(),
        "end": pd.Timestamp(ut.end).isoformat(),
    }


def compute_registry_hashes(
    scenario_config: CanonicalScenarioConfig,
    protocol_config: DoubleOOSWFOProtocolConfig,
) -> tuple[str, str, str]:
    """Return ``(scenario_hash, protocol_config_hash, unseen_period_hash)``."""

    scenario_hash = stable_hash(scenario_config.to_dict())
    protocol_plain = protocol_config_to_plain(protocol_config)
    protocol_config_hash = stable_hash(protocol_plain)
    unseen_period_hash = stable_hash(unseen_period_plain(protocol_config))
    return scenario_hash, protocol_config_hash, unseen_period_hash


class ResearchRunRegistry:
    """YAML-backed append/update registry under ``<scenario>/research_protocol/registry.yaml``."""

    def __init__(self, registry_path: Path | str) -> None:
        self.path = Path(registry_path)

    def load_runs(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not raw:
            return []
        runs = raw.get("runs")
        if not isinstance(runs, list):
            return []
        return [dict(x) for x in runs if isinstance(x, Mapping)]

    def _save_runs(self, runs: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"runs": runs}
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.path)

    def record_lock(
        self,
        *,
        run_id: str,
        scenario_hash: str,
        protocol_config_hash: str,
        unseen_period_hash: str,
        lock_path: str,
        created_at: str | None = None,
    ) -> None:
        """Append a run row with ``unseen_completed`` false."""

        runs = self.load_runs()
        ts = created_at or datetime.now(timezone.utc).isoformat()
        runs.append(
            {
                "run_id": run_id,
                "scenario_hash": scenario_hash,
                "protocol_config_hash": protocol_config_hash,
                "unseen_period_hash": unseen_period_hash,
                "lock_path": lock_path,
                "unseen_completed": False,
                "created_at": ts,
            }
        )
        self._save_runs(runs)

    def mark_unseen_completed(self, run_id: str) -> None:
        runs = self.load_runs()
        changed = False
        for row in runs:
            if row.get("run_id") == run_id:
                row["unseen_completed"] = True
                changed = True
                break
        if not changed:
            msg = f"registry has no run_id {run_id!r} to mark unseen completed"
            raise ResearchRegistryError(msg)
        self._save_runs(runs)

    def find_completed_duplicate(
        self,
        scenario_hash: str,
        protocol_config_hash: str,
        unseen_period_hash: str,
    ) -> dict[str, Any] | None:
        for row in self.load_runs():
            if (
                row.get("scenario_hash") == scenario_hash
                and row.get("protocol_config_hash") == protocol_config_hash
                and row.get("unseen_period_hash") == unseen_period_hash
                and row.get("unseen_completed") is True
            ):
                return dict(row)
        return None

    def assert_no_completed_duplicate(
        self,
        scenario_hash: str,
        protocol_config_hash: str,
        unseen_period_hash: str,
        *,
        force_new_research_run: bool,
    ) -> None:
        if force_new_research_run:
            return
        dup = self.find_completed_duplicate(scenario_hash, protocol_config_hash, unseen_period_hash)
        if dup is None:
            return
        prior_id = dup.get("run_id", "?")
        msg = (
            "research registry already has unseen_completed for this "
            f"scenario_hash / protocol_config_hash / unseen_period_hash (run_id={prior_id}); "
            "use --force-new-research-run to start a new run directory or change inputs"
        )
        raise ResearchRegistryError(msg)
