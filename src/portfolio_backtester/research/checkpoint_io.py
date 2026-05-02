"""Filesystem checkpoints for interrupted research-protocol grid executions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .hashing import stable_hash
from .results import WFOArchitecture

ARCH_CHECKPOINT_SUBDIR_NAME = "grid_architecture_snapshots"


def checkpoint_key_for_architecture(architecture: WFOArchitecture) -> str:
    """Return the stable basename (without suffix) used for snapshots."""

    return stable_hash(architecture.to_dict())


def write_grid_cell_checkpoint(
    checkpoints_root: Path,
    *,
    architecture: WFOArchitecture,
    checkpoint_body: Mapping[str, Any],
) -> Path:
    """Write an atomic-ish snapshot YAML for ``architecture``.

    Uses a deterministic filename keyed by architecture hash under ``checkpoints_root``.

    Args:
        checkpoints_root: Parent directory holding per-cell checkpoints.
        architecture: Architecture key for deterministic naming.
        checkpoint_body: Serialized cell fields (excluding ``architecture`` duplication).

    Returns:
        Absolute path written.
    """

    checkpoints_root.mkdir(parents=True, exist_ok=True)
    ck_name = checkpoint_key_for_architecture(architecture) + ".yaml"
    out = checkpoints_root / ck_name
    tmp = checkpoints_root / (".tmp_" + ck_name)
    merged: dict[str, Any] = {
        "checkpoint_version": 1,
        "architecture": architecture.to_dict(),
        "metrics": dict(checkpoint_body.get("metrics") or {}),
        "score": float(checkpoint_body["score"]),
        "robust_score": checkpoint_body.get("robust_score"),
        "best_parameters": dict(checkpoint_body.get("best_parameters") or {}),
        "n_evaluations": int(checkpoint_body["n_evaluations"]),
        "constraint_passed": bool(checkpoint_body["constraint_passed"]),
        "constraint_failures": list(checkpoint_body.get("constraint_failures") or ()),
    }
    tmp.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=False), encoding="utf-8")
    tmp.replace(out)
    return out


def read_grid_cell_checkpoint(path: Path | str) -> dict[str, Any]:
    """Load a snapshot mapping from disk."""

    p = Path(path)
    loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        msg = f"checkpoint {p} must deserialize to a mapping"
        raise ValueError(msg)
    return dict(loaded)


def load_checkpoint_snapshots_map(checkpoints_root: Path | str) -> dict[str, Mapping[str, Any]]:
    """Load all ``*.yaml`` snapshots keyed by deterministic architecture hash basename."""

    root = Path(checkpoints_root)
    if not root.is_dir():
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for p in sorted(root.glob("*.yaml")):
        if p.name.startswith(".tmp_"):
            continue
        ck = read_grid_cell_checkpoint(p)
        arch_any = ck.get("architecture")
        if isinstance(arch_any, Mapping):
            key = stable_hash(dict(arch_any))
            out[key] = ck
    return out
