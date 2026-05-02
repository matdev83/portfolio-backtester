"""Execution manifest helpers for deterministic grid resume checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .hashing import stable_hash
from .protocol_config import DateRangeConfig, ResearchProtocolConfigError
from .results import WFOArchitecture

MANIFEST_VERSION: int = 1
MANIFEST_FILENAME: str = "research_execution_manifest.yaml"

RESUME_SCENARIO_HASH_MISMATCH_MESSAGE: str = "resume scenario_hash mismatch"
RESUME_PROTOCOL_CONFIG_HASH_MISMATCH_MESSAGE: str = "resume protocol_config_hash mismatch"
RESUME_ARCHITECTURE_ORDER_MISMATCH_MESSAGE: str = "resume architecture grid mismatch"


def architecture_order_digest(architectures: Sequence[WFOArchitecture]) -> str:
    """Return a digest of the expanded architecture grid order."""

    return stable_hash([a.to_dict() for a in architectures])


def split_global_train_blocked_folds(
    period: DateRangeConfig, n_folds: int
) -> tuple[DateRangeConfig, ...]:
    """Split ``period`` into ``n_folds`` consecutive blocked business-day folds.

    Folds are disjoint on calendar business days spanning ``period`` inclusively.

    Raises:
        ResearchProtocolConfigError: If ``period`` lacks enough business observations.
    """

    if n_folds < 2:
        msg = "n_folds must be >= 2"
        raise ResearchProtocolConfigError(msg)
    start = pd.Timestamp(period.start)
    end = pd.Timestamp(period.end)
    if start > end:
        msg = "global_train_period start after end"
        raise ResearchProtocolConfigError(msg)
    bidx = pd.bdate_range(start, end)
    if len(bidx) == 0:
        msg = "global_train_period has no business days to split for cross-validation"
        raise ResearchProtocolConfigError(msg)
    if len(bidx) < n_folds:
        msg = (
            f"cross_validation.n_folds={n_folds} exceeds usable business observations "
            f"({len(bidx)}) inside global_train_period"
        )
        raise ResearchProtocolConfigError(msg)
    splits = np.array_split(bidx.astype("datetime64[ns]").to_numpy(), n_folds)
    folds: list[DateRangeConfig] = []
    for ch in splits:
        if len(ch) == 0:
            msg = "cross-validation produced an empty temporal fold split"
            raise ResearchProtocolConfigError(msg)
        folds.append(DateRangeConfig(start=pd.Timestamp(ch[0]), end=pd.Timestamp(ch[-1])))
    return tuple(folds)


def write_execution_manifest(
    run_dir: Path | str,
    *,
    scenario_hash: str,
    protocol_config_hash: str,
    architectures: Sequence[WFOArchitecture],
) -> Path:
    """Write ``research_execution_manifest.yaml`` under ``run_dir``."""

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    arch_plain = [a.to_dict() for a in architectures]
    payload: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "scenario_hash": scenario_hash,
        "protocol_config_hash": protocol_config_hash,
        "architecture_order_hash": architecture_order_digest(architectures),
        "architectures": arch_plain,
    }
    path = root / MANIFEST_FILENAME
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return path


def load_execution_manifest_or_raise(run_dir: Path | str) -> dict[str, Any]:
    """Load execution manifest or raise :class:`ResearchProtocolConfigError`."""

    root = Path(run_dir)
    path = root / MANIFEST_FILENAME
    if not path.is_file():
        msg = f"missing research execution manifest at {path}"
        raise ResearchProtocolConfigError(msg)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        msg = "research execution manifest must be a mapping"
        raise ResearchProtocolConfigError(msg)
    return dict(raw)


def assert_resume_manifest_matches(
    manifest: Mapping[str, Any],
    *,
    scenario_hash: str,
    protocol_config_hash: str,
    architectures: Sequence[WFOArchitecture],
) -> None:
    """Validate resume manifest fingerprints against current run inputs."""

    if str(manifest.get("scenario_hash", "")) != scenario_hash:
        raise ResearchProtocolConfigError(RESUME_SCENARIO_HASH_MISMATCH_MESSAGE)
    if str(manifest.get("protocol_config_hash", "")) != protocol_config_hash:
        raise ResearchProtocolConfigError(RESUME_PROTOCOL_CONFIG_HASH_MISMATCH_MESSAGE)
    want = architecture_order_digest(architectures)
    if str(manifest.get("architecture_order_hash", "")) != want:
        raise ResearchProtocolConfigError(RESUME_ARCHITECTURE_ORDER_MISMATCH_MESSAGE)
