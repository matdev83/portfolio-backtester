"""Structured research protocol result types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

_ALLOWED_WALK_FORWARD_TYPES: frozenset[str] = frozenset({"rolling", "expanding"})


@dataclass(frozen=True, order=True)
class WFOArchitecture:
    """Serializable walk-forward grid tuple used as an architecture key."""

    train_window_months: int
    test_window_months: int
    wfo_step_months: int
    walk_forward_type: str

    def __post_init__(self) -> None:
        if self.walk_forward_type not in _ALLOWED_WALK_FORWARD_TYPES:
            raise ValueError(f"unsupported walk_forward_type: {self.walk_forward_type!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain mapping."""

        return {
            "train_window_months": self.train_window_months,
            "test_window_months": self.test_window_months,
            "wfo_step_months": self.wfo_step_months,
            "walk_forward_type": self.walk_forward_type,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> WFOArchitecture:
        """Deserialize from a mapping."""

        tw = data.get("train_window_months", data.get("train_months"))
        te = data.get("test_window_months", data.get("test_months"))
        st = data.get("wfo_step_months", data.get("step_months"))
        wft = data.get("walk_forward_type")
        if tw is None or te is None or st is None or wft is None:
            msg = "WFOArchitecture.from_dict requires window month fields and walk_forward_type"
            raise ValueError(msg)
        return WFOArchitecture(
            train_window_months=int(tw),
            test_window_months=int(te),
            wfo_step_months=int(st),
            walk_forward_type=str(wft),
        )


@dataclass(frozen=True)
class WFOArchitectureResult:
    """In-sample outcome for one architecture."""

    architecture: WFOArchitecture
    metrics: Mapping[str, float]
    score: float
    robust_score: float | None
    best_parameters: Mapping[str, Any]
    n_evaluations: int
    stitched_returns: pd.Series | None = None
    constraint_passed: bool = True
    constraint_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectedProtocol:
    """Ranked architecture bundle after inner-loop scoring."""

    rank: int
    architecture: WFOArchitecture
    metrics: Mapping[str, float]
    score: float
    robust_score: float | None
    selected_parameters: Mapping[str, Any]
    constraint_passed: bool = True
    constraint_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnseenValidationResult:
    """Held-out unseen metrics for the chosen protocol."""

    selected_protocol: SelectedProtocol
    metrics: Mapping[str, float]
    returns: pd.Series
    mode: str
    trade_history: pd.DataFrame | None = None


@dataclass(frozen=True)
class ResearchProtocolResult:
    """Top-level aggregate returned by a research protocol run."""

    scenario_name: str
    grid_results: Sequence[WFOArchitectureResult]
    selected_protocols: tuple[SelectedProtocol, ...]
    unseen_result: UnseenValidationResult | None
    artifact_dir: Path
