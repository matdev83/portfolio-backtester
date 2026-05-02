"""YAML-ready research protocol configuration (pure parsing + validation)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import pandas as pd

from .constraints import MetricConstraint
from .scoring import (
    ROBUST_COMPOSITE_METRIC_NAME,
    CompositeRankScoringConfig,
    ResearchScoringError,
    RobustSelectionConfig,
    canonical_metric_display_key,
    default_composite_rank_scoring,
)

_DEFAULT_HEATMAP_METRICS: tuple[str, ...] = ("score", "robust_score")

_ALLOWED_WALK_FORWARD_TYPES: frozenset[str] = frozenset({"rolling", "expanding"})


class ResearchProtocolConfigError(ValueError):
    """Invalid or unusable research protocol configuration."""


def normalize_heatmap_metric_token(raw: str) -> str:
    """Normalize a ``heatmap_metrics`` entry to a column name or reserved ``score`` key."""

    s = str(raw).strip()
    if not s:
        msg = "heatmap_metrics entries must be non-empty strings"
        raise ResearchProtocolConfigError(msg)
    low = s.lower()
    if low == "score":
        return "score"
    if low == "robust_score":
        return "robust_score"
    try:
        return canonical_metric_display_key(s)
    except ResearchScoringError as exc:
        msg = f"invalid heatmap_metrics entry: {raw!r}"
        raise ResearchProtocolConfigError(msg) from exc


class FinalUnseenMode(str, Enum):
    """Supported modes for final unseen evaluation."""

    REOPTIMIZE_WITH_LOCKED_ARCHITECTURE = "reoptimize_with_locked_architecture"
    FIXED_SELECTED_PARAMS = "fixed_selected_params"


_SUPPORTED_FINAL_UNSEEN_MODES: frozenset[str] = frozenset(m.value for m in FinalUnseenMode)

RESEARCH_PROTOCOL_ARTIFACT_VERSION: int = 1


@dataclass(frozen=True)
class DateRangeConfig:
    """Inclusive calendar range."""

    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class WFOGridConfig:
    """Walk-forward hyperparameter grid (months + type labels)."""

    train_window_months: tuple[int, ...]
    test_window_months: tuple[int, ...]
    wfo_step_months: tuple[int, ...]
    walk_forward_type: tuple[str, ...]


@dataclass(frozen=True)
class SelectionConfig:
    """Inner-loop metric selection."""

    top_n: int
    metric: str


@dataclass(frozen=True)
class ArchitectureLockConfig:
    """Controls architecture locking behavior for unseen stages."""

    enabled: bool
    refuse_overwrite: bool


@dataclass(frozen=True)
class ReportingConfig:
    """Reporting toggles."""

    enabled: bool
    generate_heatmaps: bool = False
    heatmap_metrics: tuple[str, ...] = _DEFAULT_HEATMAP_METRICS
    generate_html: bool = False


class CostSensitivityRunOn(str, Enum):
    """Supported targets for post-selection cost sensitivity sweeps."""

    UNSEEN = "unseen"


_SUPPORTED_COST_SENSITIVITY_RUN_ON: frozenset[str] = frozenset(
    m.value for m in CostSensitivityRunOn
)


@dataclass(frozen=True)
class CostSensitivityConfig:
    """Post-selection transaction cost stress test (does not affect ranking)."""

    enabled: bool
    slippage_bps_grid: tuple[float, ...]
    commission_multiplier_grid: tuple[float, ...]
    run_on: CostSensitivityRunOn


@dataclass(frozen=True)
class RandomWfoArchitectureBootstrapConfig:
    """Bootstrap against random draws of in-sample architecture scores."""

    enabled: bool


@dataclass(frozen=True)
class BlockShuffledReturnsBootstrapConfig:
    """Block-bootstrap unseen returns to assess path dependence (does not affect selection)."""

    enabled: bool
    block_size_days: int


@dataclass(frozen=True)
class BlockShuffledPositionsBootstrapConfig:
    """Block-bootstrap position paths from trade history (does not affect selection)."""

    enabled: bool
    block_size_days: int = 20


@dataclass(frozen=True)
class RandomStrategyParametersBootstrapConfig:
    """Bootstrap against random draws from a discrete strategy parameter space."""

    enabled: bool
    sample_size: int = 100


@dataclass(frozen=True)
class BootstrapConfig:
    """Post-validation significance/bootstrap options (does not affect selection)."""

    enabled: bool
    n_samples: int
    random_seed: int
    random_wfo_architecture: RandomWfoArchitectureBootstrapConfig
    block_shuffled_returns: BlockShuffledReturnsBootstrapConfig
    block_shuffled_positions: BlockShuffledPositionsBootstrapConfig
    random_strategy_parameters: RandomStrategyParametersBootstrapConfig


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime execution limits for research protocol runs."""

    max_grid_cells: int = 100
    fail_fast: bool = True


@dataclass(frozen=True)
class DoubleOOSWFOProtocolConfig:
    """Parsed double OOS walk-forward research protocol configuration."""

    enabled: bool
    global_train_period: DateRangeConfig
    unseen_test_period: DateRangeConfig
    wfo_window_grid: WFOGridConfig
    selection: SelectionConfig
    composite_scoring: CompositeRankScoringConfig | None
    constraints: tuple[MetricConstraint, ...]
    final_unseen_mode: FinalUnseenMode
    lock: ArchitectureLockConfig
    reporting: ReportingConfig
    robust_selection: RobustSelectionConfig
    cost_sensitivity: CostSensitivityConfig
    bootstrap: BootstrapConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def validate_enabled(self) -> None:
        """Ensure the protocol is enabled before execution paths run."""

        if not self.enabled:
            msg = "research protocol is disabled"
            raise ResearchProtocolConfigError(msg)

    def validate_for_mode(self, mode: str) -> None:
        """Validate configuration for a backtester run mode.

        Args:
            mode: CLI or engine mode string (reserved for future branching).
        """

        _ = mode
        self.research_validate()

    def research_validate(self) -> None:
        """Ensure the protocol is enabled and ranges/grid remain coherent."""

        self.validate_enabled()
        self._validate_periods_and_grid()

    def _validate_periods_and_grid(self) -> None:
        if self.global_train_period.start > self.global_train_period.end:
            msg = "global_train_period start after end"
            raise ResearchProtocolConfigError(msg)
        if self.unseen_test_period.start > self.unseen_test_period.end:
            msg = "unseen_test_period start after end"
            raise ResearchProtocolConfigError(msg)
        if _ranges_overlap_inclusive(
            self.global_train_period.start,
            self.global_train_period.end,
            self.unseen_test_period.start,
            self.unseen_test_period.end,
        ):
            msg = "global_train_period overlaps unseen_test_period"
            raise ResearchProtocolConfigError(msg)


def parse_double_oos_wfo_protocol(raw: Mapping[str, Any]) -> DoubleOOSWFOProtocolConfig:
    """Parse ``research_protocol`` for ``double_oos_wfo``."""

    if "research_protocol" not in raw:
        msg = "missing research_protocol section"
        raise ResearchProtocolConfigError(msg)
    inner_any = raw["research_protocol"]
    if not isinstance(inner_any, Mapping):
        msg = "research_protocol must be a mapping"
        raise ResearchProtocolConfigError(msg)
    inner: Mapping[str, Any] = inner_any

    proto_type = inner.get("type")
    if proto_type != "double_oos_wfo":
        msg = f"unsupported research protocol type: {proto_type!r}"
        raise ResearchProtocolConfigError(msg)

    enabled = bool(inner.get("enabled", True))

    global_train_period = _parse_period_block(inner, ("global_train_period", "global_train"))
    unseen_test_period = _parse_period_block(inner, ("unseen_test_period", "unseen_holdout"))

    wfo_any = _pick_first_mapping(inner, ("wfo_window_grid", "wfo_grid"))
    if wfo_any is None:
        msg = "missing wfo_window_grid (or legacy wfo_grid) block"
        raise ResearchProtocolConfigError(msg)
    wfo_window_grid = _parse_wfo_grid(wfo_any)

    selection_any = inner.get("selection")
    if not isinstance(selection_any, Mapping):
        msg = "selection must be a mapping"
        raise ResearchProtocolConfigError(msg)
    top_n_any = selection_any.get("top_n")
    if not isinstance(top_n_any, int) or isinstance(top_n_any, bool):
        msg = "selection.top_n must be an integer"
        raise ResearchProtocolConfigError(msg)
    if top_n_any <= 0:
        msg = "top_n must be positive"
        raise ResearchProtocolConfigError(msg)

    metric_any = selection_any.get("metric")
    if metric_any is None or str(metric_any).strip() == "":
        metric_str = ROBUST_COMPOSITE_METRIC_NAME
    else:
        metric_str = str(metric_any).strip()
        if metric_str.lower() == "robustcomposite":
            metric_str = ROBUST_COMPOSITE_METRIC_NAME

    scoring_raw = inner.get("scoring")
    parsed_scoring: CompositeRankScoringConfig | None = None
    if scoring_raw is not None:
        if not isinstance(scoring_raw, Mapping):
            msg = "scoring must be a mapping when provided"
            raise ResearchProtocolConfigError(msg)
        parsed_scoring = _parse_composite_scoring_config(scoring_raw)

    if metric_str == ROBUST_COMPOSITE_METRIC_NAME:
        composite_eff = (
            parsed_scoring if parsed_scoring is not None else default_composite_rank_scoring()
        )
    else:
        if parsed_scoring is not None:
            msg = (
                "research_protocol.scoring is only allowed when selection.metric is RobustComposite"
            )
            raise ResearchProtocolConfigError(msg)
        composite_eff = None

    selection = SelectionConfig(top_n=int(top_n_any), metric=metric_str)

    final_mode_raw = inner.get(
        "final_unseen_mode",
        FinalUnseenMode.REOPTIMIZE_WITH_LOCKED_ARCHITECTURE.value,
    )
    if str(final_mode_raw) not in _SUPPORTED_FINAL_UNSEEN_MODES:
        msg = f"unsupported final_unseen_mode: {final_mode_raw!r}"
        raise ResearchProtocolConfigError(msg)
    final_unseen_mode = FinalUnseenMode(str(final_mode_raw))

    lock_block = _pick_first_mapping(inner, ("lock", "architecture_lock"))
    lock = _parse_architecture_lock(lock_block)
    reporting = _parse_reporting(inner.get("reporting"))
    constraints = _parse_constraints(inner.get("constraints"))
    robust_selection = _parse_robust_selection(inner.get("robust_selection"))
    cost_sensitivity = _parse_cost_sensitivity(inner.get("cost_sensitivity"))
    bootstrap = _parse_bootstrap(inner.get("bootstrap"))
    execution = _parse_execution(inner.get("execution"))

    cfg = DoubleOOSWFOProtocolConfig(
        enabled=enabled,
        global_train_period=global_train_period,
        unseen_test_period=unseen_test_period,
        wfo_window_grid=wfo_window_grid,
        selection=selection,
        composite_scoring=composite_eff,
        constraints=constraints,
        final_unseen_mode=final_unseen_mode,
        lock=lock,
        reporting=reporting,
        robust_selection=robust_selection,
        cost_sensitivity=cost_sensitivity,
        bootstrap=bootstrap,
        execution=execution,
    )
    cfg._validate_periods_and_grid()
    return cfg


def _default_composite_direction(display_key: str) -> str:
    if display_key == "Turnover":
        return "lower"
    return "higher"


def _parse_composite_scoring_config(raw: Mapping[str, Any]) -> CompositeRankScoringConfig:
    typ = raw.get("type")
    if typ != "composite_rank":
        msg = f"unsupported scoring.type: {typ!r}"
        raise ResearchProtocolConfigError(msg)
    w_block = raw.get("weights")
    if not isinstance(w_block, Mapping) or len(w_block) == 0:
        msg = "scoring.weights must be a non-empty mapping"
        raise ResearchProtocolConfigError(msg)
    pairs: list[tuple[str, float]] = []
    for k, v in w_block.items():
        try:
            dk = canonical_metric_display_key(str(k))
        except ResearchScoringError as exc:
            raise ResearchProtocolConfigError(str(exc)) from exc
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            msg = f"invalid weight value for metric {dk!r}"
            raise ResearchProtocolConfigError(msg) from exc
        if fv <= 0:
            msg = "composite weights must be positive"
            raise ResearchProtocolConfigError(msg)
        pairs.append((dk, fv))
    total = sum(w for _, w in pairs)
    if total <= 0:
        msg = "composite weights must sum to a positive total"
        raise ResearchProtocolConfigError(msg)
    pairs_norm = tuple((k, w / total) for k, w in pairs)
    weight_keys = {k for k, _ in pairs_norm}
    merged: dict[str, str] = {dk: _default_composite_direction(dk) for dk in weight_keys}
    dirs_raw = raw.get("directions")
    if dirs_raw is not None:
        if not isinstance(dirs_raw, Mapping):
            msg = "scoring.directions must be a mapping when provided"
            raise ResearchProtocolConfigError(msg)
        for dk_key, dv in dirs_raw.items():
            try:
                dk2 = canonical_metric_display_key(str(dk_key))
            except ResearchScoringError as exc:
                raise ResearchProtocolConfigError(str(exc)) from exc
            if dk2 not in weight_keys:
                msg = f"scoring.directions references metric not in scoring.weights: {dk2!r}"
                raise ResearchProtocolConfigError(msg)
            dv_norm = str(dv).strip().lower()
            if dv_norm not in ("higher", "lower"):
                msg = f"unsupported scoring.direction for {dk2!r}: {dv!r}"
                raise ResearchProtocolConfigError(msg)
            merged[dk2] = dv_norm
    return CompositeRankScoringConfig(weights=pairs_norm, directions=dict(merged))


def _pick_first_mapping(
    inner: Mapping[str, Any], keys: tuple[str, ...]
) -> Mapping[str, Any] | None:
    for key in keys:
        block_any = inner.get(key)
        if isinstance(block_any, Mapping):
            return block_any
    return None


def _parse_period_block(inner: Mapping[str, Any], keys: tuple[str, ...]) -> DateRangeConfig:
    block_any = _pick_first_mapping(inner, keys)
    if block_any is None:
        msg = f"missing or invalid period block (tried {', '.join(keys)})"
        raise ResearchProtocolConfigError(msg)
    return _parse_range_mapping(block_any)


def _parse_range_mapping(block: Mapping[str, Any]) -> DateRangeConfig:
    start_raw = block.get("start_date", block.get("start"))
    end_raw = block.get("end_date", block.get("end"))
    if start_raw is None or end_raw is None:
        msg = "period must specify start_date/end_date (or legacy start/end)"
        raise ResearchProtocolConfigError(msg)
    try:
        start = pd.Timestamp(start_raw)
        end = pd.Timestamp(end_raw)
    except Exception as exc:  # noqa: BLE001 - normalize parse failures
        msg = "invalid dates in period block"
        raise ResearchProtocolConfigError(msg) from exc
    return DateRangeConfig(start=start, end=end)


def _positive_int_tuple(values: Any, label: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        msg = f"{label} must be a non-empty sequence"
        raise ResearchProtocolConfigError(msg)
    out: list[int] = []
    for i, v in enumerate(values):
        if not isinstance(v, int) or isinstance(v, bool):
            msg = f"{label}[{i}] must be an integer"
            raise ResearchProtocolConfigError(msg)
        if v <= 0:
            msg = f"{label} values must be positive integers"
            raise ResearchProtocolConfigError(msg)
        out.append(v)
    return tuple(out)


def _coerce_str_sequence(raw: Any, label: str) -> tuple[str, ...]:
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            msg = f"{label} must be a non-empty sequence"
            raise ResearchProtocolConfigError(msg)
        items = list(raw)
    else:
        items = [raw]
    out: list[str] = []
    for i, t in enumerate(items):
        ts = str(t)
        if ts not in _ALLOWED_WALK_FORWARD_TYPES:
            msg = f"unknown {label} entry: {t!r} at index {i}"
            raise ResearchProtocolConfigError(msg)
        out.append(ts)
    return tuple(out)


def _parse_wfo_grid(grid: Mapping[str, Any]) -> WFOGridConfig:
    raw_train = grid.get("train_window_months", grid.get("train_months"))
    raw_test = grid.get("test_window_months", grid.get("test_months"))
    raw_step = grid.get("wfo_step_months", grid.get("step_months"))
    raw_type = grid.get("walk_forward_type", grid.get("walk_forward_types"))

    train_window_months = _positive_int_tuple(raw_train, "train_window_months")
    test_window_months = _positive_int_tuple(raw_test, "test_window_months")
    wfo_step_months = _positive_int_tuple(raw_step, "wfo_step_months")
    walk_forward_type = _coerce_str_sequence(raw_type, "walk_forward_type")

    return WFOGridConfig(
        train_window_months=train_window_months,
        test_window_months=test_window_months,
        wfo_step_months=wfo_step_months,
        walk_forward_type=walk_forward_type,
    )


def _parse_architecture_lock(block_any: Any) -> ArchitectureLockConfig:
    if block_any is None:
        return ArchitectureLockConfig(enabled=True, refuse_overwrite=True)
    if not isinstance(block_any, Mapping):
        msg = "lock must be a mapping when provided"
        raise ResearchProtocolConfigError(msg)
    block: Mapping[str, Any] = block_any
    enabled = bool(block.get("enabled", True))
    refuse_overwrite = bool(block.get("refuse_overwrite", True))
    return ArchitectureLockConfig(enabled=enabled, refuse_overwrite=refuse_overwrite)


def _parse_constraints(raw: Any) -> tuple[MetricConstraint, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        msg = "constraints must be a list when provided"
        raise ResearchProtocolConfigError(msg)
    out: list[MetricConstraint] = []
    for i, item_any in enumerate(raw):
        if not isinstance(item_any, Mapping):
            msg = f"constraints[{i}] must be a mapping"
            raise ResearchProtocolConfigError(msg)
        m: Mapping[str, Any] = item_any
        metric_raw = m.get("metric")
        if metric_raw is None or str(metric_raw).strip() == "":
            msg = f"constraints[{i}].metric must be non-empty"
            raise ResearchProtocolConfigError(msg)
        try:
            display = canonical_metric_display_key(str(metric_raw))
        except ResearchScoringError as exc:
            raise ResearchProtocolConfigError(str(exc)) from exc
        min_raw = m.get("min_value")
        max_raw = m.get("max_value")
        min_v = None if min_raw is None else float(min_raw)
        max_v = None if max_raw is None else float(max_raw)
        if min_v is None and max_v is None:
            msg = f"constraints[{i}] requires at least one of min_value, max_value"
            raise ResearchProtocolConfigError(msg)
        if min_v is not None and max_v is not None and min_v > max_v:
            msg = f"constraints[{i}]: min_value must be <= max_value"
            raise ResearchProtocolConfigError(msg)
        out.append(MetricConstraint(display_key=display, min_value=min_v, max_value=max_v))
    return tuple(out)


def _default_robust_selection_weights() -> tuple[float, float, float]:
    return (0.5, 0.3, 0.2)


def _parse_robust_selection(raw: Any) -> RobustSelectionConfig:
    if raw is None:
        cell_w, med_w, min_w = _default_robust_selection_weights()
        return RobustSelectionConfig(
            enabled=False,
            cell_weight=cell_w,
            neighbor_median_weight=med_w,
            neighbor_min_weight=min_w,
        )
    if not isinstance(raw, Mapping):
        msg = "robust_selection must be a mapping when provided"
        raise ResearchProtocolConfigError(msg)
    block: Mapping[str, Any] = raw
    enabled = bool(block.get("enabled", False))
    w_any = block.get("weights")
    if w_any is None:
        cell_w, med_w, min_w = _default_robust_selection_weights()
    else:
        if not isinstance(w_any, Mapping):
            msg = "robust_selection.weights must be a mapping when provided"
            raise ResearchProtocolConfigError(msg)
        wm: Mapping[str, Any] = w_any
        for key in ("cell", "neighbor_median", "neighbor_min"):
            if key not in wm:
                msg = f"robust_selection.weights missing {key!r}"
                raise ResearchProtocolConfigError(msg)
        try:
            cell_w = float(wm["cell"])
            med_w = float(wm["neighbor_median"])
            min_w = float(wm["neighbor_min"])
        except (TypeError, ValueError) as exc:
            msg = "robust_selection weights must be numeric"
            raise ResearchProtocolConfigError(msg) from exc
    for label, val in (
        ("cell", cell_w),
        ("neighbor_median", med_w),
        ("neighbor_min", min_w),
    ):
        if val <= 0 or math.isnan(val):
            msg = f"robust_selection weight {label!r} must be positive"
            raise ResearchProtocolConfigError(msg)
    total = cell_w + med_w + min_w
    if total <= 0:
        msg = "robust_selection weights must sum to a positive total"
        raise ResearchProtocolConfigError(msg)
    return RobustSelectionConfig(
        enabled=enabled,
        cell_weight=cell_w,
        neighbor_median_weight=med_w,
        neighbor_min_weight=min_w,
    )


def _parse_cost_sensitivity(raw: Any) -> CostSensitivityConfig:
    if raw is None:
        return CostSensitivityConfig(
            enabled=False,
            slippage_bps_grid=(),
            commission_multiplier_grid=(1.0,),
            run_on=CostSensitivityRunOn.UNSEEN,
        )
    if not isinstance(raw, Mapping):
        msg = "cost_sensitivity must be a mapping when provided"
        raise ResearchProtocolConfigError(msg)
    block: Mapping[str, Any] = raw
    enabled = bool(block.get("enabled", False))
    run_raw = block.get("run_on", CostSensitivityRunOn.UNSEEN.value)
    run_s = str(run_raw).strip().lower()
    if run_s not in _SUPPORTED_COST_SENSITIVITY_RUN_ON:
        msg = f"unsupported cost_sensitivity.run_on: {run_raw!r} (only 'unseen' is supported)"
        raise ResearchProtocolConfigError(msg)
    run_on = CostSensitivityRunOn(run_s)
    if not enabled:
        return CostSensitivityConfig(
            enabled=False,
            slippage_bps_grid=(),
            commission_multiplier_grid=(1.0,),
            run_on=run_on,
        )
    slip_any = block.get("slippage_bps_grid")
    mult_any = block.get("commission_multiplier_grid")
    if not isinstance(slip_any, (list, tuple)) or len(slip_any) == 0:
        msg = "cost_sensitivity.slippage_bps_grid must be a non-empty list when enabled"
        raise ResearchProtocolConfigError(msg)
    if not isinstance(mult_any, (list, tuple)) or len(mult_any) == 0:
        msg = "cost_sensitivity.commission_multiplier_grid must be a non-empty list when enabled"
        raise ResearchProtocolConfigError(msg)
    slip_out: list[float] = []
    for i, v in enumerate(slip_any):
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            msg = f"cost_sensitivity.slippage_bps_grid[{i}] must be numeric"
            raise ResearchProtocolConfigError(msg) from exc
        if fv < 0 or math.isnan(fv):
            msg = f"cost_sensitivity.slippage_bps_grid[{i}] must be non-negative"
            raise ResearchProtocolConfigError(msg)
        slip_out.append(fv)
    mult_out: list[float] = []
    for i, v in enumerate(mult_any):
        try:
            mv = float(v)
        except (TypeError, ValueError) as exc:
            msg = f"cost_sensitivity.commission_multiplier_grid[{i}] must be numeric"
            raise ResearchProtocolConfigError(msg) from exc
        if mv <= 0 or math.isnan(mv):
            msg = f"cost_sensitivity.commission_multiplier_grid[{i}] must be positive"
            raise ResearchProtocolConfigError(msg)
        mult_out.append(mv)
    return CostSensitivityConfig(
        enabled=True,
        slippage_bps_grid=tuple(slip_out),
        commission_multiplier_grid=tuple(mult_out),
        run_on=run_on,
    )


def _parse_bootstrap(raw: Any) -> BootstrapConfig:
    if raw is None:
        return BootstrapConfig(
            enabled=False,
            n_samples=200,
            random_seed=42,
            random_wfo_architecture=RandomWfoArchitectureBootstrapConfig(enabled=False),
            block_shuffled_returns=BlockShuffledReturnsBootstrapConfig(
                enabled=False,
                block_size_days=20,
            ),
            block_shuffled_positions=BlockShuffledPositionsBootstrapConfig(
                enabled=False,
                block_size_days=20,
            ),
            random_strategy_parameters=RandomStrategyParametersBootstrapConfig(
                enabled=False,
                sample_size=100,
            ),
        )
    if not isinstance(raw, Mapping):
        msg = "bootstrap must be a mapping when provided"
        raise ResearchProtocolConfigError(msg)
    block: Mapping[str, Any] = raw
    enabled = bool(block.get("enabled", False))
    n_samples_raw = block.get("n_samples", 200)
    if not isinstance(n_samples_raw, int) or isinstance(n_samples_raw, bool):
        msg = "bootstrap.n_samples must be an integer"
        raise ResearchProtocolConfigError(msg)
    if n_samples_raw <= 0:
        msg = "bootstrap.n_samples must be positive"
        raise ResearchProtocolConfigError(msg)
    seed_raw = block.get("random_seed", 42)
    if not isinstance(seed_raw, int) or isinstance(seed_raw, bool):
        msg = "bootstrap.random_seed must be an integer"
        raise ResearchProtocolConfigError(msg)
    rw_any = block.get("random_wfo_architecture")
    if rw_any is None:
        random_wfo = RandomWfoArchitectureBootstrapConfig(enabled=False)
    else:
        if not isinstance(rw_any, Mapping):
            msg = "bootstrap.random_wfo_architecture must be a mapping when provided"
            raise ResearchProtocolConfigError(msg)
        random_wfo = RandomWfoArchitectureBootstrapConfig(
            enabled=bool(rw_any.get("enabled", False)),
        )
    br_any = block.get("block_shuffled_returns")
    if br_any is None:
        block_shuf = BlockShuffledReturnsBootstrapConfig(
            enabled=False,
            block_size_days=20,
        )
    else:
        if not isinstance(br_any, Mapping):
            msg = "bootstrap.block_shuffled_returns must be a mapping when provided"
            raise ResearchProtocolConfigError(msg)
        bs_en = bool(br_any.get("enabled", False))
        bs_days_raw = br_any.get("block_size_days", 20)
        if not isinstance(bs_days_raw, int) or isinstance(bs_days_raw, bool):
            msg = "bootstrap.block_shuffled_returns.block_size_days must be an integer"
            raise ResearchProtocolConfigError(msg)
        if bs_en and bs_days_raw <= 0:
            msg = "bootstrap.block_shuffled_returns.block_size_days must be positive when enabled"
            raise ResearchProtocolConfigError(msg)
        block_shuf = BlockShuffledReturnsBootstrapConfig(
            enabled=bs_en,
            block_size_days=int(bs_days_raw),
        )
    bpos_any = block.get("block_shuffled_positions")
    if bpos_any is None:
        block_pos = BlockShuffledPositionsBootstrapConfig(
            enabled=False,
            block_size_days=20,
        )
    else:
        if not isinstance(bpos_any, Mapping):
            msg = "bootstrap.block_shuffled_positions must be a mapping when provided"
            raise ResearchProtocolConfigError(msg)
        pos_en = bool(bpos_any.get("enabled", False))
        pos_days_raw = bpos_any.get("block_size_days", 20)
        if not isinstance(pos_days_raw, int) or isinstance(pos_days_raw, bool):
            msg = "bootstrap.block_shuffled_positions.block_size_days must be an integer"
            raise ResearchProtocolConfigError(msg)
        if pos_en and pos_days_raw <= 0:
            msg = "bootstrap.block_shuffled_positions.block_size_days must be positive when enabled"
            raise ResearchProtocolConfigError(msg)
        block_pos = BlockShuffledPositionsBootstrapConfig(
            enabled=pos_en,
            block_size_days=int(pos_days_raw),
        )
    rsp_any = block.get("random_strategy_parameters")
    if rsp_any is None:
        random_strategy = RandomStrategyParametersBootstrapConfig(enabled=False, sample_size=100)
    else:
        if not isinstance(rsp_any, Mapping):
            msg = "bootstrap.random_strategy_parameters must be a mapping when provided"
            raise ResearchProtocolConfigError(msg)
        rsp_en = bool(rsp_any.get("enabled", False))
        sz_raw = rsp_any.get("sample_size", 100)
        if not isinstance(sz_raw, int) or isinstance(sz_raw, bool):
            msg = "bootstrap.random_strategy_parameters.sample_size must be an integer"
            raise ResearchProtocolConfigError(msg)
        if sz_raw <= 0:
            msg = "bootstrap.random_strategy_parameters.sample_size must be positive"
            raise ResearchProtocolConfigError(msg)
        random_strategy = RandomStrategyParametersBootstrapConfig(
            enabled=rsp_en,
            sample_size=int(sz_raw),
        )
    return BootstrapConfig(
        enabled=enabled,
        n_samples=int(n_samples_raw),
        random_seed=int(seed_raw),
        random_wfo_architecture=random_wfo,
        block_shuffled_returns=block_shuf,
        block_shuffled_positions=block_pos,
        random_strategy_parameters=random_strategy,
    )


def default_bootstrap_config() -> BootstrapConfig:
    """Return defaults used when ``research_protocol.bootstrap`` is omitted."""

    return _parse_bootstrap(None)


def _parse_execution(raw: Any) -> ExecutionConfig:
    if raw is None:
        return ExecutionConfig()
    if not isinstance(raw, Mapping):
        msg = "execution must be a mapping when provided"
        raise ResearchProtocolConfigError(msg)
    block: Mapping[str, Any] = raw
    max_raw = block.get("max_grid_cells", 100)
    if not isinstance(max_raw, int) or isinstance(max_raw, bool):
        msg = "execution.max_grid_cells must be an integer"
        raise ResearchProtocolConfigError(msg)
    if max_raw <= 0:
        msg = "execution.max_grid_cells must be positive"
        raise ResearchProtocolConfigError(msg)
    fail_fast = bool(block.get("fail_fast", True))
    return ExecutionConfig(max_grid_cells=int(max_raw), fail_fast=fail_fast)


def _parse_reporting(block_any: Any) -> ReportingConfig:
    if block_any is None:
        return ReportingConfig(enabled=True)
    if not isinstance(block_any, Mapping):
        msg = "reporting must be a mapping when provided"
        raise ResearchProtocolConfigError(msg)
    block: Mapping[str, Any] = block_any
    enabled = bool(block.get("enabled", True))
    generate_heatmaps = bool(block.get("generate_heatmaps", False))
    generate_html = bool(block.get("generate_html", False))
    hm_raw = block.get("heatmap_metrics")
    if hm_raw is None:
        heatmap_metrics = _DEFAULT_HEATMAP_METRICS
    else:
        if not isinstance(hm_raw, (list, tuple)) or len(hm_raw) == 0:
            msg = "reporting.heatmap_metrics must be a non-empty list when provided"
            raise ResearchProtocolConfigError(msg)
        heatmap_metrics = tuple(normalize_heatmap_metric_token(str(x)) for x in hm_raw)
    return ReportingConfig(
        enabled=enabled,
        generate_heatmaps=generate_heatmaps,
        heatmap_metrics=heatmap_metrics,
        generate_html=generate_html,
    )


def _ranges_overlap_inclusive(
    a_start: pd.Timestamp,
    a_end: pd.Timestamp,
    b_start: pd.Timestamp,
    b_end: pd.Timestamp,
) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)
