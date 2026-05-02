"""Cross-validation averaging for temporal blocked folds on architecture-grid results."""

from __future__ import annotations

from numbers import Real
from statistics import mean
from typing import Any, Sequence

from .results import WFOArchitectureResult


def _avg_best_parameters(rows: Sequence[WFOArchitectureResult]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = sorted(set.intersection(*[set(r.best_parameters.keys()) for r in rows]))
    merged: dict[str, Any] = {}
    for k in keys:
        vals = [r.best_parameters[k] for r in rows]
        if all(isinstance(v, Real) and not isinstance(v, bool) for v in vals):
            merged[k] = float(mean(float(v) for v in vals))
            continue
        merged[k] = vals[0]
    return merged


def aggregate_blocked_fold_architecture_rows(
    per_fold_rows: Sequence[Sequence[WFOArchitectureResult]],
) -> tuple[WFOArchitectureResult, ...]:
    """Average metrics and bookkeeping across identically ordered fold result rows.

    Rows align strictly by index across folds. Architecture keys must match for each slot.

    If any fold violates constraints for a slot, aggregated ``constraint_passed`` is ``False``.
    Failed slots aggregate metrics and numeric ``score`` via fold-wise means across keys present
    in every fold's metrics map; aggregated rows clear ``stitched_returns``.
    Passed slots average metric intersection maps and defer scoring to callers.

    Args:
        per_fold_rows: Outer fold axis, inner per-architecture results in grid expansion order.

    Returns:
        One aggregated :class:`WFOArchitectureResult` per architecture index.
    """

    if len(per_fold_rows) == 0:
        return ()
    n_fold = len(per_fold_rows)
    base = per_fold_rows[0]
    n_arch = len(base)
    for fi, fr in enumerate(per_fold_rows[1:], start=1):
        if len(fr) != n_arch:
            msg = (
                f"cross-validation fold {fi} has {len(fr)} architectures; "
                f"expected {n_arch} to match fold 0"
            )
            raise ValueError(msg)
    merged: list[WFOArchitectureResult] = []
    for i in range(n_arch):
        rows = [tuple(per_fold_rows[f])[i] for f in range(n_fold)]
        arch0 = rows[0].architecture
        if any(r.architecture != arch0 for r in rows[1:]):
            msg = "architecture mismatch inside cross-validation fold aggregation slot"
            raise ValueError(msg)
        passed = all(r.constraint_passed for r in rows)
        failures = tuple(sorted({msg for r in rows for msg in r.constraint_failures}))
        metrics_keys_sets = [set(r.metrics.keys()) for r in rows]
        intersect_keys = sorted(set.intersection(*metrics_keys_sets))
        avg_metrics = {mk: float(mean(float(r.metrics[mk]) for r in rows)) for mk in intersect_keys}
        if not passed:
            avg_score = float(mean(float(r.score) for r in rows))
            merged.append(
                WFOArchitectureResult(
                    architecture=arch0,
                    metrics=avg_metrics,
                    score=avg_score,
                    robust_score=None,
                    best_parameters=_avg_best_parameters(tuple(rows)),
                    n_evaluations=int(mean(r.n_evaluations for r in rows)),
                    stitched_returns=None,
                    constraint_passed=False,
                    constraint_failures=failures,
                )
            )
            continue
        merged.append(
            WFOArchitectureResult(
                architecture=arch0,
                metrics=avg_metrics,
                score=0.0,
                robust_score=None,
                best_parameters=dict(rows[-1].best_parameters),
                n_evaluations=int(round(mean(r.n_evaluations for r in rows))),
                stitched_returns=None,
                constraint_passed=True,
                constraint_failures=(),
            )
        )
    return tuple(merged)
