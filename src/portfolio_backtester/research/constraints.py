"""Metric bound constraints for research protocol architecture selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


class ResearchConstraintError(RuntimeError):
    """Raised when no grid architecture satisfies configured metric constraints."""


@dataclass(frozen=True)
class MetricConstraint:
    """Single metric bound: canonical display key plus optional min/max inclusive."""

    display_key: str
    min_value: float | None
    max_value: float | None


class ConstraintEvaluator:
    """Evaluates frozen metric constraints against a metrics dict."""

    def __init__(self, rules: tuple[MetricConstraint, ...]) -> None:
        self._rules = rules

    def evaluate(self, metrics: Mapping[str, float]) -> tuple[bool, tuple[str, ...]]:
        """Return (all_passed, failure_messages)."""

        failures: list[str] = []
        for rule in self._rules:
            key = rule.display_key
            if key not in metrics:
                failures.append(f"{key}: missing metric in result")
                continue
            raw = metrics[key]
            if isinstance(raw, (float, int)):
                value = float(raw)
            else:
                failures.append(f"{key}: value is not numeric")
                continue
            if math.isnan(value):
                failures.append(f"{key}: value is NaN")
                continue
            if rule.min_value is not None and value < float(rule.min_value):
                failures.append(
                    f"{key}: {value} below minimum {rule.min_value}",
                )
                continue
            if rule.max_value is not None and value > float(rule.max_value):
                failures.append(
                    f"{key}: {value} above maximum {rule.max_value}",
                )
                continue
        return (len(failures) == 0, tuple(failures))


def evaluate_architecture_constraints(
    metrics: Mapping[str, float],
    rules: Sequence[MetricConstraint],
) -> tuple[bool, tuple[str, ...]]:
    """Evaluate ``rules`` against ``metrics`` (convenience wrapper)."""

    return ConstraintEvaluator(tuple(rules)).evaluate(metrics)
