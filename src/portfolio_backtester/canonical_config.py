from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence
from frozendict import frozendict


def freeze_config(data: Any) -> Any:
    """Recursively freeze dictionaries and lists into immutable types.

    Args:
        data: The data to freeze.

    Returns:
        The frozen data (frozendict for dicts, tuple for lists).
    """
    if isinstance(data, dict):
        return frozendict({k: freeze_config(v) for k, v in data.items()})
    elif isinstance(data, list):
        return tuple(freeze_config(v) for v in data)
    elif isinstance(data, frozendict):
        return data
    return data


@dataclass(frozen=True)
class CanonicalScenarioConfig:
    """Canonical, immutable representation of a backtesting scenario.

    This object serves as the single source of truth for all runtime components,
    ensuring consistent interpretation of scenario configuration across all
    execution paths (backtest, optimize, evaluation).
    """

    name: str
    strategy: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    benchmark_ticker: Optional[str] = None
    timing_config: Mapping[str, Any] = field(default_factory=frozendict)
    universe_definition: Mapping[str, Any] = field(default_factory=frozendict)
    position_sizer: Optional[str] = None
    optimization_metric: Optional[str] = None
    wfo_config: Mapping[str, Any] = field(default_factory=frozendict)
    optimizer_config: Mapping[str, Any] = field(default_factory=frozendict)
    strategy_params: Mapping[str, Any] = field(default_factory=frozendict)
    optimize: Optional[Sequence[Mapping[str, Any]]] = None
    extras: Mapping[str, Any] = field(default_factory=frozendict)

    def get(self, key: str, default: Any = None) -> Any:
        """Compatibility method for dictionary-like access."""
        if hasattr(self, key):
            val = getattr(self, key)
            if val is not None:
                return val
        return self.extras.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Compatibility method for dictionary-like access."""
        if not isinstance(key, str):
            # This can happen if dict(config) is called and it tries to iterate
            # or use non-string keys, though unlikely for a mapping-like object.
            # But dict(obj) on a non-iterable might cause issues.
            raise TypeError(f"attribute name must be string, not {type(key).__name__}")
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.extras:
            return self.extras[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Compatibility method for dictionary-like usage."""
        return hasattr(self, key) or key in self.extras

    def keys(self) -> Sequence[str]:
        """Get all available keys (known fields + extras)."""
        known_keys = [
            "name",
            "strategy",
            "start_date",
            "end_date",
            "benchmark_ticker",
            "timing_config",
            "universe_definition",
            "position_sizer",
            "optimization_metric",
            "wfo_config",
            "optimizer_config",
            "strategy_params",
            "optimize",
            "extras",
        ]
        return known_keys + list(self.extras.keys())

    def items(self):
        """Get all available items (known fields + extras)."""
        return [(k, self[k]) for k in self.keys()]

    def __iter__(self):
        """Iterate over all keys."""
        return iter(self.keys())

    def __post_init__(self) -> None:
        """Freeze nested structures to ensure full immutability."""
        # Note: frozen=True prevents direct assignment, but we can bypass it
        # using object.__setattr__ if we really needed to.
        # However, since we want to freeze the inputs *before* they are stored,
        # we should probably do that in a factory method or the normalizer.
        # dataclass(frozen=True) doesn't easily allow mutating fields in __post_init__.
        pass

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CanonicalScenarioConfig:
        """Create a CanonicalScenarioConfig from a dictionary, freezing it recursively.

        Args:
            data: Raw dictionary data.

        Returns:
            A frozen CanonicalScenarioConfig instance.
        """
        known_keys = {
            "name",
            "strategy",
            "start_date",
            "end_date",
            "benchmark_ticker",
            "timing_config",
            "universe_definition",
            "position_sizer",
            "optimization_metric",
            "wfo_config",
            "optimizer_config",
            "strategy_params",
            "optimize",
        }

        init_data = {}
        extras: dict[str, Any] = {}

        for k, v in data.items():
            frozen_v = freeze_config(v)
            if k in known_keys:
                init_data[k] = frozen_v
            elif k == "extras":
                # If 'extras' is already provided, merge it
                if isinstance(frozen_v, Mapping):
                    extras.update(frozen_v)
            else:
                extras[k] = frozen_v

        init_data["extras"] = frozendict(extras)
        return cls(**init_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for logging or serialization."""
        return {
            "name": self.name,
            "strategy": self.strategy,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "benchmark_ticker": self.benchmark_ticker,
            "timing_config": dict(self.timing_config),
            "universe_definition": dict(self.universe_definition),
            "position_sizer": self.position_sizer,
            "optimization_metric": self.optimization_metric,
            "wfo_config": dict(self.wfo_config),
            "optimizer_config": dict(self.optimizer_config),
            "strategy_params": dict(self.strategy_params),
            "optimize": [dict(x) for x in self.optimize] if self.optimize is not None else None,
            "extras": dict(self.extras),
        }
