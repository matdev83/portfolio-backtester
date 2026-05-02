"""Run-scoped cache for full strategy signal matrices produced by ``generate_signals``."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Optional

import pandas as pd

from ..canonical_config import freeze_config


def default_never_timed_out() -> bool:
    """Default timeout predicate used when no timeout is configured for backtests."""
    return False


def _json_default(obj: Any) -> str:
    return str(obj)


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=_json_default, separators=(",", ":"))


def signal_affecting_param_subset(
    strategy: Any, strategy_params: Mapping[str, Any]
) -> dict[str, Any]:
    """Return strategy_params entries that participate in the signal cache key.

    If the strategy class defines ``signal_affecting_parameter_names`` as a classmethod
    returning an optional frozenset of names, only those keys are used (when present in
    strategy_params). If it returns None or the method is absent, all keys are used.
    """
    raw = dict(strategy_params)
    cls = type(strategy)
    fn = getattr(cls, "signal_affecting_parameter_names", None)
    if fn is None:
        return raw
    if not callable(fn):
        return raw
    try:
        names = fn()
    except Exception:  # noqa: BLE001
        return raw
    if names is None:
        return raw
    out: dict[str, Any] = {}
    for k in names:
        if k in raw:
            out[k] = raw[k]
    return out


def strategy_allows_signal_matrix_cache(strategy: Any) -> bool:
    """Return True when the strategy class opts into signal matrix caching."""
    cls = type(strategy)
    if getattr(cls, "signal_matrix_cache_deterministic", False) is True:
        return True
    meth = getattr(cls, "is_signal_matrix_cache_deterministic", None)
    if callable(meth):
        try:
            return bool(meth())
        except Exception:  # noqa: BLE001
            return False
    return False


def index_fingerprint(idx: pd.Index) -> tuple[int, int, str]:
    """Fingerprint daily index shape and endpoints for cache keys."""
    n = len(idx)
    if n == 0:
        return 0, 0, ""
    first = idx[0]
    last = idx[-1]
    t0 = int(pd.Timestamp(first).value)
    t1 = int(pd.Timestamp(last).value)
    return n, t0, f"{t0}:{t1}"


def build_signal_cache_key_digest(payload: Mapping[str, Any]) -> str:
    """Build a SHA256 hex digest for a JSON-serializable payload."""
    frozen = freeze_config(dict(payload))
    blob = _stable_json(frozen)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_signal_matrix_cache_digest(
    *,
    strategy_module_qualname: str,
    universe_tickers: tuple[str, ...],
    benchmark_ticker: str,
    non_universe_tickers: tuple[str, ...],
    rebalance_dates_ns: tuple[int, ...],
    use_sparse_nan_for_inactive_rows: bool,
    timing_mode: str,
    timing_config: Any,
    scenario_bounds: Mapping[str, Any],
    strategy_params_slice: Mapping[str, Any],
    feature_flags: Mapping[str, Any],
    index_fp: tuple[int, int, str],
) -> str:
    """Compute a cache digest from all signal-generation-affecting inputs."""
    payload = {
        "strategy": strategy_module_qualname,
        "universe": universe_tickers,
        "benchmark": benchmark_ticker,
        "non_universe": non_universe_tickers,
        "rebalance_ns": rebalance_dates_ns,
        "sparse_nan_rows": use_sparse_nan_for_inactive_rows,
        "timing_mode": timing_mode,
        "timing_config": (
            freeze_config(dict(timing_config))
            if isinstance(timing_config, Mapping)
            else timing_config
        ),
        "scenario_bounds": freeze_config(dict(scenario_bounds)),
        "strategy_params": freeze_config(dict(strategy_params_slice)),
        "feature_flags": freeze_config(dict(feature_flags)),
        "index_fp": index_fp,
    }
    return build_signal_cache_key_digest(payload)


class SignalCache:
    """Bounded, run-scoped cache for signal DataFrames keyed by digest strings."""

    def __init__(self, max_entries: int = 512) -> None:
        self._max_entries = max_entries
        self._data: dict[str, pd.DataFrame] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key_digest: str) -> Optional[pd.DataFrame]:
        hit = self._data.get(key_digest)
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        return hit

    def put(self, key_digest: str, signals: pd.DataFrame) -> None:
        if len(self._data) >= self._max_entries and key_digest not in self._data:
            try:
                drop_key = next(iter(self._data))
            except StopIteration:
                drop_key = None
            if drop_key is not None:
                del self._data[drop_key]
        self._data[key_digest] = signals

    def stats(self) -> dict[str, int]:
        """Return hit/miss/size counters for profiling and diagnostics."""
        return {"hits": self.hits, "misses": self.misses, "size": len(self._data)}
