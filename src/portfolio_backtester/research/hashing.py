"""Stable hashing for research protocol artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def stable_hash(obj: Any) -> str:
    """Return a deterministic SHA-256 hex digest for nested runtime objects.

    Dict key order is ignored. Supported leaves include primitives, pandas
    timestamps, paths, mappings, sequences, dataclass instances, objects
    exposing ``to_dict()``, and nested combinations thereof.

    Args:
        obj: Value to hash.

    Returns:
        Lowercase hex digest string.

    Raises:
        TypeError: When ``obj`` cannot be normalized into a JSON-compatible tree.
    """

    normalized = _normalize_for_hash(obj)
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_for_hash(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, pd.Timestamp):
        return pd.Timestamp(obj).isoformat()
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return _normalize_for_hash(to_dict())
    if is_dataclass(obj) and not isinstance(obj, type):
        return _normalize_as_dataclass(obj)
    if isinstance(obj, Mapping):
        return {str(k): _normalize_for_hash(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_hash(v) for v in obj]
    raise TypeError(f"unsupported type for stable_hash: {type(obj)!r}")


def _normalize_as_dataclass(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(obj):
        name = f.name
        out[name] = _normalize_for_hash(getattr(obj, name))
    return {k: out[k] for k in sorted(out.keys())}
