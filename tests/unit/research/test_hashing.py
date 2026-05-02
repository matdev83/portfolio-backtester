"""Tests for research hashing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from portfolio_backtester.canonical_config import CanonicalScenarioConfig
from portfolio_backtester.research.hashing import stable_hash


def test_stable_hash_ignores_dict_key_order() -> None:
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


def test_stable_hash_changes_when_data_changes() -> None:
    assert stable_hash({"x": 1}) != stable_hash({"x": 2})


def test_stable_hash_tuple_and_list_equivalent_payload() -> None:
    assert stable_hash([1, 2, (3, 4)]) == stable_hash((1, 2, [3, 4]))


def test_stable_hash_timestamp_path_dataclass_mapping() -> None:
    ts = pd.Timestamp("2020-03-15", tz="UTC")
    assert stable_hash(ts) == stable_hash(pd.Timestamp("2020-03-15", tz="UTC"))

    @dataclass(frozen=True)
    class Box:
        v: int

    assert stable_hash(Box(3)) == stable_hash({"v": 3})
    assert stable_hash({"path": Path("a/b")}) == stable_hash({"path": "a/b"})
    assert stable_hash({"when": ts}) == stable_hash({"when": "2020-03-15T00:00:00+00:00"})


def test_stable_hash_canonical_scenario() -> None:
    cfg = CanonicalScenarioConfig(name="n", strategy="sig")
    h1 = stable_hash(cfg)
    h2 = stable_hash(cfg.to_dict())
    assert h1 == h2


def test_stable_hash_object_with_to_dict_only() -> None:
    class Payload:
        def to_dict(self) -> dict[str, int]:
            return {"z": 9}

    assert stable_hash(Payload()) == stable_hash({"z": 9})


def test_stable_hash_rejects_unsupported_type() -> None:
    class NoDice:
        pass

    with pytest.raises(TypeError):
        stable_hash(NoDice())
