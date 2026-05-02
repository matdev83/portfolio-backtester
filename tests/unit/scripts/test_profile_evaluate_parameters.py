"""Unit tests for ``scripts/profile_evaluate_parameters`` CLI helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_profile_module():
    path = _REPO_ROOT / "scripts" / "profile_evaluate_parameters.py"
    spec = importlib.util.spec_from_file_location("_profile_evaluate_parameters", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_feature_flag_assignment_bool() -> None:
    m = _load_profile_module()
    assert m.parse_feature_flag_assignment("signal_cache=true") == ("signal_cache", True)
    assert m.parse_feature_flag_assignment("  x = FALSE ") == ("x", False)
    assert m.parse_feature_flag_assignment("n=1") == ("n", True)
    assert m.parse_feature_flag_assignment("n=0") == ("n", False)


def test_parse_feature_flag_assignment_raw_string() -> None:
    m = _load_profile_module()
    assert m.parse_feature_flag_assignment("mode=abc") == ("mode", "abc")


def test_apply_feature_flag_pairs_to_global_config() -> None:
    m = _load_profile_module()
    gc: dict = {}
    m.apply_feature_flag_pairs_to_global_config(gc, [("signal_cache", True), ("a", False)])
    assert gc["feature_flags"]["signal_cache"] is True
    assert gc["feature_flags"]["a"] is False


def test_format_profile_header_metadata_includes_flags() -> None:
    m = _load_profile_module()
    text = m.format_profile_header_metadata({"feature_flags": {"signal_cache": True}})
    assert "signal_cache" in text
    assert "True" in text
