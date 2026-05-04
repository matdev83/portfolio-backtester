"""Ensure the legacy timing schema shim warns once per fresh import."""

import sys
import warnings

import pytest


def test_config_schema_refactored_emits_deprecation_on_fresh_import() -> None:
    name = "portfolio_backtester.timing.config_schema_refactored"
    sys.modules.pop(name, None)
    with warnings.catch_warnings():
        warnings.simplefilter("always", DeprecationWarning)
        with pytest.warns(DeprecationWarning, match="config_schema_refactored is deprecated"):
            __import__(name, fromlist=["*"])
