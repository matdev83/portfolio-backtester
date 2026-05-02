"""Integration checks for checked-in research_validate example scenario YAML files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from portfolio_backtester.scenario_normalizer import ScenarioNormalizer
from portfolio_backtester.research.protocol_config import (
    DoubleOOSWFOProtocolConfig,
    parse_double_oos_wfo_protocol,
)

_EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[3] / "config" / "scenarios" / "examples" / "research"
)


def _example_yaml_paths() -> list[Path]:
    if not _EXAMPLES_ROOT.exists():
        return []
    return sorted(_EXAMPLES_ROOT.glob("*.yaml"))


@pytest.mark.parametrize("path", _example_yaml_paths(), ids=lambda p: p.name)
def test_example_yaml_normalizes_and_parses_protocol(path: Path) -> None:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)

    norm = ScenarioNormalizer()
    canon = norm.normalize(scenario=dict(loaded), global_config={}, source=str(path))
    rp_any = canon.extras.get("research_protocol")
    assert isinstance(rp_any, Mapping)

    parsed = parse_double_oos_wfo_protocol({"research_protocol": rp_any})
    assert isinstance(parsed, DoubleOOSWFOProtocolConfig)
    parsed.research_validate()
