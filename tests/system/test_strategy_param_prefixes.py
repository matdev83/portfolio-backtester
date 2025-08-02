import pytest
from pathlib import Path

from src.portfolio_backtester.config_schema import StrategyConfigSchema


SCENARIO_DIR = Path(__file__).resolve().parents[2] / "config" / "scenarios"


def collect_yaml_files():
    """Return iterator of all scenario YAML paths."""
    for path in SCENARIO_DIR.rglob("*.yaml"):
        yield path


@pytest.mark.parametrize("yaml_path", list(collect_yaml_files()))
def test_strategy_param_prefixes(yaml_path):
    """All scenario files must use the <strategy>.prefix convention inside strategy_params."""
    errors = StrategyConfigSchema.validate_yaml_file(yaml_path)
    prefix_errors = [e for e in errors if e.severity == "error"]
    assert not prefix_errors, (
        f"Prefix validation failed for {yaml_path.relative_to(SCENARIO_DIR.parent)}:\n" +
        StrategyConfigSchema.format_report(prefix_errors)
    )
