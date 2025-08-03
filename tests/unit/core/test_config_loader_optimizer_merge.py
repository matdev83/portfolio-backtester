import tempfile
from pathlib import Path

from src.portfolio_backtester.config_loader import load_scenario_from_file, ConfigurationError


def _create_temp_yaml(content: str) -> Path:
    """Helper to write YAML string to a temporary file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def test_optimizer_section_is_flattened():
    """Scenario files containing an 'optimizers' section should be flattened so that
    the selected optimizer's keys (e.g. 'optimize') reside at the top level while
    the 'optimizers' mapping itself is removed.
    """

    yaml_content = """
name: test_optimizer_flatten
strategy: dummy_strategy_for_testing
strategy_params:
  dummy_strategy_for_testing.open_long_prob: 0.1
optimizers:
  optuna:
    optimize:
      - parameter: open_long_prob
        min_value: 0.05
        max_value: 0.3
        step: 0.01
"""
    path = _create_temp_yaml(yaml_content)
    try:
        scenario = load_scenario_from_file(path)
        # 'optimizers' key should be removed after merge
        assert 'optimizers' not in scenario, "'optimizers' section was not flattened/removed"
        # The optimizer-specific keys (e.g. 'optimize') should be promoted to top level
        assert 'optimize' in scenario, "Optimizer configuration not promoted to top-level"
    finally:
        path.unlink(missing_ok=True)
