import tempfile
from pathlib import Path
from unittest.mock import patch

from portfolio_backtester.config_loader import load_scenario_from_file


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
strategy: SimpleMetaStrategy
strategy_params:
  initial_capital: 1000000.0
  min_allocation: 0.05
  rebalance_threshold: 0.05
  allocations:
    - strategy_id: momentum_strategy
      strategy_class: MomentumPortfolioStrategy
      strategy_params:
        lookback_period: 12
      weight: 1.0
optimizers:
  optuna:
    optimize:
      - parameter: min_allocation
        min_value: 0.01
        max_value: 0.2
        step: 0.01
      - parameter: rebalance_threshold
        min_value: 0.01
        max_value: 0.1
        step: 0.01"""
    path = _create_temp_yaml(yaml_content)
    try:
        with patch(
            "portfolio_backtester.config_loader.validate_scenario_semantics",
            return_value=[],
        ):
            scenario = load_scenario_from_file(path)
        # 'optimizers' key should be removed after merge
        assert "optimizers" not in scenario, "'optimizers' section was not flattened/removed"
        # The optimizer-specific keys (e.g. 'optimize') should be promoted to top level
        assert "optimize" in scenario, "Optimizer configuration not promoted to top-level"
    finally:
        path.unlink(missing_ok=True)
