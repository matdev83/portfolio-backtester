
import os
from pathlib import Path
import unittest
import pytest
import yaml

@pytest.mark.system
class TestConfigConsistency(unittest.TestCase):

    def test_strategy_has_scenario(self):
        """Check if each strategy has at least one corresponding scenario file."""
        strategies_dir = Path(__file__).parent.parent.parent / "src" / "portfolio_backtester" / "strategies"
        scenarios_dir = Path(__file__).parent.parent.parent / "config" / "scenarios"

        strategy_files = {f.stem.replace('_strategy', '') for f in strategies_dir.glob("*.py") if f.stem != "__init__" and f.stem != "base_strategy"}
        scenario_strategies = set()

        # Walk through subdirectories to find scenario files
        for root, dirs, files in os.walk(scenarios_dir):
            for scenario_file in files:
                if scenario_file.endswith(".yaml"):
                    scenario_path = Path(root) / scenario_file
                    with open(scenario_path, 'r') as f:
                        scenario_data = yaml.safe_load(f)
                        if scenario_data and 'strategy' in scenario_data:
                            scenario_strategies.add(scenario_data['strategy'])

        missing_scenarios = strategy_files - scenario_strategies
        self.assertEqual(len(missing_scenarios), 0, f"Strategies with missing scenarios: {missing_scenarios}")

if __name__ == '__main__':
    unittest.main()
