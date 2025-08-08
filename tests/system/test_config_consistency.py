
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

        # Exclude utility classes, abstract classes, and base classes
        excluded_files = {
            "strategy_factory", "leverage_and_smoothing", "candidate_weights",
            "signal_generator", "ema_signal_generator", "rsi_calculator", "price_data_processor",
            "base_portfolio_momentum_strategy", "portfolio_momentum_strategy", "portfolio_momentum"
        }
        strategy_files = {
            f.stem.replace('_strategy', '') 
            for f in strategies_dir.glob("**/*.py") 
            if f.stem != "__init__" 
            and not f.parent.name == "base" 
            and f.stem not in excluded_files
        }
        scenario_strategies = set()

        # Walk through subdirectories to find scenario files
        for root, dirs, files in os.walk(scenarios_dir):
            for scenario_file in files:
                if scenario_file.endswith(".yaml"):
                    scenario_path = Path(root) / scenario_file
                    with open(scenario_path, 'r') as f:
                        scenario_data = yaml.safe_load(f)
                        if scenario_data and isinstance(scenario_data, dict) and 'strategy' in scenario_data:
                            strategy_val = scenario_data['strategy']
                            if isinstance(strategy_val, dict):
                                scenario_strategies.add(strategy_val.get('type'))
                            else:
                                scenario_strategies.add(strategy_val)

        missing_scenarios = strategy_files - scenario_strategies
        self.assertEqual(len(missing_scenarios), 0, f"Strategies with missing scenarios: {missing_scenarios}")

if __name__ == '__main__':
    unittest.main()
