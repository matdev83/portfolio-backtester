import unittest
from unittest.mock import mock_open, patch

from portfolio_backtester import config_loader
from portfolio_backtester.strategies._core.base.base_strategy import BaseStrategy

# Define mock content for YAML files
MOCK_PARAMS_YAML_CONTENT = """
GLOBAL_CONFIG:
  data_source: "mock_source"
  benchmark: "MOCK_SPY"
  start_date: "2000-01-01"
OPTIMIZER_PARAMETER_DEFAULTS:
  leverage:
    type: "float"
    low: 0.5
    high: 2.0
  num_holdings:
    type: "int"
    low: 5
    high: 15
  # This one is a strategy param for MockStrategy
  mock_strategy_param1:
    type: "int"
    low: 1
    high: 5
  # This one is a sizer param for mock_sizer
  sizer_mock_window:
    type: "int"
    low: 10
    high: 20
"""

MOCK_SCENARIO_1_CONTENT = """
name: "Test_Scenario_1"
strategy: "mock_strategy"
position_sizer: "equal_weight"
strategy_params:
  existing_param: 100
"""

MOCK_SCENARIO_2_CONTENT = """
name: "Test_Scenario_2"
strategy: "another_mock_strategy"
position_sizer: "mock_sizer"
strategy_params: {}
optimize:
  - parameter: "leverage" # This one is explicitly defined
    min_value: 0.8
    max_value: 1.2
"""


class MockStrategy(BaseStrategy):
    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {
            "mock_strategy_param1",
            "num_holdings",
            "existing_param",
        }  # num_holdings is often a strategy param


class AnotherMockStrategy(BaseStrategy):
    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"leverage"}  # leverage is often a strategy param


class TestConfigLoader(unittest.TestCase):

    def setUp(self):
        # Reset the module-level config variables before each test
        config_loader.GLOBAL_CONFIG = {}
        config_loader.OPTIMIZER_PARAMETER_DEFAULTS = {}
        config_loader.BACKTEST_SCENARIOS = []

    def run_load_config_with_mocks(self, params_yaml_str, scenarios_map):
        """Helper to run load_config with mocked file contents and strategy resolution, using context managers for patches."""

        with (
            patch("builtins.open", new_callable=mock_open) as mock_open_obj,
            patch("os.walk") as mock_walk,
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.is_file", return_value=True),
            patch(
                "portfolio_backtester.config_loader.validate_scenario_semantics",
                return_value=[],
            ),
            patch("portfolio_backtester.utils._resolve_strategy") as mock_resolve_strategy,
            patch(
                "portfolio_backtester.strategy_config_validator.validate_strategy_configs_comprehensive",
                return_value=(True, []),
            ),
        ):

            # Configure mock_open_obj to return different content based on the file path
            def side_effect_open(file_path, mode, encoding=None):
                if file_path == config_loader.PARAMETERS_FILE:
                    return mock_open(read_data=params_yaml_str).return_value
                # Check if the file is one of the scenario files (now in subdirectories)
                for scenario_filename, content in scenarios_map.items():
                    # Handle both old flat structure and new subdirectory structure
                    if file_path == config_loader.SCENARIOS_DIR / scenario_filename or str(
                        file_path
                    ).endswith(scenario_filename):
                        return mock_open(read_data=content).return_value
                raise FileNotFoundError(f"Unexpected file open: {file_path}")

            mock_open_obj.side_effect = side_effect_open

            # Mock strategy resolution to return our mock strategies
            def mock_strategy_resolver(strategy_name):
                if strategy_name == "mock_strategy":
                    return MockStrategy
                elif strategy_name == "another_mock_strategy":
                    return AnotherMockStrategy
                else:
                    return None

            mock_resolve_strategy.side_effect = mock_strategy_resolver

            # Mock os.walk to simulate subdirectory structure
            walk_results = []
            if scenarios_map:
                # Simulate finding files in subdirectories
                for scenario_file in scenarios_map.keys():
                    strategy_dir = str(config_loader.SCENARIOS_DIR / "mock_strategy")
                    walk_results.append((strategy_dir, [], [scenario_file]))
            else:
                # Simulate empty scenarios directory
                walk_results = [(str(config_loader.SCENARIOS_DIR), [], [])]
            mock_walk.return_value = walk_results

            # Reset module state before loading, as load_config might be called multiple times by tests
            config_loader.GLOBAL_CONFIG = {}
            config_loader.OPTIMIZER_PARAMETER_DEFAULTS = {}
            config_loader.BACKTEST_SCENARIOS = []

            # (Re)Load config using the mocks
            config_loader.load_config()

    def test_load_config_success(self):
        """Test successful loading of valid YAML files."""
        scenarios = {
            "scenario1.yaml": MOCK_SCENARIO_1_CONTENT,
            "scenario2.yaml": MOCK_SCENARIO_2_CONTENT,
        }
        self.run_load_config_with_mocks(MOCK_PARAMS_YAML_CONTENT, scenarios)

        self.assertEqual(config_loader.GLOBAL_CONFIG.get("data_source"), "mock_source")
        self.assertEqual(config_loader.GLOBAL_CONFIG.get("benchmark"), "MOCK_SPY")
        self.assertIn("leverage", config_loader.OPTIMIZER_PARAMETER_DEFAULTS)
        self.assertEqual(len(config_loader.BACKTEST_SCENARIOS), 2)
        self.assertEqual(config_loader.BACKTEST_SCENARIOS[0]["name"], "Test_Scenario_1")

    def test_load_config_file_not_found(self):
        """Test FileNotFoundError when a config file is missing."""
        with patch("pathlib.Path.exists", return_value=False):
            with self.assertRaisesRegex(
                config_loader.ConfigurationError, "Invalid parameters.yaml file"
            ):
                config_loader.load_config()

    def test_load_config_no_scenarios(self):
        """Test ConfigurationError when no scenarios are found."""
        with self.assertLogs("portfolio_backtester.config_loader", level="WARNING") as cm:
            self.run_load_config_with_mocks(MOCK_PARAMS_YAML_CONTENT, {})
            self.assertEqual(len(config_loader.BACKTEST_SCENARIOS), 0)
            self.assertTrue(any("No scenarios found" in message for message in cm.output))


if __name__ == "__main__":
    unittest.main()
