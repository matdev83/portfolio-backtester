import unittest
from unittest.mock import mock_open, patch

from src.portfolio_backtester import config_initializer, config_loader
from src.portfolio_backtester.strategies.base_strategy import BaseStrategy

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

MOCK_SCENARIOS_YAML_CONTENT = """
BACKTEST_SCENARIOS:
  - name: "Test_Scenario_1"
    strategy: "mock_strategy"
    position_sizer: "equal_weight"
    strategy_params:
      existing_param: 100
    # 'optimize' section intentionally missing or incomplete for some tests

  - name: "Test_Scenario_2"
    strategy: "another_mock_strategy"
    position_sizer: "mock_sizer"
    strategy_params: {}
    optimize:
      - parameter: "leverage" # This one is explicitly defined
        min_value: 0.8
        max_value: 1.2
      # mock_strategy_param1 and sizer_mock_window will be added by populate_default_optimizations

  - name: "Test_Scenario_Full_Optimize"
    strategy: "mock_strategy"
    position_sizer: "mock_sizer"
    strategy_params: {}
    optimize:
      - parameter: "leverage"
        min_value: 1.0
        max_value: 1.5
      - parameter: "num_holdings"
        min_value: 7
        max_value: 12
      - parameter: "mock_strategy_param1"
        min_value: 2
        max_value: 4
      - parameter: "sizer_mock_window"
        min_value: 12
        max_value: 18
"""

class MockStrategy(BaseStrategy):
    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"mock_strategy_param1", "num_holdings"} # num_holdings is often a strategy param

class AnotherMockStrategy(BaseStrategy):
     @classmethod
     def tunable_parameters(cls) -> set[str]:
        return {"leverage"} # leverage is often a strategy param

class TestConfigLoader(unittest.TestCase):

    def setUp(self):
        # Reset the module-level config variables before each test
        config_loader.GLOBAL_CONFIG = {}
        config_loader.OPTIMIZER_PARAMETER_DEFAULTS = {}
        config_loader.BACKTEST_SCENARIOS = []

        # Patch Path.exists() to return True by default for config files
        self.patch_exists = patch('pathlib.Path.exists', return_value=True)
        self.mock_exists = self.patch_exists.start()
        self.addCleanup(self.patch_exists.stop)

        # Patch _resolve_strategy where it is used (in config_initializer)
        self.patch_resolve = patch('src.portfolio_backtester.config_initializer._resolve_strategy')
        self.mock_resolve = self.patch_resolve.start()
        self.addCleanup(self.patch_resolve.stop)

        def resolve_side_effect(strategy_name):
            if strategy_name == "mock_strategy":
                return MockStrategy
            if strategy_name == "another_mock_strategy":
                return AnotherMockStrategy
            if strategy_name == "mock_strategy_non_default":
                class MockStrategyWithNonDefaultParam(BaseStrategy):
                    @classmethod
                    def tunable_parameters(cls) -> set[str]:
                        return {"param_not_in_defaults", "num_holdings"}
                return MockStrategyWithNonDefaultParam
            return None
        self.mock_resolve.side_effect = resolve_side_effect


    def run_load_config_with_mocks(self, params_yaml_str, scenarios_yaml_str):
        """Helper to run load_config with mocked file contents and strategy resolution, using context managers for patches."""

        with patch('builtins.open', new_callable=mock_open) as mock_open_obj:

            # Configure mock_open_obj to return different content based on the file path
            def side_effect_open(file_path, mode, encoding=None):
                if file_path == config_loader.PARAMETERS_FILE:
                    return mock_open(read_data=params_yaml_str).return_value
                elif file_path == config_loader.SCENARIOS_FILE:
                    return mock_open(read_data=scenarios_yaml_str).return_value
                raise FileNotFoundError(f"Unexpected file open: {file_path}")
            mock_open_obj.side_effect = side_effect_open

            # Reset module state before loading, as load_config might be called multiple times by tests
            config_loader.GLOBAL_CONFIG = {}
            config_loader.OPTIMIZER_PARAMETER_DEFAULTS = {}
            config_loader.BACKTEST_SCENARIOS = []

            # (Re)Load config using the mocks
            config_loader.load_config()
            config_initializer.populate_default_optimizations(config_loader.BACKTEST_SCENARIOS, config_loader.OPTIMIZER_PARAMETER_DEFAULTS)


    def test_load_config_success(self):
        """Test successful loading of valid YAML files."""
        self.run_load_config_with_mocks(MOCK_PARAMS_YAML_CONTENT, MOCK_SCENARIOS_YAML_CONTENT)

        self.assertEqual(config_loader.GLOBAL_CONFIG.get("data_source"), "mock_source")
        self.assertEqual(config_loader.GLOBAL_CONFIG.get("benchmark"), "MOCK_SPY")
        self.assertIn("leverage", config_loader.OPTIMIZER_PARAMETER_DEFAULTS)
        self.assertTrue(len(config_loader.BACKTEST_SCENARIOS) > 0)
        self.assertEqual(config_loader.BACKTEST_SCENARIOS[0]["name"], "Test_Scenario_1")

    def test_load_config_file_not_found(self):
        """Test FileNotFoundError when a config file is missing."""
        # Order of checks in load_config(): PARAMETERS_FILE, then SCENARIOS_FILE

        # Test case 1: PARAMETERS_FILE is missing
        self.mock_exists.side_effect = [False, True] # PARAMETERS_FILE.exists() -> False, SCENARIOS_FILE.exists() -> True (though not reached)
        with self.assertRaisesRegex(config_loader.ConfigurationError, "Invalid parameters.yaml file"):
            config_loader.load_config() # populate_default_optimizations is not called if load_config fails
        self.assertEqual(self.mock_exists.call_count, 1) # Called for PARAMETERS_FILE

        # Reset call count and set new side_effect for the next assertion
        self.mock_exists.reset_mock()
        # Test case 2: SCENARIOS_FILE is missing
        self.mock_exists.side_effect = [True, False] # PARAMETERS_FILE.exists() -> True, SCENARIOS_FILE.exists() -> False
        with self.assertRaisesRegex(config_loader.ConfigurationError, "Invalid scenarios.yaml file"):
            config_loader.load_config()
        self.assertEqual(self.mock_exists.call_count, 2) # Called for PARAMETERS_FILE then SCENARIOS_FILE


    def test_get_strategy_tunable_params(self):
        """Test the helper function for getting tunable strategy parameters."""
        self.mock_resolve.return_value = MockStrategy
        params = config_initializer._get_strategy_tunable_params("mock_strategy")
        self.assertEqual(params, {"mock_strategy_param1", "num_holdings"})

        self.mock_resolve.return_value = None
        params = config_initializer._get_strategy_tunable_params("non_existent_strategy")
        self.assertEqual(params, set())

    def test_get_sizer_tunable_param(self):
        """Test the helper function for getting tunable sizer parameters."""
        sizer_map = {"mock_sizer": "sizer_mock_window"}
        param = config_initializer._get_sizer_tunable_param("mock_sizer", sizer_map)
        self.assertEqual(param, "sizer_mock_window")

        param = config_initializer._get_sizer_tunable_param("non_existent_sizer", sizer_map)
        self.assertIsNone(param)

        param = config_initializer._get_sizer_tunable_param(None, sizer_map)
        self.assertIsNone(param)

    def test_populate_default_optimizations(self):
        """Test that default optimization parameters are populated correctly."""
        self.run_load_config_with_mocks(MOCK_PARAMS_YAML_CONTENT, MOCK_SCENARIOS_YAML_CONTENT)

        # Scenario 1: "mock_strategy", "equal_weight" sizer (no sizer-specific param from map)
        # Tunable for MockStrategy: "mock_strategy_param1", "num_holdings"
        # OPTIMIZER_PARAMETER_DEFAULTS has all of these.
        scenario1 = next(s for s in config_loader.BACKTEST_SCENARIOS if s["name"] == "Test_Scenario_1")
        s1_opt_params = {opt["parameter"] for opt in scenario1.get("optimize", [])}
        self.assertIn("mock_strategy_param1", s1_opt_params)
        self.assertIn("num_holdings", s1_opt_params)
        self.assertEqual(len(s1_opt_params), 2) # Ensure only these were added

        # Scenario 2: "another_mock_strategy", "mock_sizer"
        # Tunable for AnotherMockStrategy: "leverage" (already in optimize)
        # Tunable for mock_sizer: "sizer_mock_window" (from sizer_param_map in populate_default_optimizations)
        # OPTIMIZER_PARAMETER_DEFAULTS has "sizer_mock_window" and "leverage".
        scenario2 = next(s for s in config_loader.BACKTEST_SCENARIOS if s["name"] == "Test_Scenario_2")
        s2_opt_params = {opt["parameter"] for opt in scenario2.get("optimize", [])}

        # another_mock_strategy only has "leverage" (explicitly in optimize list).
        # "mock_sizer" is not in config_loader's sizer_param_map, so no sizer params are added.
        expected_s2_params = {"leverage"}
        self.assertEqual(s2_opt_params, expected_s2_params)


    def test_populate_default_optimizations_parameter_not_in_defaults(self):
        """Test that parameters are not added if not in OPTIMIZER_PARAMETER_DEFAULTS."""

        custom_scenarios_content = """
BACKTEST_SCENARIOS:
  - name: "Test_Non_Default_Param"
    strategy: "mock_strategy_non_default"
    position_sizer: "equal_weight"
    strategy_params: {}
"""
        with patch('builtins.open', new_callable=mock_open) as mock_file:
            def side_effect_open(file_path, mode, encoding=None):
                if file_path == config_loader.PARAMETERS_FILE:
                    return mock_open(read_data=MOCK_PARAMS_YAML_CONTENT).return_value
                elif file_path == config_loader.SCENARIOS_FILE:
                    return mock_open(read_data=custom_scenarios_content).return_value
                raise FileNotFoundError(f"Unexpected file open: {file_path}")
            mock_file.side_effect = side_effect_open

            config_loader.load_config()
            config_initializer.populate_default_optimizations(config_loader.BACKTEST_SCENARIOS, config_loader.OPTIMIZER_PARAMETER_DEFAULTS)

        scenario = config_loader.BACKTEST_SCENARIOS[0]
        opt_params = {opt["parameter"] for opt in scenario.get("optimize", [])}

        self.assertNotIn("param_not_in_defaults", opt_params)
        self.assertIn("num_holdings", opt_params) # This one is in defaults
        self.assertEqual(len(opt_params), 1)


    def test_full_optimize_section_unchanged(self):
        """Test that a scenario with a fully defined optimize section is not modified by populate."""
        self.run_load_config_with_mocks(MOCK_PARAMS_YAML_CONTENT, MOCK_SCENARIOS_YAML_CONTENT)

        scenario_full = next(s for s in config_loader.BACKTEST_SCENARIOS if s["name"] == "Test_Scenario_Full_Optimize")
        original_optimize_params = {
            "leverage", "num_holdings", "mock_strategy_param1", "sizer_mock_window"
        }
        populated_optimize_params = {opt["parameter"] for opt in scenario_full.get("optimize", [])}

        self.assertEqual(populated_optimize_params, original_optimize_params)
        # Also check if the values were preserved (though populate_default_optimizations only adds keys)
        # This check implicitly confirms no extra keys were added.
        self.assertEqual(len(scenario_full["optimize"]), len(original_optimize_params))


if __name__ == '__main__':
    unittest.main()