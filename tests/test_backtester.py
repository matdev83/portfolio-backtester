import unittest
from typing import Optional  # Import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.portfolio_backtester.backtester import Backtester
from src.portfolio_backtester.config_loader import GLOBAL_CONFIG
# from src.portfolio_backtester.features.feature_helpers import get_required_features_from_scenarios # Removed
# from src.portfolio_backtester.feature_engineering import precompute_features # Removed

class TestBacktester(unittest.TestCase):
    class MockArgs:
        def __init__(self):
            self.optimize_min_positions: int = 10
            self.optimize_max_positions: int = 30
            self.top_n_params: int = 3
            self.n_jobs: int = 1
            self.optuna_trials: int = 10
            self.optuna_timeout_sec: Optional[int] = 60
            self.study_name: Optional[str] = "test_study"
            self.random_seed: Optional[int] = None # Allow int or None
            self.storage_url: Optional[str] = None
            self.early_stop_patience: int = 10
            self.pruning_enabled: bool = False
            self.pruning_n_startup_trials: int = 5
            self.pruning_n_warmup_steps: int = 0
            self.pruning_interval_steps: int = 1
            self.optimizer: str = "optuna"
            self.mode: str = "backtest" # Add default mode
            self.mc_simulations: int = 1000 # Add default for MC
            self.mc_years: int = 10 # Add default for MC
            self.interactive: bool = False # Add default for interactive
            self.timeout: Optional[int] = None

    def setUp(self):
        """Set up a mock backtester and data for testing."""
        self.global_config = GLOBAL_CONFIG.copy()
        self.scenarios = [
            {
                "name": "Test_Momentum_WFO",
                "strategy": "momentum",
                "strategy_params": {
                    "lookback_months": 6,
                    "leverage": 1.0,
                    "smoothing_lambda": 0.5,
                    "long_only": True,
                    "top_decile_fraction": 0.1
                },
                "rebalance_frequency": "M",
                "position_sizer": "equal_weight",
                "transaction_costs_bps": 10,
                "train_window_months": 24,
                "test_window_months": 6,
                "optimization_metric": "Sharpe", # Added scenario-level metric
                "optimize": [
                    {
                        "parameter": "lookback_months",
                        # "metric": "Sharpe", # Metric removed from here
                        "min_value": 3,
                        "max_value": 9,
                        "step": 3
                    }
                ]
            }
        ]
        self.mock_args = self.MockArgs() # Use the class attribute
        # self.backtester = Backtester(self.global_config, self.scenarios, self.mock_args) # Defer to specific tests or a more minimal setup
        
        # Create mock data with sufficient history for walk-forward optimization
        # Need at least train_window_months (24) + test_window_months (6) = 30 months minimum
        # Add extra buffer for multiple windows
        dates = pd.date_range(start="2018-01-01", periods=72, freq="ME")  # 6 years of data
        tickers = self.global_config["universe"] + [self.global_config["benchmark"]]
        self.mock_data = pd.DataFrame(
            np.random.randn(72, len(tickers)) / 100 + 0.001,
            index=dates,
            columns=tickers
        )
        # Pre-computation of features is now handled within strategies
        pass


    @patch('src.portfolio_backtester.backtester.Backtester._get_data_source')
    @patch('src.portfolio_backtester.backtester.np.random.seed')
    def test_init_basic_attributes(self, mock_np_seed, mock_get_data_source):
        mock_get_data_source.return_value = "mock_data_source_instance"
        args = self.MockArgs()
        backtester = Backtester(self.global_config, self.scenarios, args, random_state=123)

        self.assertEqual(backtester.global_config, self.global_config)
        self.assertIn("optimizer_parameter_defaults", backtester.global_config) # Check if defaults are added
        self.assertEqual(backtester.scenarios, self.scenarios)
        self.assertEqual(backtester.args, args)
        self.assertEqual(backtester.data_source, "mock_data_source_instance")
        self.assertEqual(backtester.results, {})
        self.assertEqual(backtester.n_jobs, args.n_jobs)
        self.assertEqual(backtester.early_stop_patience, 10) # Default from MockArgs or Backtester default
        mock_np_seed.assert_called_with(123)

    @patch('src.portfolio_backtester.backtester.Backtester._get_data_source')
    @patch('src.portfolio_backtester.backtester.np.random.seed')
    @patch('src.portfolio_backtester.backtester.np.random.randint')
    def test_init_random_seed_generation(self, mock_np_randint, mock_np_seed, mock_get_data_source):
        mock_np_randint.return_value = 42 # Mock the generated seed value
        args = self.MockArgs()
        args.random_seed = None # Ensure it's None to trigger generation

        backtester = Backtester(self.global_config, self.scenarios, args, random_state=None) # Explicitly pass None

        mock_np_randint.assert_called_once_with(0, 2**31 - 1)
        mock_np_seed.assert_called_with(42)
        self.assertEqual(backtester.random_state, 42)

    @patch('src.portfolio_backtester.backtester.Backtester._get_data_source')
    @patch('src.portfolio_backtester.backtester.np.random.seed')
    def test_init_provided_random_seed(self, mock_np_seed, mock_get_data_source):
        args = self.MockArgs()
        args.random_seed = 777 # Provide a seed via args

        # Test when random_state is passed directly to constructor
        backtester_direct_seed = Backtester(self.global_config, self.scenarios, args, random_state=777)
        mock_np_seed.assert_called_with(777)
        self.assertEqual(backtester_direct_seed.random_state, 777)

    @patch('src.portfolio_backtester.backtester.Backtester._get_data_source')
    def test_init_job_settings_from_args(self, mock_get_data_source):
        args = self.MockArgs()
        args.n_jobs = 4
        args.early_stop_patience = 20

        backtester = Backtester(self.global_config, self.scenarios, args)

        self.assertEqual(backtester.n_jobs, 4)
        self.assertEqual(backtester.early_stop_patience, 20)

    @patch('src.portfolio_backtester.data_sources.stooq_data_source.StooqDataSource')
    def test_get_data_source_stooq(self, mock_stooq_data_source_class):
        mock_instance = MagicMock()
        mock_stooq_data_source_class.return_value = mock_instance

        config = self.global_config.copy()
        config["data_source"] = "stooq"
        args = self.MockArgs()
        # Temporarily patch np.random.seed for Backtester instantiation
        with patch('src.portfolio_backtester.backtester.np.random.seed'):
            backtester = Backtester(config, self.scenarios, args)

        # _get_data_source is called in __init__, so we check the instance attribute
        self.assertEqual(backtester.data_source, mock_instance)
        mock_stooq_data_source_class.assert_called_once_with()


    @patch('src.portfolio_backtester.data_sources.yfinance_data_source.YFinanceDataSource')
    def test_get_data_source_yfinance(self, mock_yfinance_data_source_class):
        mock_instance = MagicMock()
        mock_yfinance_data_source_class.return_value = mock_instance

        config = self.global_config.copy()
        config["data_source"] = "yfinance"
        args = self.MockArgs()
        with patch('src.portfolio_backtester.backtester.np.random.seed'):
            backtester = Backtester(config, self.scenarios, args)

        self.assertEqual(backtester.data_source, mock_instance)
        mock_yfinance_data_source_class.assert_called_once_with()

    @patch('src.portfolio_backtester.data_sources.hybrid_data_source.HybridDataSource') # Default
    def test_get_data_source_default_to_hybrid(self, mock_hybrid_data_source_class):
        mock_instance = MagicMock()
        mock_hybrid_data_source_class.return_value = mock_instance

        config = self.global_config.copy()
        if "data_source" in config: # Ensure it's not set for this test
            del config["data_source"]
        args = self.MockArgs()
        with patch('src.portfolio_backtester.backtester.np.random.seed'):
            backtester = Backtester(config, self.scenarios, args)

        self.assertEqual(backtester.data_source, mock_instance)
        mock_hybrid_data_source_class.assert_called_once_with(prefer_stooq=True)

    @patch('src.portfolio_backtester.data_sources.hybrid_data_source.HybridDataSource')
    def test_get_data_source_hybrid(self, mock_hybrid_data_source_class):
        mock_instance = MagicMock()
        mock_hybrid_data_source_class.return_value = mock_instance

        config = self.global_config.copy()
        config["data_source"] = "hybrid"
        config["prefer_stooq"] = False
        args = self.MockArgs()
        with patch('src.portfolio_backtester.backtester.np.random.seed'):
            backtester = Backtester(config, self.scenarios, args)

        self.assertEqual(backtester.data_source, mock_instance)
        mock_hybrid_data_source_class.assert_called_once_with(prefer_stooq=False)

    def test_get_data_source_unsupported(self):
        config = self.global_config.copy()
        config["data_source"] = "unsupported_source"
        args = self.MockArgs()
        with patch('src.portfolio_backtester.backtester.np.random.seed'):
            with self.assertRaisesRegex(ValueError, "Unsupported data source: unsupported_source"):
                Backtester(config, self.scenarios, args)

    # Tests for _get_strategy
    @patch('src.portfolio_backtester.strategies.MomentumStrategy')
    def test_get_strategy_valid(self, mock_strategy_class):
        mock_strategy_instance = MagicMock()
        mock_strategy_class.return_value = mock_strategy_instance
        params = {"lookback": 10}

        # Need a backtester instance to call _get_strategy on
        # Mock _get_data_source to simplify Backtester instantiation
        with patch.object(Backtester, '_get_data_source', return_value=MagicMock()):
            args = self.MockArgs()
            backtester = Backtester(self.global_config, self.scenarios, args)

        strategy_instance = backtester._get_strategy("momentum", params)

        self.assertEqual(strategy_instance, mock_strategy_instance)
        mock_strategy_class.assert_called_once_with(params)

    @patch('src.portfolio_backtester.strategies.VAMSMomentumStrategy')
    def test_get_strategy_vams_momentum(self, mock_vams_strategy_class):
        mock_strategy_instance = MagicMock()
        mock_vams_strategy_class.return_value = mock_strategy_instance
        params = {"alpha": 0.5}
        with patch.object(Backtester, '_get_data_source', return_value=MagicMock()):
            args = self.MockArgs()
            backtester = Backtester(self.global_config, self.scenarios, args)
        strategy_instance = backtester._get_strategy("vams_momentum", params)
        self.assertEqual(strategy_instance, mock_strategy_instance)
        mock_vams_strategy_class.assert_called_once_with(params)

    @patch('src.portfolio_backtester.strategies.VAMSNoDownsideStrategy')
    def test_get_strategy_vams_no_downside(self, mock_vams_nd_strategy_class):
        mock_strategy_instance = MagicMock()
        mock_vams_nd_strategy_class.return_value = mock_strategy_instance
        params = {}
        with patch.object(Backtester, '_get_data_source', return_value=MagicMock()):
            args = self.MockArgs()
            backtester = Backtester(self.global_config, self.scenarios, args)
        strategy_instance = backtester._get_strategy("vams_no_downside", params)
        self.assertEqual(strategy_instance, mock_strategy_instance)
        mock_vams_nd_strategy_class.assert_called_once_with(params)

    def test_get_strategy_unsupported(self):
        params = {}
        with patch.object(Backtester, '_get_data_source', return_value=MagicMock()):
            args = self.MockArgs()
            backtester = Backtester(self.global_config, self.scenarios, args)

        with self.assertRaisesRegex(ValueError, "Unsupported strategy: non_existent_strategy"):
            backtester._get_strategy("non_existent_strategy", params)

    # Tests for run_scenario
    def test_run_scenario_basic_flow(self):
        args = self.MockArgs()
        # Mock _get_data_source for Backtester instantiation
        with patch.object(Backtester, '_get_data_source', return_value=MagicMock()):
            backtester = Backtester(self.global_config, self.scenarios, args)

        scenario_config = {
            "name": "TestScenario",
            "strategy": "mock_strategy",
            "strategy_params": {"p1": 1, "sizer_dvol_window": 5}, # Include a sizer param
            "rebalance_frequency": "M",
            "position_sizer": "rolling_downside_volatility", # Change to a real sizer name to trigger window param passing
            "transaction_costs_bps": 10
        }

        # Mock data
        dates_daily = pd.date_range(start="2023-01-01", periods=60, freq="B") # Approx 3 months
        dates_monthly = pd.date_range(start="2023-01-31", periods=3, freq="ME")  # Use explicit monthly end dates to avoid duplicates

        mock_universe_tickers = ["TICK1", "TICK2"]

        price_data_monthly = pd.DataFrame(np.random.rand(len(dates_monthly), len(mock_universe_tickers)), index=dates_monthly, columns=mock_universe_tickers)
        price_data_daily = pd.DataFrame(np.random.rand(len(dates_daily), len(mock_universe_tickers)), index=dates_daily, columns=mock_universe_tickers)
        # Add benchmark to price data as it's expected by _get_strategy -> get_universe
        price_data_monthly[self.global_config["benchmark"]] = np.random.rand(len(dates_monthly))
        price_data_daily[self.global_config["benchmark"]] = np.random.rand(len(dates_daily))


        # Mock dependencies
        mock_strategy_obj = MagicMock()
        mock_strategy_obj.get_universe.return_value = [(ticker, "EQUITY") for ticker in mock_universe_tickers]
        
        # Create mock signals that return a single row DataFrame for each call
        def mock_generate_signals(*args, **kwargs):
            # Return a single row DataFrame with the current date as index
            current_date = kwargs.get('current_date', dates_monthly[0])
            return pd.DataFrame(np.random.rand(1, len(mock_universe_tickers)), 
                              index=[current_date], 
                              columns=mock_universe_tickers)
        
        mock_strategy_obj.generate_signals.side_effect = mock_generate_signals

        mock_position_sizer_func = MagicMock()
        mock_sized_signals = pd.DataFrame(np.random.rand(len(dates_monthly), len(mock_universe_tickers)) * 0.5, index=dates_monthly, columns=mock_universe_tickers) # e.g. 0.5 weight per stock
        mock_position_sizer_func.return_value = mock_sized_signals

        mock_weights_monthly = pd.DataFrame(0.5, index=dates_monthly, columns=mock_universe_tickers) # Constant 0.5 weights
        mock_weights_monthly.iloc[0] = 0 # Start with no holdings for turnover calc

        # Patching internal calls
        with patch.object(backtester, '_get_strategy', return_value=mock_strategy_obj) as mock_get_strat, \
             patch('src.portfolio_backtester.backtester.get_position_sizer', return_value=mock_position_sizer_func) as mock_get_pos_sizer, \
             patch('src.portfolio_backtester.backtester.rebalance', return_value=mock_weights_monthly) as mock_rebalance_func:

            portfolio_rets = backtester.run_scenario(scenario_config, price_data_monthly, price_data_daily)

            # Assertions
            mock_get_strat.assert_called_once_with(scenario_config["strategy"], scenario_config["strategy_params"])
            # generate_signals should be called once for each rebalance date (3 times for 3 monthly dates)
            self.assertEqual(mock_strategy_obj.generate_signals.call_count, len(dates_monthly))
            # Check sizer params (dvol_window should be mapped to window)
            expected_sizer_params = {} # window is now passed as a positional argument
            mock_get_pos_sizer.assert_called_once_with(scenario_config["position_sizer"])
            mock_position_sizer_func.assert_called_once()
            args, kwargs = mock_position_sizer_func.call_args
            # Verify that signals DataFrame is passed as first argument
            self.assertIsInstance(args[0], pd.DataFrame)
            self.assertEqual(list(args[0].columns), mock_universe_tickers)
            # Verify strategy data is passed
            pd.testing.assert_frame_equal(args[1], price_data_monthly[mock_universe_tickers]) # strategy_data_monthly
            pd.testing.assert_series_equal(args[2], price_data_monthly[self.global_config["benchmark"]]) # benchmark_data_monthly is passed as a Series from backtester
            self.assertEqual(kwargs, expected_sizer_params)
            # Assert that the correct number of arguments are passed to the sizer function.
            # For "rolling_downside_volatility", it should be 5 positional arguments.
            self.assertEqual(len(args), 5)
            # Assert that window=5 is passed as the last positional argument (index 4).
            self.assertEqual(args[4], 5)


            mock_rebalance_func.assert_called_once_with(mock_sized_signals, scenario_config["rebalance_frequency"])

            self.assertIsInstance(portfolio_rets, pd.Series)
            self.assertEqual(len(portfolio_rets), len(dates_daily))

            # Validate P&L calculation for a couple of days (simplified)
            # This part is tricky without replicating the exact logic.
            # For now, focus on mocks being called and basic output properties.
            # A more detailed P&L check would require known price movements and weights.

            # Example: check if returns are roughly in an expected range if all inputs are positive,
            # but this is very loose.
            # self.assertTrue(all(r > -0.1 for r in portfolio_rets)) # Very basic sanity check

            # Basic validation - just check that we get reasonable output
            # The exact calculation is complex due to mocking, so we focus on structure
            self.assertTrue(all(abs(r) < 100 for r in portfolio_rets), "Returns should be reasonable (not extreme)")
            
            # Check that we have some non-zero returns (since we have non-zero weights and price changes)
            non_zero_returns = portfolio_rets[portfolio_rets != 0]
            self.assertGreater(len(non_zero_returns), 0, "Should have some non-zero returns")


    # Keep the existing test, but we need to ensure self.backtester is initialized for it.
    # One way is to call its setup within the test, or ensure setUp creates a default backtester.
    # For now, let's adapt its setup slightly.
    def test_walk_forward_optimization_runs(self):
        """Test that walk-forward optimization completes and produces results."""
        # Specific setup for this test
        with patch('src.portfolio_backtester.backtester.Backtester._get_data_source') as mock_get_ds:
            mock_get_ds.return_value = MagicMock() # Mock the data source
            backtester_for_wfo = Backtester(self.global_config, self.scenarios, self.mock_args)

        # Pre-computation of features is now handled within strategies

        # Pre-calculate returns for the mock data
        rets_full = self.mock_data.pct_change(fill_method=None).fillna(0)
        
        # Patch Optuna related calls if they are external or too heavy
        with patch('src.portfolio_backtester.backtester_logic.optimization.optuna.create_study'), \
             patch('src.portfolio_backtester.backtester_logic.optimization.Progress'), \
             patch('src.portfolio_backtester.backtester.Backtester.run_scenario') as mock_run_scenario:

            # Mock run_scenario to return some dummy returns
            mock_run_scenario.return_value = pd.Series(np.random.randn(100), index=pd.date_range(start="2020-01-01", periods=100))

            backtester_for_wfo.run_optimize_mode(self.scenarios[0], self.mock_data, self.mock_data, rets_full)
        
        self.assertIn("Test_Momentum_WFO (Optimized)", backtester_for_wfo.results)
        result_data = backtester_for_wfo.results["Test_Momentum_WFO (Optimized)"]
        self.assertIsInstance(result_data["returns"], pd.Series)
        self.assertFalse(result_data["returns"].empty)

    @patch('src.portfolio_backtester.backtester_logic.optimization.GeneticOptimizer')
    @patch('src.portfolio_backtester.backtester.Backtester._get_data_source') # Mock data source loading
    def test_run_optimization_with_genetic_optimizer(self, mock_get_ds, mock_genetic_optimizer_class):
        """Test that run_optimization calls GeneticOptimizer when specified."""
        args = self.MockArgs()
        args.optimizer = "genetic" # Specify genetic optimizer
        args.mode = "optimize" # Ensure optimize mode

        # Mock data source and feature computation for Backtester initialization
        mock_get_ds.return_value = MagicMock()

        # Minimal scenario for testing optimizer dispatch
        scenario_config = self.scenarios[0].copy() # Use a copy of the existing scenario

        backtester = Backtester(self.global_config, [scenario_config], args, random_state=42)

        # Mock the GeneticOptimizer instance and its run method
        mock_ga_instance = MagicMock()
        mock_ga_instance.run.return_value = ({"param1": 10}, 50) # optimal_params, num_evaluations
        mock_genetic_optimizer_class.return_value = mock_ga_instance

        # Mock data (can be simplified as GeneticOptimizer itself is mocked)
        mock_monthly_data = pd.DataFrame({'A': [1,2,3]})
        mock_daily_data = pd.DataFrame({'A': [1,2,3,4,5,6]})
        mock_rets_full = pd.DataFrame({'A': [0.1,0.01,0.02, -0.01, 0.005, 0.003]})

        # We need to ensure that the walk-forward window generation doesn't fail due to insufficient data.
        # Provide enough data for at least one window.
        # train_window_months = 24, test_window_months = 12
        # Need at least 36 months of data.
        long_monthly_idx = pd.date_range(start="2018-01-01", periods=40, freq="ME")
        mock_monthly_data_long = pd.DataFrame(index=long_monthly_idx, data={'PRICE': np.arange(len(long_monthly_idx))})


        optimal_params, num_trials = backtester.run_optimization(
            scenario_config,
            mock_monthly_data_long, # Use data that allows window generation
            mock_daily_data,
            mock_rets_full
        )

        mock_genetic_optimizer_class.assert_called_once_with(
            scenario_config=scenario_config,
            backtester_instance=backtester,
            global_config=self.global_config,
            monthly_data=mock_monthly_data_long,
            daily_data=mock_daily_data,
            rets_full=mock_rets_full,
            random_state=42
        )
        mock_ga_instance.run.assert_called_once()
        self.assertEqual(optimal_params, {"param1": 10})
        self.assertEqual(num_trials, 50)

    @patch('src.portfolio_backtester.backtester_logic.optimization._setup_optuna_study')
    @patch('src.portfolio_backtester.backtester_logic.optimization.Progress') # Mock Progress bar
    @patch('src.portfolio_backtester.backtester.Backtester._get_data_source')
    def test_run_optimization_with_optuna_optimizer(self, mock_get_ds, mock_progress, mock_setup_study):
        """Test that run_optimization calls Optuna logic when specified (or by default)."""
        args = self.MockArgs()
        args.optimizer = "optuna" # Explicitly optuna
        args.mode = "optimize"

        mock_get_ds.return_value = MagicMock()

        scenario_config = self.scenarios[0].copy()
        backtester = Backtester(self.global_config, [scenario_config], args, random_state=42)

        # Mock Optuna study and trial process
        mock_study_instance = MagicMock()
        mock_study_instance.best_trial.params = {"lookback_months": 7}
        mock_study_instance.best_trial.number = 42 # Example trial number
        mock_setup_study.return_value = (mock_study_instance, 10) # study, n_trials

        # Mock evaluation is now handled by the backtester instance method

        # Provide enough data for walk-forward window generation
        # Need at least train_window_months (24) + test_window_months (6) = 30 months minimum
        # Add extra buffer for multiple windows
        long_monthly_idx = pd.date_range(start="2018-01-01", periods=60, freq="ME")
        mock_monthly_data_long = pd.DataFrame(index=long_monthly_idx, data={'PRICE': np.arange(len(long_monthly_idx))})
        mock_daily_data = pd.DataFrame({'A': [1,2,3,4,5,6]}) # Simplified, not directly used by this level of mock
        mock_rets_full = pd.DataFrame({'A': [0.1,0.01,0.02, -0.01, 0.005, 0.003]}) # Simplified

        optimal_params, num_trials, best_trial_obj = backtester.run_optimization(
            scenario_config,
            mock_monthly_data_long,
            mock_daily_data,
            mock_rets_full
        )

        mock_setup_study.assert_called_once()
        mock_study_instance.optimize.assert_called_once()
        self.assertEqual(optimal_params["lookback_months"], 7)
        self.assertEqual(num_trials, 42) # best_trial.number


if __name__ == '__main__':
    unittest.main()
