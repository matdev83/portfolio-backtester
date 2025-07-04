import unittest
from unittest.mock import patch, MagicMock # Added MagicMock
import pandas as pd
import numpy as np
import pytest
import optuna # Import optuna

from src.portfolio_backtester.backtester import Backtester, _resolve_strategy
from src.portfolio_backtester.config import GLOBAL_CONFIG, BACKTEST_SCENARIOS
from src.portfolio_backtester.feature import get_required_features_from_scenarios
from src.portfolio_backtester.feature_engineering import precompute_features

class TestBacktester(unittest.TestCase):
    class MockArgs: # Define MockArgs as a class attribute
        def __init__(self):
            self.optimize_min_positions = 10
            self.optimize_max_positions = 30
            self.top_n_params = 3
            self.n_jobs = 1
            self.optuna_trials = 10
            self.optuna_timeout_sec = 60
            self.study_name = "test_study"
            self.random_seed = None
            self.storage_url = None
            # Add default for early_stop_patience if Backtester relies on it being present
            self.early_stop_patience = 10

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
        
        # Create mock data
        dates = pd.date_range(start="2020-01-01", periods=48, freq="ME")
        tickers = self.global_config["universe"] + [self.global_config["benchmark"]]
        self.mock_data = pd.DataFrame(
            np.random.randn(48, len(tickers)) / 100 + 0.001,
            index=dates,
            columns=tickers
        )
        # Pre-compute features for the mock data
        strategy_registry = {"momentum": _resolve_strategy("momentum")}
        # Manually add all possible lookback months to required features for testing
        from src.portfolio_backtester.feature import Momentum
        required_features = get_required_features_from_scenarios(self.scenarios, strategy_registry)
        for i in range(1, 13):
            required_features.add(Momentum(lookback_months=i))
        # self.backtester.features = precompute_features(self.mock_data, required_features, self.mock_data[self.global_config["benchmark"]])
        # The above line is problematic for __init__ tests as self.backtester might not be initialized yet or in a controlled way.
        # It's better to set up features specifically for tests that need them.
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
        self.assertIsNone(backtester.features)
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

    @patch('src.portfolio_backtester.data_sources.yfinance_data_source.YFinanceDataSource') # Default
    def test_get_data_source_default_to_yfinance(self, mock_yfinance_data_source_class):
        mock_instance = MagicMock()
        mock_yfinance_data_source_class.return_value = mock_instance

        config = self.global_config.copy()
        if "data_source" in config: # Ensure it's not set for this test
            del config["data_source"]
        args = self.MockArgs()
        with patch('src.portfolio_backtester.backtester.np.random.seed'):
            backtester = Backtester(config, self.scenarios, args)

        self.assertEqual(backtester.data_source, mock_instance)
        mock_yfinance_data_source_class.assert_called_once_with()

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
            "position_sizer": "mock_sizer",
            "transaction_costs_bps": 10
        }

        # Mock data
        dates_daily = pd.date_range(start="2023-01-01", periods=60, freq="B") # Approx 3 months
        dates_monthly = dates_daily.to_period('M').unique().to_timestamp(how='end')

        mock_universe_tickers = ["TICK1", "TICK2"]

        price_data_monthly = pd.DataFrame(np.random.rand(len(dates_monthly), len(mock_universe_tickers)), index=dates_monthly, columns=mock_universe_tickers)
        price_data_daily = pd.DataFrame(np.random.rand(len(dates_daily), len(mock_universe_tickers)), index=dates_daily, columns=mock_universe_tickers)
        # Add benchmark to price data as it's expected by _get_strategy -> get_universe
        price_data_monthly[self.global_config["benchmark"]] = np.random.rand(len(dates_monthly))
        price_data_daily[self.global_config["benchmark"]] = np.random.rand(len(dates_daily))


        # Mock dependencies
        mock_strategy_obj = MagicMock()
        mock_strategy_obj.get_universe.return_value = [(ticker, "EQUITY") for ticker in mock_universe_tickers]
        mock_signals = pd.DataFrame(np.random.rand(len(dates_monthly), len(mock_universe_tickers)), index=dates_monthly, columns=mock_universe_tickers)
        mock_strategy_obj.generate_signals.return_value = mock_signals

        mock_position_sizer_func = MagicMock()
        mock_sized_signals = pd.DataFrame(np.random.rand(len(dates_monthly), len(mock_universe_tickers)) * 0.5, index=dates_monthly, columns=mock_universe_tickers) # e.g. 0.5 weight per stock
        mock_position_sizer_func.return_value = mock_sized_signals

        mock_weights_monthly = pd.DataFrame(0.5, index=dates_monthly, columns=mock_universe_tickers) # Constant 0.5 weights
        mock_weights_monthly.iloc[0] = 0 # Start with no holdings for turnover calc

        # Patching internal calls
        with patch.object(backtester, '_get_strategy', return_value=mock_strategy_obj) as mock_get_strat, \
             patch('src.portfolio_backtester.backtester.get_position_sizer', return_value=mock_position_sizer_func) as mock_get_pos_sizer, \
             patch('src.portfolio_backtester.backtester.rebalance', return_value=mock_weights_monthly) as mock_rebalance_func:

            portfolio_rets = backtester.run_scenario(scenario_config, price_data_monthly, price_data_daily, features={})

            # Assertions
            mock_get_strat.assert_called_once_with(scenario_config["strategy"], scenario_config["strategy_params"])
            mock_strategy_obj.generate_signals.assert_called_once()
            # Check sizer params (dvol_window should be mapped to window)
            expected_sizer_params = {"window": 5}
            mock_get_pos_sizer.assert_called_once_with(scenario_config["position_sizer"])
            mock_position_sizer_func.assert_called_once()
            args, kwargs = mock_position_sizer_func.call_args
            pd.testing.assert_frame_equal(args[0], mock_signals) # signals
            pd.testing.assert_frame_equal(args[1], price_data_monthly[mock_universe_tickers]) # strategy_data_monthly
            # benchmark_data_monthly is passed as a Series from backtester
            pd.testing.assert_series_equal(args[2], price_data_monthly[self.global_config["benchmark"]])
            self.assertEqual(kwargs, expected_sizer_params)


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

            # Check transaction costs effect (very simplified)
            # If there were returns and turnover, net should be less than gross.
            # This requires calculating gross returns separately.
            daily_returns_for_calc = price_data_daily[mock_universe_tickers].pct_change().fillna(0)
            weights_daily_manual = mock_weights_monthly.reindex(price_data_daily.index, method="ffill").shift(1).fillna(0.0)

            # Ensure weights_daily_manual has the same columns as mock_universe_tickers, filling missing with 0
            weights_daily_manual = weights_daily_manual.reindex(columns=mock_universe_tickers, fill_value=0.0)

            gross_returns_manual = (weights_daily_manual * daily_returns_for_calc[mock_universe_tickers]).sum(axis=1)
            turnover_manual = (weights_daily_manual - weights_daily_manual.shift(1)).abs().sum(axis=1).fillna(0.0)
            transaction_costs_manual = turnover_manual * (scenario_config["transaction_costs_bps"] / 10000.0)
            expected_net_returns = gross_returns_manual - transaction_costs_manual

            pd.testing.assert_series_equal(portfolio_rets, expected_net_returns, check_dtype=False, atol=1e-7)


    # Keep the existing test, but we need to ensure self.backtester is initialized for it.
    # One way is to call its setup within the test, or ensure setUp creates a default backtester.
    # For now, let's adapt its setup slightly.
    def test_walk_forward_optimization_runs(self):
        """Test that walk-forward optimization completes and produces results."""
        # Specific setup for this test
        with patch('src.portfolio_backtester.backtester.Backtester._get_data_source') as mock_get_ds:
            mock_get_ds.return_value = MagicMock() # Mock the data source
            backtester_for_wfo = Backtester(self.global_config, self.scenarios, self.mock_args)

        # Pre-compute features for the mock data for this specific backtester instance
        strategy_registry = {"momentum": _resolve_strategy("momentum")}
        from src.portfolio_backtester.feature import Momentum
        required_features = get_required_features_from_scenarios(self.scenarios, strategy_registry)
        for i in range(1, 13): # Using a smaller, more relevant range
            required_features.add(Momentum(lookback_months=i))

        # Mock precompute_features if it's complex or involves external things for this unit-like test
        # For now, assuming it works with mock_data if mock_data is appropriate
        backtester_for_wfo.features = precompute_features(self.mock_data, required_features, self.mock_data[self.global_config["benchmark"]])

        # Pre-calculate returns for the mock data
        rets_full = self.mock_data.pct_change().fillna(0)
        
        # Patch Optuna related calls if they are external or too heavy
        with patch('src.portfolio_backtester.backtester.optuna.create_study'), \
             patch('src.portfolio_backtester.backtester.Progress'), \
             patch('src.portfolio_backtester.backtester.Backtester.run_scenario') as mock_run_scenario:

            # Mock run_scenario to return some dummy returns
            mock_run_scenario.return_value = pd.Series(np.random.randn(100), index=pd.date_range(start="2020-01-01", periods=100))

            backtester_for_wfo._run_optimize_mode(self.scenarios[0], self.mock_data, self.mock_data, rets_full)
        
        self.assertIn("Test_Momentum_WFO (Optimized)", backtester_for_wfo.results)
        result_data = backtester_for_wfo.results["Test_Momentum_WFO (Optimized)"]
        self.assertIsInstance(result_data["returns"], pd.Series)
        self.assertFalse(result_data["returns"].empty)

    # @patch('src.portfolio_backtester.backtester.optuna.create_study') # Do not mock create_study for true integration
    @patch('src.portfolio_backtester.backtester.Backtester.run_scenario') # Mock the scenario runner
    @patch('src.portfolio_backtester.backtester.calculate_metrics') # Mock metrics calculation
    def test_early_stopping_integration(self, mock_calculate_metrics, mock_run_scenario): # mock_create_study removed
        """Test that early stopping halts optimization when criteria are met."""

        # Optuna will create a real study (in-memory by default)
        # We need to ensure that the study object used by the callback to call .stop()
        # is the actual study object. The callback has access to it via its arguments.
        # We can check if study.stop() was called by spying on it if possible,
        # or by checking the number of trials run.

        # Store a reference to the real study object to check its state later if needed,
        # although for GridSampler, n_trials is deterministic if not stopped.
        # The most reliable check is that our mock_metrics_side_effect_new's counter
        # (optuna_trial_idx_for_mock) stops at the expected trial number.

        # --- Setup Scenario Config for Early Stopping ---
        early_stop_scenario = {
            "name": "Test_Early_Stop",
            "strategy": "momentum", # Needs to be a valid strategy for _resolve_strategy
            "strategy_params": {"leverage": 1.0, "smoothing_lambda": 0.5, "long_only": True, "top_decile_fraction": 0.1},
            "rebalance_frequency": "M",
            "position_sizer": "equal_weight",
            "train_window_months": 12, # Short window for faster test
            "test_window_months": 3,
            "optimization_metric": "Sharpe",
            "optimize": [
                {"parameter": "lookback_months", "min_value": 3, "max_value": 9, "step": 1}
            ],
            # Early stopping config
            "early_stopping_enabled": True,
            "early_stopping_metric": "Sharpe", # Explicitly set
            "early_stopping_patience_trials": 2, # Stop after 2 non-improving trials
            "early_stopping_min_delta": 0.01,
            "early_stopping_warmup_trials": 1, # 1 warmup trial (trial 0)
        }

        args = self.MockArgs()
        args.optuna_trials = 10 # Total trials if no early stopping
        args.early_stop_patience = 5 # For zero-return streak, higher than metric patience
        args.random_seed = 12345 # Ensure a fresh study by providing a seed

        # --- Mock run_scenario and calculate_metrics ---
        # run_scenario will be called by the objective function for each walk-forward window
        # We need it to return some dummy pandas Series (portfolio returns)
        # Ensure the index of these returns is compatible with slicing by te_start, te_end
        mock_daily_dates_for_run_scenario_index = pd.date_range(start="2020-01-01", periods=24*22, freq="B") # Approx 2 years for 24 months
        mock_run_scenario.return_value = pd.Series(
            np.random.randn(len(mock_daily_dates_for_run_scenario_index)),
            index=mock_daily_dates_for_run_scenario_index
        )

        # calculate_metrics will be called by the objective function
        # We want to control the "Sharpe" ratio it returns to simulate stalling
        # This list now represents the desired AVERAGE metric value for each Optuna trial
        avg_metric_per_optuna_trial = [
            0.5,    # Optuna Trial 0 (warmup, callback sees this value)
            0.6,    # Optuna Trial 1 (warmup done, new best)
            0.605,  # Optuna Trial 2 (not > 0.6+0.01) -> patience 1
            0.6,    # Optuna Trial 3 (not > 0.6+0.01) -> patience 2, STOP
            0.7,    # Optuna Trial 4 (should not be reached if stop works)
        ]

        optuna_trial_idx_for_mock = 0 # To iterate through avg_metric_per_optuna_trial
        calls_within_current_optuna_trial_objective = 0

        # Based on early_stop_scenario: train_window_m=12, test_window_m=3.
        # With mock_monthly_data (24 periods), this yields 4 walk-forward windows.
        num_windows_per_trial_estimate = 4

        def mock_metrics_side_effect_new(portfolio_rets, benchmark_rets, benchmark_name_str, **kwargs):
            nonlocal optuna_trial_idx_for_mock
            nonlocal calls_within_current_optuna_trial_objective

            # This function is called for each window within an Optuna trial's objective.
            # We make it return the target average value for the current Optuna trial.
            # This way, when the objective averages these, it gets the desired value.
            target_avg_for_current_optuna_trial = avg_metric_per_optuna_trial[optuna_trial_idx_for_mock]

            calls_within_current_optuna_trial_objective += 1
            if calls_within_current_optuna_trial_objective >= num_windows_per_trial_estimate:
                optuna_trial_idx_for_mock += 1 # Move to next Optuna trial's target metric
                calls_within_current_optuna_trial_objective = 0

            return pd.Series({"Sharpe": target_avg_for_current_optuna_trial})

        mock_calculate_metrics.side_effect = mock_metrics_side_effect_new

        # --- Instantiate Backtester ---
        with patch('src.portfolio_backtester.backtester.Backtester._get_data_source') as mock_get_ds:
            mock_get_ds.return_value = MagicMock()
            backtester = Backtester(self.global_config, [early_stop_scenario], args, random_state=123)

        # Prepare mock data (minimal, as run_scenario is mocked)
        # Monthly data is used for walk-forward window generation and feature slicing
        dates_monthly = pd.date_range(start="2020-01-01", periods=24, freq="ME") # Enough for a few windows
        mock_monthly_data = pd.DataFrame(index=dates_monthly, columns=self.global_config["universe"] + [self.global_config["benchmark"]])
        mock_monthly_data = mock_monthly_data.fillna(0.01) # Avoid div by zero if any feature uses pct_change

        # Daily data also needed for the objective function context
        dates_daily = pd.date_range(start="2020-01-01", periods=len(dates_monthly)*22, freq="B")
        mock_daily_data = pd.DataFrame(index=dates_daily, columns=self.global_config["universe"] + [self.global_config["benchmark"]])
        mock_daily_data = mock_daily_data.fillna(0.01)

        mock_rets_full = mock_daily_data.pct_change().fillna(0)

        # Minimal features
        backtester.features = {"mock_feature": pd.Series(index=mock_monthly_data.index, data=0)}


        # --- Run Optimization ---
        # The objective function inside run_optimization will call mock_run_scenario and mock_calculate_metrics
        # The callback inside run_optimization should use the values from mock_calculate_metrics

        # We need to count how many times the objective is called.
        # Optuna calls the objective function for each trial.
        # The callback is called after each trial.
        # We can check if study.stop() was called on the mock_study_instance.

        # Patch the progress bar as it can cause issues in non-interactive test environments
        with patch('src.portfolio_backtester.backtester.Progress'):
            optimal_params, num_actual_trials = backtester.run_optimization(
                early_stop_scenario,
                mock_monthly_data,
                mock_daily_data, # Pass mock daily data
                mock_rets_full   # Pass mock full returns
            )

        # --- Assertions ---
        # Expected trials:
        # Trial 0 (warmup, metric 0.5) -> best = 0.5
        # Trial 1 (warmup done, metric 0.6) -> new best = 0.6, patience_streak = 0
        # Trial 2 (metric 0.605, not > 0.6+0.01) -> best = 0.6, patience_streak = 1
        # Trial 3 (metric 0.6, not > 0.6+0.01) -> best = 0.6, patience_streak = 2 -> STOP
        # So, 4 trials should have run (0, 1, 2, 3). Optuna trials are 0-indexed.
        # The num_actual_trials returned is study.best_trial.number, which isn't directly the count.
        # optuna_trial_idx_for_mock tracks how many Optuna trials' worth of metrics were generated.
        # If trials 0, 1, 2, 3 ran, optuna_trial_idx_for_mock will be 4.

        self.assertEqual(optuna_trial_idx_for_mock, 4, "Optimization should have stopped early after 4 Optuna trials.")
        # mock_study_instance.stop.assert_called_once() # Removed as study is no longer a MagicMock

        # Check that the best_params are still returned correctly based on what Optuna found before stopping.
        # In this controlled test, the "best" trial would be trial 1 with Sharpe 0.6.
        # The `objective` function suggests params, so we can't directly check optimal_params here
        # without also mocking the `trial.suggest_` methods.
        # The key is that the study stopped early.

if __name__ == '__main__':
    unittest.main()
