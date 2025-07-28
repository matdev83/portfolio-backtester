import unittest
from typing import Optional  # Import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.portfolio_backtester.strategies import enumerate_strategies_with_params
from src.portfolio_backtester.backtester import Backtester
from src.portfolio_backtester.config_loader import GLOBAL_CONFIG
from src.portfolio_backtester.strategies import enumerate_strategies_with_params

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
        
        dates = pd.date_range(start="2018-01-01", periods=72, freq="ME")  # 6 years of data
        tickers = self.global_config["universe"] + [self.global_config["benchmark"]]
        self.mock_data = pd.DataFrame(
            np.random.randn(72, len(tickers)) / 100 + 0.001,
            index=dates,
            columns=tickers
        )
        pass


    @patch('src.portfolio_backtester.backtester.Backtester._get_data_source')
    @patch('numpy.random.seed')
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
    @patch('numpy.random.seed')
    @patch('numpy.random.randint')
    def test_init_random_seed_generation(self, mock_np_randint, mock_np_seed, mock_get_data_source):
        mock_np_randint.return_value = 42 # Mock the generated seed value
        args = self.MockArgs()
        args.random_seed = None # Ensure it's None to trigger generation

        backtester = Backtester(self.global_config, self.scenarios, args, random_state=None) # Explicitly pass None

        mock_np_randint.assert_called_once_with(0, 2**31 - 1)
        mock_np_seed.assert_called_with(42)
        self.assertEqual(backtester.random_state, 42)

    @patch('src.portfolio_backtester.backtester.Backtester._get_data_source')
    @patch('numpy.random.seed')
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

    @patch('numpy.random.seed')
    def test_get_data_source_stooq(self, mock_np_seed, mock_stooq_data_source_class):
        mock_stooq_data_source_class.__name__ = 'StooqDataSource'
        mock_instance = MagicMock()
        mock_stooq_data_source_class.return_value = mock_instance

        config = self.global_config.copy()
        config["data_source"] = "stooq"
        args = self.MockArgs()
        # Temporarily patch np.random.seed for Backtester instantiation
        with patch('numpy.random.seed'):
            backtester = Backtester(config, self.scenarios, args)

        # _get_data_source is called in __init__, so we check the instance attribute
        self.assertEqual(backtester.data_source, mock_instance)
        mock_stooq_data_source_class.assert_called_once_with()


    @patch('numpy.random.seed')
    def test_get_data_source_yfinance(self, mock_np_seed, mock_yfinance_data_source_class):
        mock_yfinance_data_source_class.__name__ = 'YFinanceDataSource'
        mock_instance = MagicMock()
        mock_yfinance_data_source_class.return_value = mock_instance

        config = self.global_config.copy()
        config["data_source"] = "yfinance"
        args = self.MockArgs()
        with patch('numpy.random.seed'):
            backtester = Backtester(config, self.scenarios, args)

        self.assertEqual(backtester.data_source, mock_instance)
        mock_yfinance_data_source_class.assert_called_once_with()

    @patch('src.portfolio_backtester.data_sources.hybrid_data_source.HybridDataSource') # Default
    def test_get_data_source_default_to_hybrid(self, mock_hybrid_data_source_class):
        mock_hybrid_data_source_class.__name__ = 'HybridDataSource'
        mock_instance = MagicMock()
        mock_hybrid_data_source_class.return_value = mock_instance

    @patch('src.portfolio_backtester.data_sources.hybrid_data_source.HybridDataSource')
    def test_get_data_source_hybrid(self, mock_hybrid_data_source_class):
        mock_hybrid_data_source_class.__name__ = 'HybridDataSource'
        mock_instance = MagicMock()
        mock_hybrid_data_source_class.return_value = mock_instance

    def test_get_data_source_unsupported(self):
        config = self.global_config.copy()
        config["data_source"] = "unsupported_source"
        args = self.MockArgs()
        with patch('numpy.random.seed'):
            with self.assertRaisesRegex(ValueError, "Unsupported data source: unsupported_source"):
                Backtester(config, self.scenarios, args)

    # Tests for _get_strategy
    @patch('src.portfolio_backtester.strategies.enumerate_strategies_with_params')
    def test_get_strategy_valid(self, mock_enumerate_strategies):
        mock_strategy_class = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class.return_value = mock_strategy_instance
        mock_enumerate_strategies.return_value = {"momentum": mock_strategy_class}
        params = {"lookback": 10}

        # Need a backtester instance to call _get_strategy on
        # Mock _get_data_source to simplify Backtester instantiation
        with patch.object(Backtester, '_get_data_source', return_value=MagicMock()):
            args = self.MockArgs()
            backtester = Backtester(self.global_config, self.scenarios, args)

        strategy_instance = backtester._get_strategy("momentum", params)

        self.assertEqual(strategy_instance, mock_strategy_instance)
        mock_strategy_class.assert_called_once_with(params)

    @patch('src.portfolio_backtester.strategies.enumerate_strategies_with_params')
    def test_get_strategy_vams_momentum(self, mock_enumerate_strategies):
        mock_strategy_class = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class.return_value = mock_strategy_instance
        mock_enumerate_strategies.return_value = {"vams_momentum": mock_strategy_class}
        params = {"alpha": 0.5}
        with patch.object(Backtester, '_get_data_source', return_value=MagicMock()):
            args = self.MockArgs()
            backtester = Backtester(self.global_config, self.scenarios, args)
        strategy_instance = backtester._get_strategy("vams_momentum", params)
        self.assertEqual(strategy_instance, mock_strategy_instance)
        mock_strategy_class.assert_called_once_with(params)

    @patch('src.portfolio_backtester.strategies.enumerate_strategies_with_params')
    def test_get_strategy_vams_no_downside(self, mock_enumerate_strategies):
        mock_strategy_class = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class.return_value = mock_strategy_instance
        mock_enumerate_strategies.return_value = {"vams_no_downside": mock_strategy_class}
        params = {}
        with patch.object(Backtester, '_get_data_source', return_value=MagicMock()):
            args = self.MockArgs()
            backtester = Backtester(self.global_config, self.scenarios, args)
        strategy_instance = backtester._get_strategy("vams_no_downside", params)
        self.assertEqual(strategy_instance, mock_strategy_instance)
        mock_strategy_class.assert_called_once_with(params)

    def test_get_strategy_unsupported(self):
        params = {}
        with patch.object(Backtester, '_get_data_source', return_value=MagicMock()):
            args = self.MockArgs()
            backtester = Backtester(self.global_config, self.scenarios, args)

        with self.assertRaisesRegex(ValueError, "Unsupported strategy: non_existent_strategy"):
            backtester._get_strategy("non_existent_strategy", params)

if __name__ == '__main__':
    unittest.main()