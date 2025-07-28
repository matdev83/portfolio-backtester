"""
Enhanced Backtesting Integration Tests

This module tests the integration of all new backtesting features:
- Two-stage Monte Carlo process
- WFO robustness features
- Advanced reporting and visualization
- Enhanced configuration system
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.portfolio_backtester.backtester import Backtester
from src.portfolio_backtester.utils import generate_randomized_wfo_windows
from src.portfolio_backtester.monte_carlo.asset_replacement import AssetReplacementManager


class TestEnhancedBacktestingIntegration:
    """Test integration of enhanced backtesting features."""
    
    @pytest.fixture
    def enhanced_global_config(self):
        """Enhanced global configuration with all new features."""
        return {
            'data_source': 'yfinance',
            'universe': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            'benchmark': 'SPY',
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
            
            # WFO Robustness Configuration
            'wfo_robustness_config': {
                'enable_window_randomization': True,
                'enable_start_date_randomization': True,
                'train_window_randomization': {
                    'min_offset': 3,
                    'max_offset': 12
                },
                'test_window_randomization': {
                    'min_offset': 2,
                    'max_offset': 8
                },
                'start_date_randomization': {
                    'min_offset': 0,
                    'max_offset': 6
                },
                'stability_metrics': {
                    'enable': True,
                    'worst_percentile': 10,
                    'consistency_threshold': 0.0
                },
                'random_seed': 42
            },
            
            # Monte Carlo Configuration
            'monte_carlo_config': {
                'enable_synthetic_data': True,
                'enable_during_optimization': True,
                'enable_stage2_stress_testing': True,
                'replacement_percentage': 0.05,
                'min_historical_observations': 150,
                'garch_config': {
                    'model_type': 'GARCH',
                    'p': 1,
                    'q': 1,
                    'distribution': 'studentt',
                    'bounds': {
                        'omega': [1e-6, 1.0],
                        'alpha': [0.01, 0.3],
                        'beta': [0.5, 0.99],
                        'nu': [2.1, 30.0]
                    }
                },
                'generation_config': {
                    'buffer_multiplier': 1.2,
                    'max_attempts': 2,
                    'validation_tolerance': 0.3
                },
                'validation_config': {
                    'enable_validation': False,
                    'tolerance': 0.8
                },
                'random_seed': 42
            }
        }
    
    @pytest.fixture
    def enhanced_scenario_config(self):
        """Enhanced scenario configuration with new features."""
        return {
            'name': 'Enhanced_Momentum_Test',
            'strategy': 'momentum',
            'strategy_params': {
                'lookback_months': 6,
                'num_holdings': 10,
                'top_decile_fraction': 0.1,
                'long_only': True,
                'smoothing_lambda': 0.0,
                'leverage': 1.0
            },
            'rebalance_frequency': 'ME',
            'position_sizer': 'equal_weight',
            'transaction_costs_bps': 10,
            'train_window_months': 24,
            'test_window_months': 12,
            'optimization_targets': [
                {'name': 'Sortino', 'direction': 'maximize'},
                {'name': 'Max Drawdown', 'direction': 'minimize'}
            ],
            'optimize': [
                {
                    'parameter': 'lookback_months',
                    'min_value': 3,
                    'max_value': 12,
                    'step': 3
                },
                {
                    'parameter': 'num_holdings',
                    'min_value': 5,
                    'max_value': 15,
                    'step': 5
                }
            ]
        }
    
    @pytest.fixture
    def mock_market_data(self):
        """Create comprehensive mock market data."""
        # Create longer time series for robust testing
        dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
        
        # Generate realistic price movements
        np.random.seed(42)
        returns = np.random.multivariate_normal(
            mean=[0.0008, 0.0007, 0.0009, 0.0012, 0.0006],
            cov=np.array([
                [0.0004, 0.0002, 0.0002, 0.0003, 0.0002],
                [0.0002, 0.0003, 0.0002, 0.0002, 0.0002],
                [0.0002, 0.0002, 0.0005, 0.0003, 0.0002],
                [0.0003, 0.0002, 0.0003, 0.0008, 0.0003],
                [0.0002, 0.0002, 0.0002, 0.0003, 0.0002]
            ]),
            size=len(dates)
        )
        
        # Convert to prices
        prices = 100 * np.exp(np.cumsum(returns, axis=0))
        
        # Create MultiIndex DataFrame (Ticker, Field)
        columns = pd.MultiIndex.from_product(
            [tickers, ['Open', 'High', 'Low', 'Close', 'Volume']],
            names=['Ticker', 'Field']
        )
        
        data = np.zeros((len(dates), len(columns)))
        
        for i, ticker in enumerate(tickers):
            ticker_prices = prices[:, i]
            # Open (with small gap)
            data[:, i*5] = ticker_prices * (1 + np.random.normal(0, 0.001, len(dates)))
            # High
            data[:, i*5+1] = ticker_prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
            # Low
            data[:, i*5+2] = ticker_prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
            # Close
            data[:, i*5+3] = ticker_prices
            # Volume
            data[:, i*5+4] = np.random.randint(1000000, 10000000, len(dates))
        
        return pd.DataFrame(data, index=dates, columns=columns)
    
    def test_end_to_end_enhanced_optimization(self, enhanced_global_config, enhanced_scenario_config, mock_market_data):
        """Test complete enhanced optimization workflow."""
        # Create mock args
        args = Mock()
        args.mode = 'optimize'
        args.optimizer = 'optuna'
        args.optuna_trials = 10
        args.timeout = 1200
        args.pruning_enabled = True
        args.pruning_n_startup_trials = 2
        args.pruning_n_warmup_steps = 0
        args.pruning_interval_steps = 1
        args.n_jobs = 1
        args.early_stop_patience = 5
        args.interactive = False
        
        # Mock data source
        mock_data_source = Mock()
        mock_data_source.get_data.return_value = mock_market_data
        
        with patch.object(Backtester, '_get_data_source', return_value=mock_data_source), \
             patch('src.portfolio_backtester.backtester_logic.optimization.optuna.create_study') as mock_create_study, \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('os.makedirs'):
            
            # Create mock study
            mock_study = Mock()
            mock_trial = Mock()
            mock_trial.params = {'lookback_months': 6, 'num_holdings': 10}
            mock_trial.number = 5
            mock_trial.value = 0.75
            mock_study.best_trial = mock_trial
            mock_study.trials = [mock_trial]
            mock_create_study.return_value = mock_study
            
            # Create backtester
            backtester = Backtester(
                enhanced_global_config, 
                [enhanced_scenario_config], 
                args, 
                random_state=42
            )
            
            # Mock run_scenario to return reasonable results
            with patch.object(backtester, 'run_scenario') as mock_run_scenario:
                dates = pd.date_range('2020-01-01', periods=100, freq='D')
                mock_returns = pd.Series(
                    np.random.randn(100) * 0.01 + 0.0005, 
                    index=dates
                )
                mock_run_scenario.return_value = mock_returns
                
                # Run optimization
                backtester.run_optimize_mode(
                    enhanced_scenario_config,
                    mock_market_data.xs('Close', level='Field', axis=1).resample('ME').last(),
                    mock_market_data,
                    mock_market_data.xs('Close', level='Field', axis=1).pct_change(fill_method=None).fillna(0)
                )
                
                # Verify optimization ran
                assert mock_run_scenario.call_count > 0
                
                # Verify results were stored
                assert len(backtester.results) > 0
                result_key = list(backtester.results.keys())[0]
                assert 'returns' in backtester.results[result_key]
    
    def test_wfo_robustness_integration(self, enhanced_global_config, enhanced_scenario_config):
        """Test WFO robustness features integration."""
        # Create monthly data index
        monthly_index = pd.date_range('2018-01-31', '2023-12-31', freq='ME')
        
        # Test randomized window generation
        windows_set1 = generate_randomized_wfo_windows(
            monthly_index, enhanced_scenario_config, enhanced_global_config, random_state=1
        )
        
        windows_set2 = generate_randomized_wfo_windows(
            monthly_index, enhanced_scenario_config, enhanced_global_config, random_state=2
        )
        
        # Should generate valid windows
        assert len(windows_set1) > 0
        assert len(windows_set2) > 0
        
        # Should be different due to randomization
        assert windows_set1 != windows_set2
        
        # All windows should be valid
        for windows in [windows_set1, windows_set2]:
            for train_start, train_end, test_start, test_end in windows:
                assert train_start <= train_end < test_start <= test_end
                
                # Check minimum window sizes (with tolerance)
                train_months = (train_end - train_start).days / 30.44
                test_months = (test_end - test_start).days / 30.44
                
                assert train_months >= enhanced_scenario_config['train_window_months'] - 2
                assert test_months >= enhanced_scenario_config['test_window_months'] - 2
    
    def test_two_stage_monte_carlo_integration(self, enhanced_global_config):
        """Test two-stage Monte Carlo integration."""
        # Test Stage 1 configuration
        stage1_config = enhanced_global_config['monte_carlo_config'].copy()
        stage1_config['stage1_optimization'] = True
        stage1_config['enable_during_optimization'] = True
        
        # Test Stage 2 configuration
        stage2_config = enhanced_global_config['monte_carlo_config'].copy()
        stage2_config['stage1_optimization'] = False
        stage2_config['enable_stage2_stress_testing'] = True
        
        # Create test data
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        test_data = {}
        for ticker in ['AAPL', 'MSFT']:
            prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.02, axis=0))
            test_data[ticker] = pd.DataFrame({
                'Open': prices * (1 + np.random.normal(0, 0.001, 300)),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.005, 300))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, 300))),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, 300)
            }, index=dates)
        
        universe = ['AAPL', 'MSFT']
        test_start = pd.Timestamp('2020-06-01')
        test_end = pd.Timestamp('2020-12-31')
        
        # Test Stage 1 (should be fast)
        stage1_manager = AssetReplacementManager(stage1_config)
        stage1_data, stage1_info = stage1_manager.create_monte_carlo_dataset(
            test_data, universe, test_start, test_end, "stage1_test", 42
        )
        
        # Test Stage 2 (should be thorough)
        stage2_manager = AssetReplacementManager(stage2_config)
        stage2_data, stage2_info = stage2_manager.create_monte_carlo_dataset(
            test_data, universe, test_start, test_end, "stage2_test", 42
        )
        
        # Both should produce valid results
        assert stage1_data is not None
        assert stage2_data is not None
        assert len(stage1_info.selected_assets) > 0
        assert len(stage2_info.selected_assets) > 0
        
        # Should select same assets with same seed
        assert stage1_info.selected_assets == stage2_info.selected_assets
    
    def test_enhanced_backtester_initialization(self, enhanced_global_config, enhanced_scenario_config):
        """Test backtester initialization with enhanced configuration."""
        args = Mock()
        args.mode = 'backtest'
        args.interactive = False
        
        with patch.object(Backtester, '_get_data_source', return_value=Mock()):
            backtester = Backtester(
                enhanced_global_config, 
                [enhanced_scenario_config], 
                args, 
                random_state=42
            )
            
            # Verify enhanced configuration is loaded
            assert 'wfo_robustness_config' in backtester.global_config
            assert 'monte_carlo_config' in backtester.global_config
            
            # Verify WFO robustness config
            wfo_config = backtester.global_config['wfo_robustness_config']
            assert wfo_config['enable_window_randomization'] == True
            assert wfo_config['enable_start_date_randomization'] == True
            assert 'stability_metrics' in wfo_config
            
            # Verify Monte Carlo config
            mc_config = backtester.global_config['monte_carlo_config']
            assert mc_config['enable_synthetic_data'] == True
            assert mc_config['enable_during_optimization'] == True
            assert mc_config['enable_stage2_stress_testing'] == True
    
    def test_evaluate_params_walk_forward_with_monte_carlo(self, enhanced_global_config, enhanced_scenario_config):
        """Test _evaluate_params_walk_forward with Monte Carlo integration."""
        args = Mock()
        args.pruning_enabled = False
        args.pruning_interval_steps = 1
        
        with patch.object(Backtester, '_get_data_source', return_value=Mock()):
            backtester = Backtester(
                enhanced_global_config, 
                [enhanced_scenario_config], 
                args
            )
            
            # Create test data
            dates = pd.date_range('2020-01-01', periods=100, freq='D')
            monthly_dates = pd.date_range('2020-01-31', periods=12, freq='ME')
            
            daily_data = pd.DataFrame(
                np.random.randn(100, 5), 
                columns=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'],
                index=dates
            )
            monthly_data = pd.DataFrame(
                np.random.randn(12, 5),
                columns=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'],
                index=monthly_dates
            )
            rets_full = daily_data.pct_change(fill_method=None).fillna(0)
            
            # Create windows
            windows = [
                (monthly_dates[0], monthly_dates[5], monthly_dates[6], monthly_dates[8]),
                (monthly_dates[0], monthly_dates[7], monthly_dates[8], monthly_dates[10])
            ]
            
            # Mock trial
            mock_trial = Mock()
            mock_trial.number = 3
            mock_trial.set_user_attr = Mock()
            
            # Mock run_scenario
            with patch.object(backtester, 'run_scenario') as mock_run_scenario:
                mock_run_scenario.return_value = pd.Series(
                    np.random.randn(len(dates)) * 0.01 + 0.0005, 
                    index=dates
                )
                
                # Test evaluation with Monte Carlo
                result = backtester._evaluate_params_walk_forward(
                    mock_trial,
                    enhanced_scenario_config,
                    windows,
                    monthly_data,
                    daily_data,
                    rets_full,
                    ['Sortino', 'Max Drawdown'],
                    True  # Multi-objective
                )
                
                # Should return tuple for multi-objective
                assert isinstance(result, tuple)
                assert len(result) == 2
                
                # Should have called run_scenario for each window
                assert mock_run_scenario.call_count == len(windows)
                
                # Should have set trial attributes
                mock_trial.set_user_attr.assert_called()
    
    def test_stability_metrics_calculation_integration(self, enhanced_global_config):
        """Test stability metrics calculation in integration context."""
        from src.portfolio_backtester.utils import calculate_stability_metrics
        
        # Simulate metric values from multiple WFO windows
        metric_values_per_objective = [
            [0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.6, 0.88, 0.82, 0.77],  # Sortino ratios
            [-0.15, -0.12, -0.18, -0.10, -0.20, -0.08, -0.25, -0.11, -0.14, -0.16]  # Max drawdowns
        ]
        
        metrics_to_optimize = ['Sortino', 'Max Drawdown']
        
        stability_metrics = calculate_stability_metrics(
            metric_values_per_objective, 
            metrics_to_optimize, 
            enhanced_global_config
        )
        
        # Should calculate all stability metrics
        expected_metrics = [
            'stability_Sortino_Std',
            'stability_Sortino_CV',
            'stability_Sortino_Worst_10pct',
            'stability_Sortino_Consistency_Ratio',
            'stability_Max Drawdown_Std',
            'stability_Max Drawdown_CV',
            'stability_Max Drawdown_Worst_10pct',
            'stability_Max Drawdown_Consistency_Ratio'
        ]
        
        for metric in expected_metrics:
            assert metric in stability_metrics
            assert not np.isnan(stability_metrics[metric])
        
        # Verify reasonable values
        assert stability_metrics['stability_Sortino_Std'] > 0
        assert stability_metrics['stability_Sortino_CV'] > 0
        assert 0 <= stability_metrics['stability_Sortino_Consistency_Ratio'] <= 1
    
    def test_configuration_validation_and_defaults(self):
        """Test configuration validation and default handling."""
        # Test with minimal configuration
        minimal_config = {
            'universe': ['AAPL', 'MSFT'],
            'benchmark': 'SPY',
            'wfo_robustness_config': {},
            'monte_carlo_config': {},
        }
        
        scenario_config = {
            'name': 'Minimal_Test',
            'strategy': 'momentum',
            'strategy_params': {}
        }
        
        args = Mock()
        args.mode = 'backtest'
        
        with patch.object(Backtester, '_get_data_source', return_value=Mock()):
            # Should not crash with minimal config
            backtester = Backtester(minimal_config, [scenario_config], args)
            
            # Should have default values for missing configs
            assert 'wfo_robustness_config' in backtester.global_config
            assert 'monte_carlo_config' in backtester.global_config
    
    def test_error_handling_and_recovery(self, enhanced_global_config, enhanced_scenario_config):
        """Test error handling and recovery in enhanced features."""
        args = Mock()
        args.mode = 'optimize'
        args.pruning_enabled = False
        
        with patch.object(Backtester, '_get_data_source', return_value=Mock()):
            backtester = Backtester(
                enhanced_global_config, 
                [enhanced_scenario_config], 
                args
            )
            
            # Test with invalid Monte Carlo data
            invalid_data = {
                'AAPL': pd.DataFrame({'Close': [np.nan, np.nan]})  # All NaN data
            }
            
            # Should handle gracefully
            mc_config = enhanced_global_config['monte_carlo_config']
            manager = AssetReplacementManager(mc_config)
            
            try:
                data, info = manager.create_monte_carlo_dataset(
                    invalid_data, 
                    ['AAPL'], 
                    pd.Timestamp('2020-01-01'),
                    pd.Timestamp('2020-01-02'),
                    "error_test",
                    42
                )
                # If successful, should return something reasonable
                assert data is not None
                assert info is not None
            except Exception as e:
                # If it fails, should be a controlled failure
                assert isinstance(e, (ValueError, RuntimeError))
    
    def test_performance_benchmarks(self, enhanced_global_config):
        """Test performance characteristics of enhanced features."""
        import time
        
        # Test WFO window generation performance
        monthly_index = pd.date_range('2015-01-31', '2023-12-31', freq='ME')
        scenario_config = {
            'train_window_months': 36,
            'test_window_months': 12
        }
        
        # Time window generation
        start_time = time.time()
        windows = generate_randomized_wfo_windows(
            monthly_index, scenario_config, enhanced_global_config, random_state=42
        )
        window_gen_time = time.time() - start_time
        
        # Should be fast (< 1 second for reasonable data size)
        assert window_gen_time < 1.0, f"Window generation too slow: {window_gen_time:.3f}s"
        assert len(windows) > 0
        
        # Test Monte Carlo manager initialization performance
        start_time = time.time()
        mc_config = enhanced_global_config['monte_carlo_config']
        manager = AssetReplacementManager(mc_config)
        init_time = time.time() - start_time
        
        # Should be fast (< 0.5 seconds)
        assert init_time < 0.5, f"Monte Carlo manager initialization too slow: {init_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])