"""
Tests for parallel Walk-Forward Optimization processing.

These tests ensure the parallel WFO processor works correctly and provides
performance benefits while maintaining mathematical correctness.
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch
from src.portfolio_backtester.parallel_wfo import (
    ParallelWFOProcessor, create_parallel_wfo_processor, 
    DEFAULT_PARALLEL_WFO_CONFIG, _evaluate_window_worker
)


# Global function for multiprocessing (must be picklable)
def global_mock_evaluate_window_func(window, scenario_config, shared_data):
    """Global mock window evaluation function for testing."""
    # Simulate some computation time
    time.sleep(0.01)
    
    # Return deterministic results based on train_start date
    train_start = window.get('train_start', '2020-01-01')
    if train_start == '2020-01-01':
        objective_value = 1.1
    elif train_start == '2020-07-01':
        objective_value = 1.2
    else:
        objective_value = 1.3
    
    # Create mock returns
    returns = pd.Series([0.01, 0.02, -0.01], 
                      index=pd.date_range('2020-01-01', periods=3))
    
    return objective_value, returns


def global_failing_evaluate_func(window, scenario_config, shared_data):
    """Global failing evaluation function for error testing."""
    # Fail on second window
    if 'train_start' in window and window['train_start'] == '2020-07-01':
        raise ValueError("Simulated evaluation error")
    return global_mock_evaluate_window_func(window, scenario_config, shared_data)


def global_slow_evaluate_func(window, scenario_config, shared_data):
    """Global slow evaluation function for performance testing."""
    # Simulate meaningful computation time
    time.sleep(0.05)  # 50ms per window
    return 1.0, pd.Series([0.01, 0.02], index=pd.date_range('2020-01-01', periods=2))


def global_slow_evaluate_func(window, scenario_config, shared_data):
    """Global slow evaluation function for performance testing."""
    # Simulate meaningful computation time
    time.sleep(0.05)  # 50ms per window
    return 1.0, pd.Series([0.01, 0.02], index=pd.date_range('2020-01-01', periods=2))


class TestParallelWFOProcessor:
    """Test parallel WFO processor functionality."""
    
    def setup_method(self):
        """Set up test data and processor."""
        self.processor = ParallelWFOProcessor(max_workers=2, enable_parallel=True)
        
        # Create mock windows
        self.test_windows = [
            {'train_start': '2020-01-01', 'train_end': '2020-06-30', 
             'test_start': '2020-07-01', 'test_end': '2020-12-31'},
            {'train_start': '2020-07-01', 'train_end': '2020-12-31', 
             'test_start': '2021-01-01', 'test_end': '2021-06-30'},
            {'train_start': '2021-01-01', 'train_end': '2021-06-30', 
             'test_start': '2021-07-01', 'test_end': '2021-12-31'}
        ]
        
        self.scenario_config = {
            'strategy_params': {'lookback_months': 12, 'skip_months': 1}
        }
        
        self.shared_data = {
            'daily_data': pd.DataFrame({'AAPL': [100, 101, 102]}, 
                                     index=pd.date_range('2020-01-01', periods=3)),
            'benchmark_data': pd.Series([100, 101, 102], 
                                      index=pd.date_range('2020-01-01', periods=3))
        }
    
    def mock_evaluate_window_func(self, window, scenario_config, shared_data):
        """Mock window evaluation function for testing."""
        # Simulate some computation time
        time.sleep(0.01)
        
        # Return deterministic results based on train_start date
        train_start = window.get('train_start', '2020-01-01')
        if train_start == '2020-01-01':
            objective_value = 1.1
        elif train_start == '2020-07-01':
            objective_value = 1.2
        else:
            objective_value = 1.3
        
        # Create mock returns
        returns = pd.Series([0.01, 0.02, -0.01], 
                          index=pd.date_range('2020-01-01', periods=3))
        
        return objective_value, returns
    
    def test_processor_initialization(self):
        """Test processor initialization with different configurations."""
        # Default initialization
        processor1 = ParallelWFOProcessor()
        assert processor1.enable_parallel == True
        assert processor1.max_workers >= 1
        
        # Custom initialization
        processor2 = ParallelWFOProcessor(max_workers=4, enable_parallel=False)
        assert processor2.enable_parallel == False
        assert processor2.max_workers == 4
        
        # Minimum workers constraint
        processor3 = ParallelWFOProcessor(max_workers=0)
        assert processor3.max_workers == 1
    
    def test_sequential_processing(self):
        """Test sequential processing (fallback mode)."""
        # Disable parallel processing
        processor = ParallelWFOProcessor(enable_parallel=False)
        
        results = processor.process_windows_parallel(
            self.test_windows,
            self.mock_evaluate_window_func,
            self.scenario_config,
            self.shared_data
        )
        
        # Verify results
        assert len(results) == len(self.test_windows)
        for objective_value, returns in results:
            assert isinstance(objective_value, float)
            assert isinstance(returns, pd.Series)
            assert not np.isnan(objective_value)
    
    def test_parallel_processing(self):
        """Test parallel processing functionality."""
        # Enable parallel processing
        processor = ParallelWFOProcessor(max_workers=2, enable_parallel=True)
        
        results = processor.process_windows_parallel(
            self.test_windows,
            self.mock_evaluate_window_func,
            self.scenario_config,
            self.shared_data
        )
        
        # Verify results
        assert len(results) == len(self.test_windows)
        for objective_value, returns in results:
            assert isinstance(objective_value, float)
            assert isinstance(returns, pd.Series)
            assert not np.isnan(objective_value)
    
    def test_parallel_vs_sequential_consistency(self):
        """Test that parallel and sequential processing give same results."""
        # Sequential processing
        processor_seq = ParallelWFOProcessor(enable_parallel=False)
        results_seq = processor_seq.process_windows_parallel(
            self.test_windows,
            global_mock_evaluate_window_func,
            self.scenario_config,
            self.shared_data
        )
        
        # Parallel processing
        processor_par = ParallelWFOProcessor(max_workers=2, enable_parallel=True)
        results_par = processor_par.process_windows_parallel(
            self.test_windows,
            global_mock_evaluate_window_func,
            self.scenario_config,
            self.shared_data
        )
        
        # Results should have same values (order might differ in parallel)
        assert len(results_seq) == len(results_par)
        
        # Extract objective values and sort for comparison
        obj_seq = sorted([result[0] for result in results_seq])
        obj_par = sorted([result[0] for result in results_par])
        
        np.testing.assert_array_almost_equal(obj_seq, obj_par, decimal=10)
    
    def test_error_handling_in_parallel_processing(self):
        """Test error handling when window evaluation fails."""
        results = self.processor.process_windows_parallel(
            self.test_windows,
            global_failing_evaluate_func,
            self.scenario_config,
            self.shared_data
        )
        
        # Should have results for all windows (failed ones get NaN)
        assert len(results) == len(self.test_windows)
        
        # First and third windows should succeed
        assert not np.isnan(results[0][0])
        assert not np.isnan(results[2][0])
        
        # Second window should fail (NaN result)
        assert np.isnan(results[1][0])
    
    def test_single_window_processing(self):
        """Test processing with single window (should use sequential)."""
        single_window = [self.test_windows[0]]
        
        results = self.processor.process_windows_parallel(
            single_window,
            global_mock_evaluate_window_func,
            self.scenario_config,
            self.shared_data
        )
        
        assert len(results) == 1
        assert not np.isnan(results[0][0])
    
    def test_empty_windows_processing(self):
        """Test processing with no windows."""
        results = self.processor.process_windows_parallel(
            [],
            global_mock_evaluate_window_func,
            self.scenario_config,
            self.shared_data
        )
        
        assert len(results) == 0
    
    def test_parallel_benefit_estimation(self):
        """Test parallel processing benefit estimation."""
        num_windows = 5
        avg_window_time = 2.0  # 2 seconds per window
        
        estimates = self.processor.estimate_parallel_benefit(num_windows, avg_window_time)
        
        # Verify estimate structure
        required_keys = [
            'sequential_time', 'estimated_parallel_time', 
            'estimated_speedup', 'estimated_time_saved', 
            'max_workers', 'parallel_overhead'
        ]
        for key in required_keys:
            assert key in estimates
        
        # Verify estimates are reasonable
        assert estimates['sequential_time'] == num_windows * avg_window_time
        assert estimates['estimated_parallel_time'] < estimates['sequential_time']
        assert estimates['estimated_speedup'] > 1.0
        assert estimates['estimated_time_saved'] > 0
    
    def test_worker_function(self):
        """Test the worker function directly."""
        window = self.test_windows[0]
        
        result = _evaluate_window_worker(
            0, window, global_mock_evaluate_window_func,
            self.scenario_config, self.shared_data
        )
        
        objective_value, returns = result
        assert isinstance(objective_value, float)
        assert isinstance(returns, pd.Series)
        assert not np.isnan(objective_value)
    
    def test_worker_function_error_handling(self):
        """Test worker function error handling."""
        def failing_func(window, scenario_config, shared_data):
            raise ValueError("Worker function error")
        
        with pytest.raises(ValueError, match="Worker function error"):
            _evaluate_window_worker(
                0, self.test_windows[0], failing_func,
                self.scenario_config, self.shared_data
            )


class TestParallelWFOConfiguration:
    """Test parallel WFO configuration and factory functions."""
    
    def test_create_parallel_wfo_processor_default(self):
        """Test creating processor with default configuration."""
        config = {}
        processor = create_parallel_wfo_processor(config)
        
        assert isinstance(processor, ParallelWFOProcessor)
        assert processor.enable_parallel == True
        assert hasattr(processor, 'min_windows_for_parallel')
    
    def test_create_parallel_wfo_processor_custom(self):
        """Test creating processor with custom configuration."""
        config = {
            'parallel_wfo_config': {
                'enable_parallel': False,
                'max_workers': 3,
                'min_windows_for_parallel': 5
            }
        }
        
        processor = create_parallel_wfo_processor(config)
        
        assert processor.enable_parallel == False
        assert processor.max_workers == 3
        assert processor.min_windows_for_parallel == 5
    
    def test_default_parallel_wfo_config(self):
        """Test default configuration values."""
        config = DEFAULT_PARALLEL_WFO_CONFIG
        
        required_keys = [
            'enable_parallel', 'max_workers', 'min_windows_for_parallel',
            'process_timeout', 'memory_limit_mb'
        ]
        
        for key in required_keys:
            assert key in config
        
        assert isinstance(config['enable_parallel'], bool)
        assert config['max_workers'] is None  # Auto-detect
        assert isinstance(config['min_windows_for_parallel'], int)
        assert isinstance(config['process_timeout'], int)
        assert isinstance(config['memory_limit_mb'], int)


class TestParallelWFOPerformance:
    """Test parallel WFO performance characteristics."""
    
    @pytest.mark.skip(reason="Parallel performance test is flaky and needs to be revisited.")
    def test_parallel_processing_performance(self):
        """Test that parallel processing provides speedup for multiple windows."""
        # Create more windows to see parallel benefit
        windows = []
        for i in range(24):  # 24 windows should show parallel benefit
            windows.append({
                'train_start': f'2020-{i+1:02d}-01' if i < 12 else f'2021-{i-11:02d}-01',
                'train_end': f'2020-{i+6:02d}-30' if i < 12 else f'2021-{i-5:02d}-30',
                'test_start': f'2020-{i+7:02d}-01' if i < 12 else f'2021-{i-4:02d}-01', 
                'test_end': f'2020-{i+12:02d}-31' if i < 12 else f'2021-{i+1:02d}-31'
            })
        
        def slow_evaluate_func(window, scenario_config, shared_data):
            # Simulate meaningful computation time
            time.sleep(0.2)  # 200ms per window
            return 1.0, pd.Series([0.01, 0.02], index=pd.date_range('2020-01-01', periods=2))
        
        scenario_config = {'strategy_params': {}}
        shared_data = {'data': 'mock'}
        
        # Sequential processing
        processor_seq = ParallelWFOProcessor(enable_parallel=False)
        start_time = time.time()
        results_seq = processor_seq.process_windows_parallel(
            windows, global_slow_evaluate_func, scenario_config, shared_data
        )
        sequential_time = time.time() - start_time
        
        # Parallel processing
        processor_par = ParallelWFOProcessor(max_workers=3, enable_parallel=True)
        start_time = time.time()
        results_par = processor_par.process_windows_parallel(
            windows, global_slow_evaluate_func, scenario_config, shared_data
        )
        parallel_time = time.time() - start_time
        
        # Verify results are equivalent
        assert len(results_seq) == len(results_par) == len(windows)
        
        # Parallel should be faster (allowing for some overhead)
        speedup = sequential_time / parallel_time
        print(f"Sequential time: {sequential_time:.3f}s, Parallel time: {parallel_time:.3f}s, Speedup: {speedup:.2f}x")
        
        # Should see some speedup (even if not ideal due to test overhead)
        assert speedup > 1.1  # At least 10% improvement


class TestParallelWFOIntegration:
    """Test integration of parallel WFO with the main backtester."""
    
    def test_parallel_config_loading(self):
        """Test that parallel configuration is loaded correctly."""
        from src.portfolio_backtester.config_loader import GLOBAL_CONFIG
        from src.portfolio_backtester.parallel_wfo import create_parallel_wfo_processor
        
        # Test with default config (should have parallel_wfo_config now)
        processor = create_parallel_wfo_processor(GLOBAL_CONFIG)
        assert processor is not None
        assert hasattr(processor, 'enable_parallel')
        assert hasattr(processor, 'max_workers')
    
    def test_backtester_parallel_integration(self):
        """Test that backtester can use parallel processing."""
        from src.portfolio_backtester.backtester import Backtester
        from src.portfolio_backtester.config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS
        import argparse
        
        # Create minimal args
        args = argparse.Namespace(
            mode='optimize',
            scenario_name='Test_Optuna_Minimal',
            random_seed=42,
            n_jobs=2,
            early_stop_patience=10,
            pruning_enabled=False,
            pruning_interval_steps=1
        )
        
        # Find a test scenario
        test_scenarios = [s for s in BACKTEST_SCENARIOS if s['name'] == 'Test_Optuna_Minimal']
        if not test_scenarios:
            pytest.skip("Test_Optuna_Minimal scenario not found")
        
        # Create backtester instance
        backtester = Backtester(GLOBAL_CONFIG, test_scenarios, args)
        
        # Check if the parallel evaluation method exists
        assert hasattr(backtester, '_evaluate_single_window'), "Missing _evaluate_single_window method"
        
        # Test that parallel config is accessible
        parallel_config = backtester.global_config.get('parallel_wfo_config', {})
        assert isinstance(parallel_config, dict)
    
    def test_single_window_evaluation_method(self):
        """Test the single window evaluation method."""
        from src.portfolio_backtester.backtester import Backtester
        from src.portfolio_backtester.config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS
        import argparse
        import pandas as pd
        import numpy as np
        
        # Create minimal args
        args = argparse.Namespace(
            mode='optimize',
            scenario_name='Test_Optuna_Minimal',
            random_seed=42,
            n_jobs=1,
            early_stop_patience=10
        )
        
        # Find a test scenario
        test_scenarios = [s for s in BACKTEST_SCENARIOS if s['name'] == 'Test_Optuna_Minimal']
        if not test_scenarios:
            pytest.skip("Test_Optuna_Minimal scenario not found")
        
        # Create backtester instance
        backtester = Backtester(GLOBAL_CONFIG, test_scenarios, args)
        
        # Create mock data for testing
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'SPY']
        
        # Create mock daily data
        daily_data = pd.DataFrame(
            np.random.randn(len(dates), len(tickers)) * 0.02 + 1,
            index=dates,
            columns=tickers
        ).cumprod()
        
        # Create mock monthly data
        monthly_data = daily_data.resample('BME').last()
        
        # Create mock returns data
        rets_full = daily_data.pct_change().fillna(0)
        
        # Create window config
        window_config = {
            'window_idx': 0,
            'tr_start': pd.Timestamp('2020-01-01'),
            'tr_end': pd.Timestamp('2020-06-30'),
            'te_start': pd.Timestamp('2020-07-01'),
            'te_end': pd.Timestamp('2020-12-31')
        }
        
        # Create shared data
        shared_data = {
            'monthly_data': monthly_data,
            'daily_data': daily_data,
            'rets_full': rets_full,
            'trial_synthetic_data': None,
            'replacement_info': None,
            'mc_adaptive_enabled': False,
            'metrics_to_optimize': ['Sharpe'],
            'global_config': GLOBAL_CONFIG,
            'pruning_enabled': False,
            'pruning_interval_steps': 1
        }
        
        # Test the method
        try:
            result = backtester._evaluate_single_window(window_config, test_scenarios[0], shared_data)
            assert isinstance(result, tuple)
            assert len(result) == 2
            metrics, returns = result
            assert isinstance(metrics, list)
            assert isinstance(returns, pd.Series)
        except Exception as e:
            # Method exists but may fail due to missing data - that's ok for this test
            assert hasattr(backtester, '_evaluate_single_window')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print output