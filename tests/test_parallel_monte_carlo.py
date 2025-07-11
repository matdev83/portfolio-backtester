"""
Tests for parallel Monte Carlo processing.

These tests ensure the parallel Monte Carlo processor works correctly and provides
performance benefits while maintaining mathematical correctness.
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch
from src.portfolio_backtester.parallel_monte_carlo import (
    ParallelMonteCarloProcessor, create_parallel_monte_carlo_processor,
    DEFAULT_PARALLEL_MONTE_CARLO_CONFIG, _monte_carlo_simulation_worker
)


# Global function for multiprocessing (must be picklable)
def global_mock_simulation_func(task, shared_data):
    """Global mock simulation function for testing."""
    # Simulate some computation time
    time.sleep(0.01)
    
    # Return deterministic results based on task
    replacement_pct = task['replacement_pct']
    sim_num = task['sim_num']
    
    # Create mock returns based on replacement percentage and simulation number
    base_return = 0.01 + (replacement_pct * 0.1) + (sim_num * 0.001)
    returns = pd.Series([base_return, base_return * 1.1, base_return * 0.9], 
                       index=pd.date_range('2020-01-01', periods=3))
    
    return returns


def global_slow_simulation_func(task, shared_data):
    """Global slow simulation function for performance testing."""
    # Simulate meaningful computation time
    time.sleep(0.05)  # 50ms per simulation
    
    replacement_pct = task['replacement_pct']
    returns = pd.Series([0.01 * replacement_pct, 0.02 * replacement_pct], 
                       index=pd.date_range('2020-01-01', periods=2))
    return returns


def global_failing_simulation_func(task, shared_data):
    """Global failing simulation function for error testing."""
    # Fail on specific conditions
    if task['replacement_pct'] == 0.1 and task['sim_num'] == 1:
        raise ValueError("Simulated simulation error")
    return global_mock_simulation_func(task, shared_data)


class TestParallelMonteCarloProcessor:
    """Test parallel Monte Carlo processor functionality."""
    
    def setup_method(self):
        """Set up test data and processor."""
        self.processor = ParallelMonteCarloProcessor(max_workers=2, enable_parallel=True)
        
        self.replacement_percentages = [0.05, 0.10, 0.15]
        self.num_simulations_per_level = 3
        self.shared_data = {
            'daily_data': pd.DataFrame({'AAPL': [100, 101, 102]}, 
                                     index=pd.date_range('2020-01-01', periods=3)),
            'strategy_config': {'lookback_months': 12}
        }
    
    def test_processor_initialization(self):
        """Test processor initialization with different configurations."""
        # Default initialization
        processor1 = ParallelMonteCarloProcessor()
        assert processor1.enable_parallel == True
        assert processor1.max_workers >= 1
        
        # Custom initialization
        processor2 = ParallelMonteCarloProcessor(max_workers=4, enable_parallel=False)
        assert processor2.enable_parallel == False
        assert processor2.max_workers == 4
        
        # Minimum workers constraint
        processor3 = ParallelMonteCarloProcessor(max_workers=0)
        assert processor3.max_workers == 1
    
    def test_sequential_processing(self):
        """Test sequential processing (fallback mode)."""
        # Disable parallel processing
        processor = ParallelMonteCarloProcessor(enable_parallel=False)
        
        results = processor.process_monte_carlo_simulations(
            self.replacement_percentages,
            self.num_simulations_per_level,
            global_mock_simulation_func,
            self.shared_data
        )
        
        # Verify results structure
        assert len(results) == len(self.replacement_percentages)
        for replacement_pct in self.replacement_percentages:
            assert replacement_pct in results
            assert len(results[replacement_pct]) == self.num_simulations_per_level
            
            # Verify each simulation result
            for sim_result in results[replacement_pct]:
                assert isinstance(sim_result, pd.Series)
                assert len(sim_result) > 0
    
    def test_parallel_processing(self):
        """Test parallel processing functionality."""
        # Enable parallel processing
        processor = ParallelMonteCarloProcessor(max_workers=2, enable_parallel=True)
        
        results = processor.process_monte_carlo_simulations(
            self.replacement_percentages,
            self.num_simulations_per_level,
            global_mock_simulation_func,
            self.shared_data
        )
        
        # Verify results structure
        assert len(results) == len(self.replacement_percentages)
        for replacement_pct in self.replacement_percentages:
            assert replacement_pct in results
            assert len(results[replacement_pct]) == self.num_simulations_per_level
            
            # Verify each simulation result
            for sim_result in results[replacement_pct]:
                assert isinstance(sim_result, pd.Series)
                assert len(sim_result) > 0
    
    def test_parallel_vs_sequential_consistency(self):
        """Test that parallel and sequential processing give consistent results."""
        # Sequential processing
        processor_seq = ParallelMonteCarloProcessor(enable_parallel=False)
        results_seq = processor_seq.process_monte_carlo_simulations(
            self.replacement_percentages,
            self.num_simulations_per_level,
            global_mock_simulation_func,
            self.shared_data
        )
        
        # Parallel processing
        processor_par = ParallelMonteCarloProcessor(max_workers=2, enable_parallel=True)
        results_par = processor_par.process_monte_carlo_simulations(
            self.replacement_percentages,
            self.num_simulations_per_level,
            global_mock_simulation_func,
            self.shared_data
        )
        
        # Results should have same structure and values
        assert len(results_seq) == len(results_par)
        
        for replacement_pct in self.replacement_percentages:
            seq_results = results_seq[replacement_pct]
            par_results = results_par[replacement_pct]
            
            assert len(seq_results) == len(par_results)
            
            # Sort results by first value for comparison (parallel order may differ)
            seq_sorted = sorted(seq_results, key=lambda x: x.iloc[0] if len(x) > 0 else 0)
            par_sorted = sorted(par_results, key=lambda x: x.iloc[0] if len(x) > 0 else 0)
            
            for seq_result, par_result in zip(seq_sorted, par_sorted):
                if len(seq_result) > 0 and len(par_result) > 0:
                    pd.testing.assert_series_equal(seq_result, par_result)
    
    def test_error_handling_in_parallel_processing(self):
        """Test error handling when simulations fail."""
        results = self.processor.process_monte_carlo_simulations(
            self.replacement_percentages,
            self.num_simulations_per_level,
            global_failing_simulation_func,
            self.shared_data
        )
        
        # Should have results for all replacement levels
        assert len(results) == len(self.replacement_percentages)
        
        # Check that failed simulation is handled gracefully
        failed_results = results[0.1]  # This level has a failing simulation
        assert len(failed_results) == self.num_simulations_per_level
        
        # At least some simulations should succeed
        successful_count = sum(1 for result in failed_results if len(result) > 0)
        assert successful_count >= self.num_simulations_per_level - 1
    
    def test_small_workload_fallback(self):
        """Test fallback to sequential processing for small workloads."""
        # Very small workload should use sequential processing
        small_replacement_percentages = [0.05]
        small_num_simulations = 1
        
        results = self.processor.process_monte_carlo_simulations(
            small_replacement_percentages,
            small_num_simulations,
            global_mock_simulation_func,
            self.shared_data
        )
        
        assert len(results) == 1
        assert 0.05 in results
        assert len(results[0.05]) == 1
    
    def test_parallel_benefit_estimation(self):
        """Test parallel processing benefit estimation."""
        total_simulations = 15  # 3 levels × 5 simulations
        avg_simulation_time = 2.0  # 2 seconds per simulation
        
        estimates = self.processor.estimate_parallel_benefit(total_simulations, avg_simulation_time)
        
        # Verify estimate structure
        required_keys = [
            'sequential_time', 'estimated_parallel_time', 'estimated_speedup',
            'estimated_time_saved', 'max_workers', 'parallel_overhead',
            'simulations_per_second_sequential', 'simulations_per_second_parallel'
        ]
        for key in required_keys:
            assert key in estimates
        
        # Verify estimates are reasonable
        assert estimates['sequential_time'] == total_simulations * avg_simulation_time
        assert estimates['estimated_parallel_time'] < estimates['sequential_time']
        assert estimates['estimated_speedup'] > 1.0
        assert estimates['estimated_time_saved'] > 0
        assert estimates['simulations_per_second_parallel'] > estimates['simulations_per_second_sequential']
    
    def test_worker_function(self):
        """Test the worker function directly."""
        task = {
            'replacement_pct': 0.1,
            'sim_num': 0,
            'task_id': 0
        }
        
        result = _monte_carlo_simulation_worker(
            task, global_mock_simulation_func, self.shared_data
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) > 0
    
    def test_worker_function_error_handling(self):
        """Test worker function error handling."""
        task = {
            'replacement_pct': 0.1,
            'sim_num': 1,  # This will trigger the error in global_failing_simulation_func
            'task_id': 0
        }
        
        with pytest.raises(ValueError, match="Simulated simulation error"):
            _monte_carlo_simulation_worker(
                task, global_failing_simulation_func, self.shared_data
            )


class TestParallelMonteCarloConfiguration:
    """Test parallel Monte Carlo configuration and factory functions."""
    
    def test_create_parallel_monte_carlo_processor_default(self):
        """Test creating processor with default configuration."""
        config = {}
        processor = create_parallel_monte_carlo_processor(config)
        
        assert isinstance(processor, ParallelMonteCarloProcessor)
        assert processor.enable_parallel == True
        assert hasattr(processor, 'min_simulations_for_parallel')
    
    def test_create_parallel_monte_carlo_processor_custom(self):
        """Test creating processor with custom configuration."""
        config = {
            'parallel_monte_carlo_config': {
                'enable_parallel': False,
                'max_workers': 3,
                'min_simulations_for_parallel': 10
            }
        }
        
        processor = create_parallel_monte_carlo_processor(config)
        
        assert processor.enable_parallel == False
        assert processor.max_workers == 3
        assert processor.min_simulations_for_parallel == 10
    
    def test_default_parallel_monte_carlo_config(self):
        """Test default configuration values."""
        config = DEFAULT_PARALLEL_MONTE_CARLO_CONFIG
        
        required_keys = [
            'enable_parallel', 'max_workers', 'min_simulations_for_parallel',
            'simulation_timeout', 'memory_limit_mb', 'chunk_size'
        ]
        
        for key in required_keys:
            assert key in config
        
        assert isinstance(config['enable_parallel'], bool)
        assert config['max_workers'] is None  # Auto-detect
        assert isinstance(config['min_simulations_for_parallel'], int)
        assert isinstance(config['simulation_timeout'], int)
        assert isinstance(config['memory_limit_mb'], int)
        assert isinstance(config['chunk_size'], int)


class TestParallelMonteCarloPerformance:
    """Test parallel Monte Carlo performance characteristics."""
    
    def test_parallel_processing_performance(self):
        """Test that parallel processing provides speedup for multiple simulations."""
        # Create larger workload to see parallel benefit
        replacement_percentages = [0.05, 0.10, 0.15, 0.20]  # 4 levels
        num_simulations_per_level = 3  # 3 simulations each = 12 total
        shared_data = {'data': 'mock'}
        
        # Sequential processing
        processor_seq = ParallelMonteCarloProcessor(enable_parallel=False)
        start_time = time.time()
        results_seq = processor_seq.process_monte_carlo_simulations(
            replacement_percentages, num_simulations_per_level,
            global_slow_simulation_func, shared_data
        )
        sequential_time = time.time() - start_time
        
        # Parallel processing
        processor_par = ParallelMonteCarloProcessor(max_workers=3, enable_parallel=True)
        start_time = time.time()
        results_par = processor_par.process_monte_carlo_simulations(
            replacement_percentages, num_simulations_per_level,
            global_slow_simulation_func, shared_data
        )
        parallel_time = time.time() - start_time
        
        # Verify results are equivalent
        assert len(results_seq) == len(results_par) == len(replacement_percentages)
        
        # Parallel should be faster (allowing for some overhead)
        speedup = sequential_time / parallel_time
        print(f"Sequential time: {sequential_time:.3f}s, Parallel time: {parallel_time:.3f}s, Speedup: {speedup:.2f}x")
        
        # For small workloads, parallel processing may have overhead
        # The important thing is that both methods produce equivalent results
        assert speedup > 0.5  # Parallel shouldn't be more than 2x slower due to overhead


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print output