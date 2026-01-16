import pytest
from unittest.mock import MagicMock, ANY
import pandas as pd
from typing import Dict, Any, Tuple, List

from portfolio_backtester.parallel_wfo import ParallelWFOProcessor
from portfolio_backtester.interfaces.parallel_executor_interface import IParallelExecutor
from portfolio_backtester.interfaces.parallel_benefit_estimator_interface import IParallelBenefitEstimator
from portfolio_backtester.interfaces.math_operations_interface import IMathOperations

# -------------------------------------------------------------------------
# Mocks & Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def mock_executor():
    executor = MagicMock(spec=IParallelExecutor)
    # Default behavior: return whatever mocked results we want
    return executor

@pytest.fixture
def mock_benefit_estimator():
    estimator = MagicMock(spec=IParallelBenefitEstimator)
    estimator.estimate_parallel_benefit.return_value = {"estimated_speedup": 2.0}
    return estimator

@pytest.fixture
def mock_math_ops():
    ops = MagicMock(spec=IMathOperations)
    ops.max_value.side_effect = max  # Simple implementation
    return ops

@pytest.fixture
def processor(mock_executor, mock_benefit_estimator, mock_math_ops):
    return ParallelWFOProcessor(
        max_workers=4,
        enable_parallel=True,
        parallel_executor=mock_executor,
        benefit_estimator=mock_benefit_estimator,
        math_operations=mock_math_ops
    )

@pytest.fixture
def sample_windows():
    return [
        {"start": "2023-01-01", "end": "2023-01-31"},
        {"start": "2023-02-01", "end": "2023-02-28"}
    ]

@pytest.fixture
def mock_eval_func():
    return MagicMock(return_value=(1.0, pd.Series([0.1, 0.2])))

# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

def test_initialization(processor, mock_executor):
    assert processor.enable_parallel is True
    assert processor.max_workers == 4
    # Check if dependencies are correctly assigned
    assert processor._parallel_executor == mock_executor

def test_process_windows_parallel_success(processor, mock_executor, sample_windows, mock_eval_func):
    # Setup mock executor to return success results
    expected_results = [(1.0, pd.Series([0.1])), (2.0, pd.Series([0.2]))]
    mock_executor.execute_parallel.return_value = expected_results

    # Call method
    results = processor.process_windows_parallel(
        windows=sample_windows,
        evaluate_window_func=mock_eval_func,
        scenario_config={},
        shared_data={}
    )

    # Verify results
    assert results == expected_results
    
    # Verify executor was called
    mock_executor.execute_parallel.assert_called_once()
    
    # Verify arguments passed to execute_parallel
    call_args = mock_executor.execute_parallel.call_args
    assert call_args[0][0] == sample_windows # windows
    assert callable(call_args[0][1]) # worker_func
    assert callable(call_args[0][2]) # fallback_func

def test_process_windows_parallel_fallback_small_batch(processor, mock_executor, mock_eval_func):
    # Only 1 window -> should trigger sequential processing inside logic if min_windows > 1
    # But WindowManager defaults min_windows=2 usually, let's check logic.
    # ParallelWFOProcessor delegates decision to WindowManager.should_use_parallel_processing
    # We need to verify if it skipped executor.execute_parallel.
    
    single_window = [{"start": "2023-01-01"}]
    
    # By default, min_windows_for_parallel is 2 in default config, but WindowManager logic:
    # len(windows) >= min_windows_for_parallel (default 2 inside check method?)
    # Wait, WindowManager.should_use_parallel_processing uses passed min_windows_for_parallel
    # ParallelWFOProcessor calls it.
    
    # Let's inspect ParallelWFOProcessor.process_windows_parallel implementation:
    # It calls self._window_manager.should_use_parallel_processing(windows)
    # The default for that method's arg is 2.
    
    # So 1 window should return False for parallel.
    
    processor.process_windows_parallel(
        windows=single_window,
        evaluate_window_func=mock_eval_func,
        scenario_config={},
        shared_data={}
    )
    
    # Should NOT use parallel executor
    mock_executor.execute_parallel.assert_not_called()
    
    # Should have called eval func directly (via sequential fallback)
    mock_eval_func.assert_called_once()

def test_process_windows_parallel_disabled(mock_executor, mock_math_ops, mock_eval_func, sample_windows):
    # Initialize with enable_parallel=False
    proc = ParallelWFOProcessor(
        enable_parallel=False,
        parallel_executor=mock_executor,
        math_operations=mock_math_ops
    )
    
    proc.process_windows_parallel(
        windows=sample_windows,
        evaluate_window_func=mock_eval_func,
        scenario_config={},
        shared_data={}
    )
    
    # Should NOT use parallel executor
    mock_executor.execute_parallel.assert_not_called()
    
    # Should have called eval func sequentially (2 times)
    assert mock_eval_func.call_count == 2

def test_estimate_parallel_benefit(processor, mock_benefit_estimator):
    processor.estimate_parallel_benefit(num_windows=10, avg_window_time=1.0)
    mock_benefit_estimator.estimate_parallel_benefit.assert_called_with(10, 1.0)

def test_process_windows_sequential_error_handling(processor, mock_eval_func):
    # Test sequential processing exception handling (not raising, but logging/returning nan?)
    # In WindowManager.process_windows_sequential: catches Exception and appends (nan, Series)
    
    # Setup mock to raise exception
    mock_eval_func.side_effect = ValueError("Simulated Failure")
    
    windows = [{"id": 1}]
    
    # We force sequential by using single window
    results = processor.process_windows_parallel(
        windows=windows,
        evaluate_window_func=mock_eval_func,
        scenario_config={},
        shared_data={}
    )
    
    assert len(results) == 1
    val, ser = results[0]
    # Check for NaN (float('nan'))
    import math
    assert math.isnan(val)
    assert ser.empty

def test_create_parallel_wfo_processor_factory():
    from portfolio_backtester.parallel_wfo import create_parallel_wfo_processor
    
    config = {
        "parallel_wfo_config": {
            "enable_parallel": True,
            "max_workers": 2,
            "process_timeout": 60
        }
    }
    
    proc = create_parallel_wfo_processor(config)
    
    assert proc.enable_parallel is True
    assert proc.max_workers == 2
    assert proc.process_timeout == 60
