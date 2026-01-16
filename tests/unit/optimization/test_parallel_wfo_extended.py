import pytest
import pandas as pd
from unittest.mock import MagicMock
from portfolio_backtester.parallel_wfo import ParallelWFOProcessor, create_parallel_wfo_processor

def test_parallel_wfo_processor_init():
    processor = ParallelWFOProcessor(max_workers=2, enable_parallel=True)
    assert processor.max_workers == 2
    assert processor.enable_parallel is True

def test_process_windows_sequential():
    # Setup mock dependencies
    mock_executor = MagicMock()
    mock_estimator = MagicMock()
    mock_math = MagicMock()
    
    processor = ParallelWFOProcessor(
        enable_parallel=False,
        parallel_executor=mock_executor,
        benefit_estimator=mock_estimator,
        math_operations=mock_math
    )
    
    windows = [{"id": 1}]
    eval_func = MagicMock(return_value=(1.0, pd.Series([0.1])))
    scenario_config = {}
    shared_data = {}
    
    results = processor.process_windows_parallel(windows, eval_func, scenario_config, shared_data)
    
    assert len(results) == 1
    assert results[0][0] == 1.0
    # Sequential processing should NOT call parallel executor
    assert mock_executor.execute_parallel.call_count == 0

def test_process_windows_parallel_injection():
    mock_executor = MagicMock()
    # Mock result format: List[Tuple[Any, pd.Series]]
    mock_executor.execute_parallel.return_value = [(2.0, pd.Series([0.2]))]
    
    processor = ParallelWFOProcessor(
        enable_parallel=True,
        parallel_executor=mock_executor
    )
    
    windows = [{"id": 1}, {"id": 2}]
    eval_func = MagicMock()
    
    results = processor.process_windows_parallel(windows, eval_func, {}, {})
    
    assert mock_executor.execute_parallel.call_count == 1
    assert results[0][0] == 2.0

def test_estimate_parallel_benefit():
    mock_estimator = MagicMock()
    mock_estimator.estimate_parallel_benefit.return_value = {"speedup": 2.0}
    
    processor = ParallelWFOProcessor(benefit_estimator=mock_estimator)
    benefit = processor.estimate_parallel_benefit(10, 5.0)
    
    assert benefit["speedup"] == 2.0
    mock_estimator.estimate_parallel_benefit.assert_called_once_with(10, 5.0)

def test_create_parallel_wfo_processor():
    config = {
        "parallel_wfo_config": {
            "enable_parallel": True,
            "max_workers": 4,
            "min_windows_for_parallel": 5
        }
    }
    processor = create_parallel_wfo_processor(config)
    assert processor.enable_parallel is True
    assert processor.max_workers == 4
    assert processor.min_windows_for_parallel == 5
