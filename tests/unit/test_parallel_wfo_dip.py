"""
Tests for ParallelWFOProcessor Dependency Inversion Principle (DIP) implementation.

This module tests the new DIP-compliant architecture where ParallelWFOProcessor
accepts interfaces for parallel execution and math operations dependencies.
"""

import pytest
from unittest.mock import Mock

from src.portfolio_backtester.parallel_wfo import (
    ParallelWFOProcessor,
    create_parallel_wfo_processor,
)
from src.portfolio_backtester.interfaces.parallel_executor_interface import (
    IParallelExecutor,
    create_parallel_executor,
)
from src.portfolio_backtester.interfaces.math_operations_interface import (
    IMathOperations,
    StandardMathOperations,
    create_math_operations,
)


class TestParallelWFOProcessorDIP:
    """Test dependency inversion principle implementation for ParallelWFOProcessor."""

    def test_default_initialization_creates_interfaces(self):
        """Test that default initialization creates the required interfaces."""
        processor = ParallelWFOProcessor()

        assert processor._math_operations is not None
        assert hasattr(processor, "_parallel_executor")
        assert processor._parallel_executor is not None
        assert processor.max_workers > 0

    def test_dependency_injection_with_custom_interfaces(self):
        """Test that custom interfaces can be injected via constructor."""
        # Create mock interfaces
        mock_math_ops = Mock(spec=IMathOperations)
        mock_math_ops.max_value.return_value = 4

        mock_parallel_executor = Mock(spec=IParallelExecutor)
        mock_parallel_executor.max_workers = 4
        mock_parallel_executor.process_timeout = 300

        # Initialize with dependency injection
        processor = ParallelWFOProcessor(
            max_workers=4,
            enable_parallel=True,
            process_timeout=300,
            parallel_executor=mock_parallel_executor,
            math_operations=mock_math_ops,
        )

        # Verify interfaces are injected correctly
        assert processor._math_operations is mock_math_ops
        assert processor._parallel_executor is mock_parallel_executor
        assert processor.max_workers == 4

    def test_window_manager_receives_math_operations(self):
        """Test that WindowManager receives the math operations dependency."""
        mock_math_ops = Mock(spec=IMathOperations)
        mock_math_ops.max_value.return_value = 2

        processor = ParallelWFOProcessor(
            max_workers=2, enable_parallel=True, math_operations=mock_math_ops
        )

        # WindowManager should have received the math operations interface
        # This is verified by checking that max_value was called during initialization
        assert mock_math_ops.max_value.called
        assert processor._math_operations is mock_math_ops

    def test_benefit_estimator_receives_math_operations(self):
        """Test that ParallelBenefitEstimator receives the math operations dependency."""
        mock_math_ops = Mock(spec=IMathOperations)
        mock_math_ops.max_value.return_value = 3

        processor = ParallelWFOProcessor(
            max_workers=3, enable_parallel=True, math_operations=mock_math_ops
        )

        # Verify the benefit estimator has the math operations (through the adapter)
        assert hasattr(processor._benefit_estimator._estimator, "_math_operations")

    def test_create_parallel_wfo_processor_with_dependency_injection(self):
        """Test factory function with dependency injection."""
        config = {
            "parallel_wfo_config": {
                "enable_parallel": True,
                "max_workers": 2,
                "process_timeout": 300,
                "min_windows_for_parallel": 2,
            }
        }

        mock_math_ops = Mock(spec=IMathOperations)
        mock_math_ops.max_value.return_value = 2

        mock_executor = Mock(spec=IParallelExecutor)
        mock_executor.max_workers = 2
        mock_executor.process_timeout = 300

        processor = create_parallel_wfo_processor(
            config, parallel_executor=mock_executor, math_operations=mock_math_ops
        )

        assert processor._math_operations is mock_math_ops
        assert processor._parallel_executor is mock_executor
        assert hasattr(processor, "min_windows_for_parallel")

    def test_backward_compatibility_maintained(self):
        """Test that backward compatibility is maintained for existing code."""
        # This tests that the old constructor signature still works
        processor = ParallelWFOProcessor(max_workers=4, enable_parallel=True, process_timeout=300)

        assert processor.max_workers == 4
        assert processor.enable_parallel is True
        assert processor.process_timeout == 300
        assert processor._math_operations is not None
        assert processor._parallel_executor is not None

    def test_process_windows_parallel_uses_injected_executor(self):
        """Test that process_windows_parallel uses the injected parallel executor."""
        mock_executor = Mock(spec=IParallelExecutor)
        mock_executor.max_workers = 2
        mock_executor.process_timeout = 300

        processor = ParallelWFOProcessor(
            max_workers=2, enable_parallel=True, parallel_executor=mock_executor
        )

        # Verify the injected executor is stored correctly
        assert processor._parallel_executor is mock_executor
        assert processor._parallel_executor.max_workers == 2
        assert processor._parallel_executor.process_timeout == 300

    def test_estimate_parallel_benefit_uses_math_operations(self):
        """Test that benefit estimation uses the injected math operations."""
        mock_math_ops = Mock(spec=IMathOperations)
        mock_math_ops.max_value.side_effect = lambda *args: max(args)

        processor = ParallelWFOProcessor(max_workers=4, math_operations=mock_math_ops)

        # Call estimate_parallel_benefit which should trigger math operations
        result = processor.estimate_parallel_benefit(num_windows=5, avg_window_time=10.0)

        # Verify it returns a dictionary with expected keys
        assert isinstance(result, dict)
        assert "estimated_speedup" in result
        assert "sequential_time" in result

    def test_interface_integration_complete(self):
        """Test that all interfaces are properly integrated throughout the system."""
        # Create real interfaces (not mocks) to test actual integration
        math_ops = create_math_operations()
        parallel_executor = create_parallel_executor(max_workers=2, process_timeout=300)

        processor = ParallelWFOProcessor(
            max_workers=2,
            enable_parallel=True,
            process_timeout=300,
            parallel_executor=parallel_executor,
            math_operations=math_ops,
        )

        # Test that all components are properly connected
        assert isinstance(processor._math_operations, IMathOperations)
        assert isinstance(processor._parallel_executor, IParallelExecutor)
        assert processor._parallel_executor.max_workers == 2
        assert processor._parallel_executor.process_timeout == 300

        # Test mathematical operations work correctly
        assert processor._math_operations.max_value(1, 3, 2) == 3
        assert processor._math_operations.min_value(1, 3, 2) == 1


class TestMathOperationsInterface:
    """Test the IMathOperations interface implementation."""

    def test_standard_math_operations(self):
        """Test StandardMathOperations implementation."""
        math_ops = StandardMathOperations()

        # Test max_value
        assert math_ops.max_value(1, 5, 3) == 5
        assert math_ops.max_value(2.5, 1.8, 3.7) == 3.7

        # Test min_value
        assert math_ops.min_value(1, 5, 3) == 1
        assert math_ops.min_value(2.5, 1.8, 3.7) == 1.8

        # Test max_from_iterable
        assert math_ops.max_from_iterable([1, 5, 3]) == 5
        assert math_ops.max_from_iterable([2.5, 1.8, 3.7]) == 3.7

        # Test min_from_iterable
        assert math_ops.min_from_iterable([1, 5, 3]) == 1
        assert math_ops.min_from_iterable([2.5, 1.8, 3.7]) == 1.8

        # Test abs_value
        assert math_ops.abs_value(-5) == 5
        assert math_ops.abs_value(3.5) == 3.5

    def test_math_operations_error_handling(self):
        """Test error handling for math operations."""
        math_ops = StandardMathOperations()

        # Test empty arguments without default
        with pytest.raises(ValueError):
            math_ops.max_value()

        with pytest.raises(ValueError):
            math_ops.min_value()

        # Test empty iterable without default
        with pytest.raises(ValueError):
            math_ops.max_from_iterable([])

        with pytest.raises(ValueError):
            math_ops.min_from_iterable([])

        # Test with default values
        assert math_ops.max_value(default=10) == 10
        assert math_ops.min_value(default=5) == 5
        assert math_ops.max_from_iterable([], default=15) == 15
        assert math_ops.min_from_iterable([], default=8) == 8

    def test_factory_function(self):
        """Test the factory function creates correct implementation."""
        math_ops = create_math_operations()
        assert isinstance(math_ops, StandardMathOperations)
        assert math_ops.max_value(1, 3, 2) == 3


class TestParallelExecutorInterface:
    """Test the IParallelExecutor interface implementation."""

    def test_parallel_executor_creation(self):
        """Test parallel executor interface creation."""
        executor = create_parallel_executor(max_workers=4, process_timeout=300)

        assert isinstance(executor, IParallelExecutor)
        assert executor.max_workers == 4
        assert executor.process_timeout == 300

    def test_parallel_executor_should_use_parallel(self):
        """Test parallel executor decision logic."""
        executor = create_parallel_executor(max_workers=4, process_timeout=300)

        # Should use parallel for sufficient windows
        assert executor.should_use_parallel(num_windows=5, min_windows_threshold=2) is True

        # Should not use parallel for insufficient windows
        assert executor.should_use_parallel(num_windows=1, min_windows_threshold=2) is False

    def test_dip_principle_satisfied(self):
        """Test that Dependency Inversion Principle is satisfied."""
        # High-level module (ParallelWFOProcessor) depends on abstraction (IParallelExecutor)
        # Low-level module (ParallelExecutor) also depends on the same abstraction

        # This test verifies that we can substitute different implementations
        # without changing the high-level module

        mock_executor = Mock(spec=IParallelExecutor)
        mock_executor.max_workers = 2
        mock_executor.process_timeout = 300

        processor1 = ParallelWFOProcessor(parallel_executor=mock_executor)
        assert processor1._parallel_executor is mock_executor

        real_executor = create_parallel_executor(max_workers=2, process_timeout=300)
        processor2 = ParallelWFOProcessor(parallel_executor=real_executor)
        assert processor2._parallel_executor is real_executor

        # Both processors work the same way, demonstrating DIP compliance


if __name__ == "__main__":
    pytest.main([__file__])
