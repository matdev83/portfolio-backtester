# mypy: ignore-errors
"""
Base test classes for integration testing patterns.

This module provides base test classes that eliminate code duplication
in integration tests and provide common integration testing patterns.
"""

import unittest
import pandas as pd
import numpy as np
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock

from tests.fixtures.market_data import MarketDataFixture
from tests.fixtures.strategy_data import StrategyDataFixture
from tests.fixtures.timing_data import TimingDataFixture


class BaseIntegrationTest(unittest.TestCase):
    """
    Base test class for integration testing.

    Provides integration test utilities and common validation methods
    for testing component interactions and end-to-end workflows.
    """

    def setUp(self):
        """Standard integration test setup."""
        # Set random seed for reproducible tests
        np.random.seed(42)

        # Default integration test configuration
        self.default_assets = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        self.default_start_date = "2020-01-01"
        self.default_end_date = "2023-12-31"
        self.default_benchmark = "SPY"

        # Generate comprehensive test data
        self.market_data = MarketDataFixture.create_basic_data(
            tickers=tuple(self.default_assets),
            start_date=self.default_start_date,
            end_date=self.default_end_date,
        )

        self.benchmark_data = MarketDataFixture.create_basic_data(
            tickers=(self.default_benchmark,),
            start_date=self.default_start_date,
            end_date=self.default_end_date,
        )

        # Create test configurations
        self.strategy_config = StrategyDataFixture.momentum_config()
        self.timing_config = TimingDataFixture.time_based_config()

        # Project root for subprocess tests
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # Temporary directory for test outputs
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Standard integration test teardown."""
        # Clean up temporary files
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_full_backtest_config(
        self,
        strategy_name: str = "momentum_strategy",
        assets: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a complete backtest configuration for integration testing.

        Args:
            strategy_name: Name of strategy to test
            assets: List of assets to include
            start_date: Backtest start date
            end_date: Backtest end date
            **kwargs: Additional configuration parameters

        Returns:
            Complete backtest configuration dictionary
        """
        assets = assets or self.default_assets
        start_date = start_date or self.default_start_date
        end_date = end_date or self.default_end_date

        config = {
            "strategy": {"name": strategy_name, "params": self.strategy_config["strategy_params"]},
            "timing": self.timing_config,
            "universe": {"assets": assets, "benchmark": self.default_benchmark},
            "backtest": {
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": 100000.0,
            },
            "output": {"save_results": True, "output_dir": str(self.temp_dir)},
        }

        # Update with any additional parameters
        config.update(kwargs)

        return config

    def run_subprocess_backtest(
        self, command_args: List[str], timeout_sec: int = 300, capture_output: bool = True
    ) -> Tuple[subprocess.CompletedProcess, Optional[str], Optional[str]]:
        """
        Run backtester as subprocess for integration testing.

        Args:
            command_args: Command line arguments for backtester
            timeout_sec: Timeout in seconds
            capture_output: Whether to capture stdout/stderr

        Returns:
            Tuple of (process, stdout, stderr)
        """
        full_command = [sys.executable, "-m", "portfolio_backtester.backtester"] + command_args

        try:
            if capture_output:
                process = subprocess.run(
                    full_command,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    cwd=self.project_root,
                )
                return process, process.stdout, process.stderr
            else:
                process = subprocess.run(full_command, timeout=timeout_sec, cwd=self.project_root)
                return process, None, None

        except subprocess.TimeoutExpired:
            self.fail(f"Subprocess timed out after {timeout_sec} seconds: {' '.join(full_command)}")
        except Exception as e:
            self.fail(f"Subprocess failed: {e}")

    def assert_backtest_results_valid(
        self, results: Dict[str, Any], expected_assets: List[str] = None, min_periods: int = 10
    ):
        """
        Assert that backtest results are valid and complete.

        Args:
            results: Backtest results dictionary
            expected_assets: Expected assets in results
            min_periods: Minimum number of periods expected
        """
        expected_assets = expected_assets or self.default_assets

        # Check that results is a dictionary
        self.assertIsInstance(results, dict, "Results should be a dictionary")

        # Check for required result components
        required_keys = ["returns", "positions", "performance_metrics"]
        for key in required_keys:
            self.assertIn(key, results, f"Results should contain {key}")

        # Validate returns data
        returns = results["returns"]
        self.assertIsInstance(returns, pd.Series, "Returns should be a Series")
        self.assertGreaterEqual(
            len(returns), min_periods, f"Should have at least {min_periods} return periods"
        )

        # Validate positions data
        positions = results["positions"]
        self.assertIsInstance(positions, pd.DataFrame, "Positions should be a DataFrame")
        self.assertGreaterEqual(
            len(positions), min_periods, f"Should have at least {min_periods} position periods"
        )

        # Check that expected assets are in positions
        position_columns = positions.columns.tolist()
        for asset in expected_assets:
            self.assertIn(asset, position_columns, f"Asset {asset} should be in positions")

        # Validate performance metrics
        metrics = results["performance_metrics"]
        self.assertIsInstance(metrics, dict, "Performance metrics should be a dictionary")

        expected_metrics = ["total_return", "annualized_return", "volatility", "sharpe_ratio"]
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Should have {metric} in performance metrics")
            self.assertIsInstance(metrics[metric], (int, float), f"{metric} should be numeric")

    def assert_data_pipeline_integrity(
        self,
        input_data: pd.DataFrame,
        processed_data: pd.DataFrame,
        expected_transformations: List[str] = None,
    ):
        """
        Assert that data pipeline maintains data integrity.

        Args:
            input_data: Original input data
            processed_data: Data after pipeline processing
            expected_transformations: List of expected transformations applied
        """
        # Check that processed data is not empty
        self.assertFalse(processed_data.empty, "Processed data should not be empty")

        # Check that date range is preserved or appropriately modified
        input_start = input_data.index.min()
        input_end = input_data.index.max()
        processed_start = processed_data.index.min()
        processed_end = processed_data.index.max()

        # Processed data should not extend beyond input data range
        self.assertGreaterEqual(
            processed_start, input_start, "Processed data start should be >= input start"
        )
        self.assertLessEqual(processed_end, input_end, "Processed data end should be <= input end")

        # Check for data quality issues
        self.assert_data_quality(processed_data)

    def assert_data_quality(self, data: pd.DataFrame, max_nan_ratio: float = 0.1):
        """
        Assert that data meets quality standards.

        Args:
            data: DataFrame to validate
            max_nan_ratio: Maximum allowed ratio of NaN values
        """
        # Check for excessive NaN values
        total_values = data.size
        nan_count = data.isna().sum().sum()
        nan_ratio = nan_count / total_values if total_values > 0 else 0

        self.assertLessEqual(
            nan_ratio, max_nan_ratio, f"Too many NaN values: {nan_ratio:.2%} > {max_nan_ratio:.2%}"
        )

        # Check for infinite values
        inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        self.assertEqual(inf_count, 0, "Data should not contain infinite values")

        # Check for duplicate index values
        duplicate_count = data.index.duplicated().sum()
        self.assertEqual(duplicate_count, 0, "Data should not have duplicate index values")

    def assert_component_integration(
        self, component_a, component_b, integration_method: str, test_data: Any = None
    ):
        """
        Assert that two components integrate properly.

        Args:
            component_a: First component to test
            component_b: Second component to test
            integration_method: Method name for integration
            test_data: Test data for integration
        """
        # Check that both components exist
        self.assertIsNotNone(component_a, "Component A should not be None")
        self.assertIsNotNone(component_b, "Component B should not be None")

        # Check that integration method exists
        self.assertTrue(
            hasattr(component_a, integration_method) or hasattr(component_b, integration_method),
            f"Integration method {integration_method} should exist",
        )

        # Test integration with provided data
        if test_data is not None:
            try:
                if hasattr(component_a, integration_method):
                    result = getattr(component_a, integration_method)(component_b, test_data)
                else:
                    result = getattr(component_b, integration_method)(component_a, test_data)

                self.assertIsNotNone(result, "Integration should return a result")

            except Exception as e:
                self.fail(f"Component integration failed: {e}")

    def create_mock_data_source(self, data: pd.DataFrame = None) -> Mock:
        """
        Create a mock data source for integration testing.

        Args:
            data: Data to return from mock data source

        Returns:
            Mock data source object
        """
        data = data if data is not None else self.market_data

        mock_source = Mock()
        mock_source.get_data = Mock(return_value=data)
        mock_source.get_benchmark_data = Mock(return_value=self.benchmark_data)
        mock_source.is_available = Mock(return_value=True)
        mock_source.validate_data = Mock(return_value=True)

        return mock_source

    def validate_output_files(
        self, output_dir: Path, expected_files: List[str] = None, check_file_sizes: bool = True
    ):
        """
        Validate that expected output files are created.

        Args:
            output_dir: Directory to check for output files
            expected_files: List of expected file names
            check_file_sizes: Whether to check that files are not empty
        """
        expected_files = expected_files or ["results.csv", "performance_metrics.json"]

        # Check that output directory exists
        self.assertTrue(output_dir.exists(), f"Output directory {output_dir} should exist")

        # Check for expected files
        for filename in expected_files:
            file_path = output_dir / filename
            self.assertTrue(file_path.exists(), f"Expected file {filename} should exist")

            if check_file_sizes:
                file_size = file_path.stat().st_size
                self.assertGreater(file_size, 0, f"File {filename} should not be empty")

    def test_integration_smoke(self):
        """
        Standard smoke test for integration testing.

        This test should be overridden in concrete integration test classes
        with integration-specific implementation.
        """
        self.skipTest(
            "Integration smoke test should be implemented in concrete integration classes"
        )

    def test_end_to_end_workflow_smoke(self):
        """
        Standard smoke test for end-to-end workflow testing.

        This test should be overridden in concrete integration test classes
        with workflow-specific implementation.
        """
        self.skipTest(
            "End-to-end workflow test should be implemented in concrete integration classes"
        )
