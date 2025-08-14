"""
Unit tests for the optimizer_report_generator module.
"""
import pytest
from pathlib import Path
from portfolio_backtester.reporting.optimizer_report_generator import OptimizerReportGenerator


class TestOptimizerReportGenerator:
    """Test suite for OptimizerReportGenerator."""

    def setup_method(self):
        """Set up the test environment."""
        self.generator = OptimizerReportGenerator(base_reports_dir="test_reports")

    def teardown_method(self):
        """Tear down the test environment."""
        import shutil

        shutil.rmtree("test_reports", ignore_errors=True)

    def test_create_unique_run_directory(self):
        """Test the creation of a unique run directory."""
        run_dir = self.generator.create_unique_run_directory("test_strategy")
        assert isinstance(run_dir, Path)
        assert run_dir.exists()
        assert "test_strategy" in run_dir.name

    def test_interpret_metric(self):
        """Test the interpretation of metrics."""
        rating, explanation = self.generator.interpret_metric("Sharpe", 1.2)
        assert rating == "Good"
        rating, explanation = self.generator.interpret_metric("Max_Drawdown", -0.4)
        assert rating == "High"
        rating, explanation = self.generator.interpret_metric("Unknown_Metric", 1.0)
        assert rating == "Unknown"

    def test_generate_performance_summary_table(self):
        """Test the generation of the performance summary table."""
        metrics = {"Sharpe": 1.2, "Calmar": 1.5, "Max_Drawdown": -0.15}
        table = self.generator.generate_performance_summary_table(metrics)
        assert "| Sharpe | 1.200 | Good | Strong risk-adjusted returns demonstrate effective risk management |" in table
        assert "| Calmar | 1.500 | Good | Strong Calmar ratio demonstrates effective drawdown control |" in table
        assert "| Max Drawdown | -15.00% | Low | Small drawdowns demonstrate good risk control |" in table

    def test_make_json_serializable(self):
        """Test the conversion of non-serializable types to JSON-compatible types."""
        import numpy as np
        import pandas as pd

        data = {
            "int": np.int64(1),
            "float": np.float64(1.0),
            "list": [np.int64(1)],
            "series": pd.Series([1]),
        }
        serializable_data = self.generator._make_json_serializable(data)
        assert isinstance(serializable_data["int"], int)
        assert isinstance(serializable_data["float"], float)
        assert isinstance(serializable_data["list"][0], int)
        assert isinstance(serializable_data["series"], dict)
