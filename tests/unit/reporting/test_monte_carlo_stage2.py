import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.portfolio_backtester.reporting.monte_carlo_stage2 import _plot_monte_carlo_robustness_analysis, _create_monte_carlo_robustness_plot

class TestMonteCarloStage2:
    @pytest.fixture
    def mock_backtester(self):
        bt = MagicMock()
        bt.logger = MagicMock()
        bt.global_config = {
            "monte_carlo_config": {
                "enable_synthetic_data": True,
                "enable_stage2_stress_testing": True,
                "num_simulations_per_level": 2
            },
            "universe": ["A", "B"]
        }
        # Mock run_scenario to return dummy returns
        bt.run_scenario.return_value = pd.Series(
            np.random.normal(0, 0.01, 100), 
            index=pd.date_range("2023-01-01", periods=100)
        )
        return bt

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range("2023-01-01", periods=100)
        df = pd.DataFrame(np.random.randn(100, 2), index=dates, columns=["A", "B"])
        return {
            "monthly": df.resample("ME").last(),
            "daily": df,
            "rets": df.pct_change().fillna(0.0)
        }

    @patch("src.portfolio_backtester.reporting.monte_carlo_stage2._create_monte_carlo_robustness_plot")
    def test_plot_monte_carlo_robustness_analysis_execution(self, mock_create_plot, mock_backtester, sample_data):
        _plot_monte_carlo_robustness_analysis(
            mock_backtester,
            "Test Scenario",
            {"universe": ["A", "B"]},
            {}, # optimal_params
            sample_data["monthly"],
            sample_data["daily"],
            sample_data["rets"]
        )
        
        # Verify that simulations ran and plot function was called
        # With 2 sims per level and 3 levels (default base_percentages len 3), total 6 sims
        assert mock_backtester.run_scenario.call_count >= 6
        mock_create_plot.assert_called_once()

    def test_plot_monte_carlo_robustness_disabled(self, mock_backtester, sample_data):
        mock_backtester.global_config["monte_carlo_config"]["enable_synthetic_data"] = False
        
        _plot_monte_carlo_robustness_analysis(
            mock_backtester,
            "Test Scenario",
            {}, {}, sample_data["monthly"], sample_data["daily"], sample_data["rets"]
        )
        
        # Should exit early
        mock_backtester.run_scenario.assert_not_called()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    @patch("os.makedirs")
    def test_create_monte_carlo_robustness_plot(self, mock_makedirs, mock_subplots, mock_savefig, mock_backtester):
        fig = MagicMock()
        ax1 = MagicMock()
        ax2 = MagicMock()
        mock_subplots.return_value = (fig, (ax1, ax2))
        
        simulation_results = {
            0.05: [pd.Series(np.random.randn(10), index=pd.date_range("2023", periods=10))]
        }
        
        _create_monte_carlo_robustness_plot(
            mock_backtester,
            "Test Scenario",
            simulation_results,
            [0.05],
            ["red"],
            {"p": 1},
            pd.Series(np.random.randn(10), index=pd.date_range("2023", periods=10))
        )
        
        mock_makedirs.assert_called()
        mock_savefig.assert_called()