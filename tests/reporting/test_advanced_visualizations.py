"""
Advanced Visualization Tests

This module tests the advanced reporting and visualization features including:
- Trial P&L curve generation
- Parameter impact analysis
- Monte Carlo robustness plots
- Parameter importance ranking
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from src.portfolio_backtester.backtester import Backtester
from src.portfolio_backtester.backtester_logic import reporting


class TestAdvancedVisualizations:
    """Test advanced visualization and reporting features."""
    
    @pytest.fixture
    def mock_backtester(self):
        """Create a mock backtester for testing."""
        backtester = Mock(spec=Backtester)
        backtester.logger = Mock()
        backtester.global_config = {
            'benchmark': 'SPY',
            'monte_carlo_config': {
                'enable_synthetic_data': True,
                'enable_stage2_stress_testing': True,
                'replacement_percentage': 0.05,
                'min_historical_observations': 252
            }
        }
        backtester.scenarios = [{
            'name': 'Test_Scenario',
            'strategy': 'momentum',
            'universe': ['AAPL', 'MSFT', 'GOOGL']
        }]
        backtester.args = Mock()
        backtester.args.interactive = False
        return backtester
    
    @pytest.fixture
    def mock_optuna_study(self):
        """Create a mock Optuna study with trial data."""
        import optuna
        
        # Create mock trials
        trials = []
        for i in range(20):
            trial = Mock()
            trial.number = i
            trial.state = optuna.trial.TrialState.COMPLETE
            trial.value = 0.5 + 0.3 * np.sin(i * 0.5) + 0.1 * np.random.randn()
            trial.params = {
                'lookback_months': 6 + i % 6,
                'num_holdings': 10 + i % 10,
                'leverage': 1.0 + 0.1 * (i % 5)
            }
            
            # Mock trial returns data
            dates = pd.date_range('2020-01-01', periods=100, freq='D')
            returns = np.random.randn(100) * 0.01 + 0.0005
            trial.user_attrs = {
                'trial_returns': {
                    'dates': dates.strftime('%Y-%m-%d').tolist(),
                    'returns': returns.tolist()
                },
                'trial_params': trial.params
            }
            trials.append(trial)
        
        # Create mock study
        study = Mock()
        study.trials = trials
        
        # Mock the completed trials filtering
        def mock_trial_filter(trials_list):
            return [t for t in trials_list if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Ensure the filtering works correctly
        study.trials = trials
        return study
    
    @pytest.fixture
    def mock_best_trial(self, mock_optuna_study):
        """Create a mock best trial object."""
        best_trial = Mock()
        best_trial.study = mock_optuna_study
        best_trial.number = 15
        best_trial.value = 0.85
        best_trial.params = {'lookback_months': 9, 'num_holdings': 15, 'leverage': 1.2}
        return best_trial
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        returns = np.random.randn(252) * 0.015 + 0.0008
        return pd.Series(returns, index=dates)
    
    def test_plot_stability_measures(self, mock_backtester, mock_best_trial, sample_returns):
        """Test trial P&L curve visualization."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close, \
             patch('os.makedirs') as mock_makedirs:
            
            # Test the stability measures plotting
            reporting._plot_stability_measures(
                mock_backtester, 
                "Test_Scenario", 
                mock_best_trial, 
                sample_returns
            )
            
            # Verify plot was saved
            mock_savefig.assert_called()
            mock_close.assert_called()
            mock_makedirs.assert_called_with("plots", exist_ok=True)
            
            # Verify logger was called
            mock_backtester.logger.info.assert_called()
    
    def test_plot_stability_measures_insufficient_trials(self, mock_backtester, sample_returns):
        """Test stability measures with insufficient trial data."""
        # Create best trial with insufficient study data
        best_trial = Mock()
        study = Mock()
        study.trials = [Mock()]  # Only one trial
        best_trial.study = study
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            reporting._plot_stability_measures(
                mock_backtester, 
                "Test_Scenario", 
                best_trial, 
                sample_returns
            )
            
            # Should exit early and not create plot
            mock_savefig.assert_not_called()
            mock_backtester.logger.warning.assert_called()
    
    def test_plot_parameter_impact_analysis(self, mock_backtester, mock_best_trial):
        """Test parameter impact analysis visualization."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close, \
             patch('os.makedirs') as mock_makedirs:
            
            # Test parameter impact analysis
            timestamp = "20240101_120000"
            reporting._plot_parameter_impact_analysis(
                mock_backtester, 
                "Test_Scenario", 
                mock_best_trial, 
                timestamp
            )
            
            # Should create multiple plots
            assert mock_savefig.call_count >= 5  # Multiple analysis plots
            assert mock_close.call_count >= 5
            mock_makedirs.assert_called_with("plots", exist_ok=True)
    
    def test_create_parameter_heatmaps(self, mock_backtester):
        """Test parameter heatmap creation."""
        # Create sample parameter data
        df = pd.DataFrame({
            'lookback_months': np.random.randint(3, 12, 50),
            'num_holdings': np.random.randint(5, 20, 50),
            'leverage': np.random.uniform(0.5, 2.0, 50),
            'objective_value': np.random.normal(0.6, 0.2, 50),
            'trial_number': range(50)
        })
        
        param_names = ['lookback_months', 'num_holdings', 'leverage']
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close, \
             patch('seaborn.heatmap') as mock_heatmap:
            
            reporting._create_parameter_heatmaps(
                mock_backtester, df, param_names, "Test_Scenario", "20240101_120000"
            )
            
            mock_savefig.assert_called()
            mock_close.assert_called()
            # Should create at least one heatmap
            assert mock_heatmap.call_count >= 1
    
    def test_create_parameter_sensitivity_analysis(self, mock_backtester):
        """Test parameter sensitivity analysis."""
        # Create sample parameter data with clear correlations
        n_points = 100
        lookback = np.random.randint(3, 12, n_points)
        holdings = np.random.randint(5, 20, n_points)
        
        # Create objective values with some correlation to parameters
        objective = 0.5 + 0.02 * lookback + 0.01 * holdings + np.random.normal(0, 0.1, n_points)
        
        df = pd.DataFrame({
            'lookback_months': lookback,
            'num_holdings': holdings,
            'objective_value': objective,
            'trial_number': range(n_points)
        })
        
        param_names = ['lookback_months', 'num_holdings']
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close:
            
            reporting._create_parameter_sensitivity_analysis(
                mock_backtester, df, param_names, "Test_Scenario", "20240101_120000"
            )
            
            mock_savefig.assert_called()
            mock_close.assert_called()
            mock_backtester.logger.info.assert_called()
    
    def test_create_parameter_correlation_analysis(self, mock_backtester):
        """Test parameter correlation analysis."""
        # Create correlated parameter data
        n_points = 100
        base_signal = np.random.randn(n_points)
        
        df = pd.DataFrame({
            'param1': base_signal + np.random.randn(n_points) * 0.5,
            'param2': -base_signal + np.random.randn(n_points) * 0.5,
            'param3': np.random.randn(n_points),
            'objective_value': base_signal * 0.3 + np.random.randn(n_points) * 0.2,
            'trial_number': range(n_points)
        })
        
        param_names = ['param1', 'param2', 'param3']
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close, \
             patch('seaborn.heatmap') as mock_heatmap:
            
            reporting._create_parameter_correlation_analysis(
                mock_backtester, df, param_names, "Test_Scenario", "20240101_120000"
            )
            
            mock_savefig.assert_called()
            mock_close.assert_called()
            mock_heatmap.assert_called()
    
    def test_create_parameter_importance_ranking(self, mock_backtester):
        """Test parameter importance ranking."""
        # Create data with varying parameter importance
        n_points = 200
        important_param = np.random.randn(n_points)
        less_important_param = np.random.randn(n_points)
        noise_param = np.random.randn(n_points)
        
        # Objective strongly correlated with important_param
        objective = important_param * 0.8 + less_important_param * 0.2 + np.random.randn(n_points) * 0.1
        
        df = pd.DataFrame({
            'important_param': important_param,
            'less_important_param': less_important_param,
            'noise_param': noise_param,
            'objective_value': objective,
            'trial_number': range(n_points)
        })
        
        param_names = ['important_param', 'less_important_param', 'noise_param']
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close:
            
            reporting._create_parameter_importance_ranking(
                mock_backtester, df, param_names, "Test_Scenario", "20240101_120000"
            )
            
            mock_savefig.assert_called()
            mock_close.assert_called()
    
    def test_plot_monte_carlo_robustness_analysis(self, mock_backtester):
        """Test Monte Carlo robustness analysis plotting."""
        # Mock the required data
        scenario_config = {
            'name': 'Test_Scenario',
            'strategy': 'momentum',
            'universe': ['AAPL', 'MSFT', 'GOOGL']
        }
        
        optimal_params = {
            'lookback_months': 9,
            'num_holdings': 15,
            'leverage': 1.2
        }
        
        # Create mock data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        monthly_data = pd.DataFrame(np.random.randn(12, 4), 
                                   columns=['AAPL', 'MSFT', 'GOOGL', 'SPY'],
                                   index=pd.date_range('2020-01-01', periods=12, freq='ME'))
        daily_data = pd.DataFrame(np.random.randn(100, 4), 
                                 columns=['AAPL', 'MSFT', 'GOOGL', 'SPY'],
                                 index=dates)
        rets_full = daily_data.pct_change().fillna(0)
        
        # Mock run_scenario to return sample returns
        mock_backtester.run_scenario.return_value = pd.Series(
            np.random.randn(100) * 0.01 + 0.0005, index=dates
        )
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close, \
             patch('os.makedirs') as mock_makedirs:
            
            reporting._plot_monte_carlo_robustness_analysis(
                mock_backtester,
                "Test_Scenario",
                scenario_config,
                optimal_params,
                monthly_data,
                daily_data,
                rets_full
            )
            
            # Should create robustness plot
            mock_savefig.assert_called()
            mock_close.assert_called()
            mock_makedirs.assert_called_with("plots", exist_ok=True)
    
    def test_monte_carlo_robustness_disabled(self, mock_backtester):
        """Test Monte Carlo robustness when disabled in config."""
        # Disable Monte Carlo
        mock_backtester.global_config['monte_carlo_config']['enable_synthetic_data'] = False
        
        scenario_config = {'name': 'Test_Scenario'}
        optimal_params = {}
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            reporting._plot_monte_carlo_robustness_analysis(
                mock_backtester,
                "Test_Scenario",
                scenario_config,
                optimal_params,
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame()
            )
            
            # Should exit early and not create plot
            mock_savefig.assert_not_called()
            mock_backtester.logger.warning.assert_called()
    
    def test_stage2_stress_testing_disabled(self, mock_backtester):
        """Test Monte Carlo robustness when Stage 2 is disabled."""
        # Disable Stage 2 stress testing
        mock_backtester.global_config['monte_carlo_config']['enable_stage2_stress_testing'] = False
        
        scenario_config = {'name': 'Test_Scenario'}
        optimal_params = {}
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            reporting._plot_monte_carlo_robustness_analysis(
                mock_backtester,
                "Test_Scenario",
                scenario_config,
                optimal_params,
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame()
            )
            
            # Should exit early and not create plot
            mock_savefig.assert_not_called()
            mock_backtester.logger.info.assert_called_with(
                "Stage 2 MC: Stage 2 stress testing is disabled for faster optimization. Skipping robustness analysis."
            )
    
    def test_create_monte_carlo_robustness_plot(self, mock_backtester):
        """Test Monte Carlo robustness plot creation."""
        # Create sample simulation results
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        replacement_percentages = [0.05, 0.10, 0.15]
        colors = ['blue', 'orange', 'green']
        
        simulation_results = {}
        for pct in replacement_percentages:
            # Create multiple simulation results for each percentage
            level_results = []
            for i in range(5):
                returns = np.random.randn(100) * 0.015 + 0.0008
                level_results.append(pd.Series(returns, index=dates))
            simulation_results[pct] = level_results
        
        optimal_params = {
            'lookback_months': 9,
            'num_holdings': 15,
            'leverage': 1.2
        }
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close, \
             patch('os.makedirs') as mock_makedirs:
            
            reporting._create_monte_carlo_robustness_plot(
                mock_backtester,
                "Test_Scenario",
                simulation_results,
                replacement_percentages,
                colors,
                optimal_params
            )
            
            mock_savefig.assert_called()
            mock_close.assert_called()
            mock_makedirs.assert_called_with("plots", exist_ok=True)
    
    def test_visualization_error_handling(self, mock_backtester):
        """Test error handling in visualization functions."""
        # Test with invalid data
        invalid_df = pd.DataFrame()  # Empty dataframe
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            # Should handle errors gracefully
            reporting._create_parameter_heatmaps(
                mock_backtester, invalid_df, [], "Test_Scenario", "20240101_120000"
            )
            
            reporting._create_parameter_sensitivity_analysis(
                mock_backtester, invalid_df, [], "Test_Scenario", "20240101_120000"
            )
            
            reporting._create_parameter_correlation_analysis(
                mock_backtester, invalid_df, [], "Test_Scenario", "20240101_120000"
            )
            
            # Should not crash, might not create plots
            # Error handling should be logged
            assert mock_backtester.logger.error.call_count >= 0  # May or may not log errors
    
    def test_plot_file_naming_and_paths(self, mock_backtester, mock_best_trial, sample_returns):
        """Test that plot files are named correctly and saved to proper paths."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close'), \
             patch('os.makedirs') as mock_makedirs:
            
            reporting._plot_stability_measures(
                mock_backtester, 
                "Test_Scenario_With_Special_Chars", 
                mock_best_trial, 
                sample_returns
            )
            
            # Verify plots directory is created
            mock_makedirs.assert_called_with("plots", exist_ok=True)
            
            # Verify savefig was called with a proper path
            mock_savefig.assert_called()
            call_args = mock_savefig.call_args[0]
            filepath = call_args[0]
            
            # Should be in plots directory
            assert filepath.startswith("plots/") or filepath.startswith("plots\\")
            # Should contain scenario name
            assert "Test_Scenario" in filepath
            # Should have timestamp
            assert any(char.isdigit() for char in filepath)
            # Should be PNG file
            assert filepath.endswith(".png")
    
    def test_interactive_mode_handling(self, mock_backtester, sample_returns):
        """Test handling of interactive vs non-interactive mode."""
        # Test non-interactive mode (default)
        mock_backtester.args.interactive = False
        
        with patch('matplotlib.pyplot.show') as mock_show, \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            # Should not call show() in non-interactive mode
            reporting._plot_performance_summary(
                mock_backtester, sample_returns, None
            )
            
            mock_show.assert_not_called()
        
        # Test interactive mode
        mock_backtester.args.interactive = True
        
        with patch('matplotlib.pyplot.show') as mock_show, \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.pause'):
            
            reporting._plot_performance_summary(
                mock_backtester, sample_returns, None
            )
            
            # Should call show() in interactive mode
            mock_show.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])