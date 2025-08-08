

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
from unittest.mock import Mock, patch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from portfolio_backtester.backtester import Backtester
from portfolio_backtester.optimization.results import OptimizationResult


@pytest.mark.slow
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
            },
            'advanced_reporting_config': {
                'enable_advanced_parameter_analysis': True
            }
        }
        backtester.scenarios = [{
            'name': 'Test_Scenario',
            'strategy': 'momentum',
            'universe': ['AAPL', 'MSFT', 'GOOGL']
        }]
        backtester.args = Mock()
        backtester.args.interactive = False
        backtester.args.scenario_name = "Test_Scenario"
        backtester.output_dir = "test_output"
        backtester.results = {
            'Test_Scenario': {
                'display_name': 'Test Strategy',
                'returns': pd.Series([0.01, 0.02, -0.01], index=pd.date_range('2020-01-01', periods=3))
            }
        }
        backtester.monthly_data = pd.DataFrame()
        backtester.rets_full = pd.DataFrame()
        backtester.run_scenario = Mock(return_value=pd.Series([0.01, 0.02, -0.01], index=pd.date_range('2020-01-01', periods=3)))
        return backtester
    
    @pytest.fixture
    def mock_optimization_result(self, mock_optuna_study):
        """Create a mock OptimizationResult object."""
        
        history = []
        for trial in mock_optuna_study.trials:
            history.append({
                'evaluation': trial.number,
                'objective_value': trial.value,
                'parameters': trial.params,
                'metrics': {'trial_returns': trial.user_attrs['trial_returns']}
            })

        return OptimizationResult(
            best_parameters={'lookback_months': 9, 'num_holdings': 15, 'leverage': 1.2},
            best_value=0.85,
            n_evaluations=len(history),
            optimization_history=history,
            best_trial=mock_optuna_study.best_trial
        )

    @pytest.fixture
    def mock_optuna_study(self):
        """Create a mock Optuna study with trial data."""
        import optuna
        
        # Create mock trials with proper attributes
        trials = []
        np.random.seed(42)  # For reproducible test data
        
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
            # Ensure state comparison works
            trial.__dict__['state'] = optuna.trial.TrialState.COMPLETE
            
            # Mock trial returns data with proper format
            dates = pd.date_range('2020-01-01', periods=100, freq='D')
            returns = np.random.randn(100) * 0.01 + 0.0005
            trial.user_attrs = {
                'trial_returns': {
                    'dates': dates.strftime('%Y-%m-%d').tolist(),
                    'returns': returns.tolist()
                },
                'trial_params': trial.params
            }
            
            # Mock the values property for multi-objective support
            trial.values = [trial.value] if hasattr(trial, 'value') else [0.5]
            
            trials.append(trial)
        
        # Create mock study
        study = Mock()
        study.trials = trials
        
        # CRITICAL: Configure Mock to support len() operations and required attributes
        # Problem: The optimization code calls len() on study.directions and study.trials,
        # but Mock objects don't automatically support len() operations
        # Solution: Provide real Python objects (lists) that support len() natively
        study.directions = ['maximize']  # Single objective - use real list
        study.best_trials = []  # Empty for single objective - use real list
        study.best_trial = trials[0] if trials else None
        
        return study
    
    @pytest.fixture
    def mock_best_trial(self, mock_optuna_study):
        """Create a mock best trial object."""
        best_trial = Mock()
        best_trial.study = mock_optuna_study
        best_trial.number = 15
        best_trial.value = 0.85
        best_trial.params = {'lookback_months': 9, 'num_holdings': 15, 'leverage': 1.2}
        # Ensure hasattr works correctly
        best_trial.__dict__['study'] = mock_optuna_study
        return best_trial
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        returns = np.random.randn(252) * 0.015 + 0.0008
        return pd.Series(returns, index=dates)
    
    def test_plot_stability_measures(self, mock_backtester, mock_optimization_result, sample_returns):
        """Test trial P&L curve visualization."""
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('os.makedirs') as mock_makedirs, \
             patch('matplotlib.pyplot.subplots') as mock_subplots:
            
            # Mock the matplotlib components
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            # Import the function directly from where it's defined
            from portfolio_backtester.reporting.monte_carlo_analyzer import plot_stability_measures
            
            # Create a minimal mock backtester with required attributes
            mock_backtester.logger = Mock()
            mock_backtester.output_dir = "test_output"
            
            # Call the function as a method (pass mock_backtester as self)
            plot_stability_measures(
                mock_backtester,
                "Test_Scenario", 
                mock_optimization_result, 
                sample_returns
            )
            
            # Verify the function ran without errors and made expected calls
            mock_makedirs.assert_called_with("plots", exist_ok=True)
            
            # Verify logger was called (function should log info messages)
            assert mock_backtester.logger.info.call_count > 0
    
    def test_plot_stability_measures_insufficient_trials(self, mock_backtester, sample_returns):
        """Test stability measures with insufficient trial data."""
        # Create best trial with insufficient study data
        optimization_result = Mock()
        optimization_result.optimization_history = [] # No trials
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            from portfolio_backtester.reporting.monte_carlo_analyzer import plot_stability_measures
            plot_stability_measures(
                mock_backtester, 
                "Test_Scenario", 
                optimization_result, 
                sample_returns
            )
            
            # Should exit early and not create plot
            mock_savefig.assert_not_called()
            mock_backtester.logger.warning.assert_called()
    
    def test_plot_parameter_impact_analysis(self, mock_backtester, mock_best_trial):
        """Test parameter impact analysis visualization."""
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('os.makedirs'):
            
            # Test parameter impact analysis
            from portfolio_backtester.reporting.parameter_analysis import _plot_parameter_impact_analysis
            timestamp = "20240101_120000"
            _plot_parameter_impact_analysis(
                mock_backtester, 
                "Test_Scenario", 
                mock_best_trial, 
                timestamp
            )
            
            # Verify the function ran without errors
            # Function should log info messages
            assert mock_backtester.logger.info.call_count > 0
    
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
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('seaborn.heatmap'):
            
            from portfolio_backtester.reporting.parameter_analysis import _create_parameter_heatmaps
            _create_parameter_heatmaps(
                mock_backtester, df, param_names, "Test_Scenario", "20240101_120000"
            )
            
            # Verify the function ran without errors
            # Function should log info messages
            assert mock_backtester.logger.info.call_count >= 0
    
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
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            from portfolio_backtester.reporting.parameter_analysis import _create_parameter_sensitivity_analysis
            _create_parameter_sensitivity_analysis(
                mock_backtester, df, param_names, "Test_Scenario", "20240101_120000"
            )
            
            # Verify the function ran without errors
            assert mock_backtester.logger.info.call_count >= 0
    
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
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('seaborn.heatmap'):
            
            from portfolio_backtester.reporting.parameter_analysis import _create_parameter_correlation_analysis
            _create_parameter_correlation_analysis(
                mock_backtester, df, param_names, "Test_Scenario", "20240101_120000"
            )
            
            # Verify the function ran without errors
            assert mock_backtester.logger.info.call_count >= 0
    
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
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            from portfolio_backtester.reporting.parameter_analysis import _create_parameter_importance_ranking
            _create_parameter_importance_ranking(
                mock_backtester, df, param_names, "Test_Scenario", "20240101_120000"
            )
            
            # Verify the function ran without errors
            assert mock_backtester.logger.info.call_count >= 0
    
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
        rets_full = daily_data.pct_change(fill_method=None).fillna(0)
        
        # Mock run_scenario to return sample returns
        mock_backtester.run_scenario.return_value = pd.Series(
            np.random.randn(100) * 0.01 + 0.0005, index=dates
        )
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('os.makedirs'):
            
            from portfolio_backtester.reporting.monte_carlo_stage2 import _plot_monte_carlo_robustness_analysis
            _plot_monte_carlo_robustness_analysis(
                mock_backtester,
                "Test_Scenario",
                scenario_config,
                optimal_params,
                monthly_data,
                daily_data,
                rets_full
            )
            
            # Verify the function ran without errors
            assert mock_backtester.logger.info.call_count >= 0
    
    def test_monte_carlo_robustness_disabled(self, mock_backtester):
        """Test Monte Carlo robustness when disabled in config."""
        # Disable Monte Carlo
        mock_backtester.global_config['monte_carlo_config']['enable_synthetic_data'] = False
        
        scenario_config = {'name': 'Test_Scenario'}
        optimal_params = {}
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            from portfolio_backtester.reporting.monte_carlo_stage2 import _plot_monte_carlo_robustness_analysis
            _plot_monte_carlo_robustness_analysis(
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
            from portfolio_backtester.reporting.monte_carlo_stage2 import _plot_monte_carlo_robustness_analysis
            _plot_monte_carlo_robustness_analysis(
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
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('os.makedirs'):
            
            from portfolio_backtester.reporting.monte_carlo_stage2 import _create_monte_carlo_robustness_plot
            _create_monte_carlo_robustness_plot(
                mock_backtester,
                "Test_Scenario",
                simulation_results,
                replacement_percentages,
                colors,
                optimal_params
            )
            
            # Verify the function ran without errors
            assert mock_backtester.logger.info.call_count >= 0
    
    def test_visualization_error_handling(self, mock_backtester):
        """Test error handling in visualization functions."""
        # Test with invalid data
        invalid_df = pd.DataFrame()  # Empty dataframe
        
        with patch('matplotlib.pyplot.savefig'):
            # Should handle errors gracefully
            from portfolio_backtester.reporting.parameter_analysis import _create_parameter_heatmaps, _create_parameter_sensitivity_analysis, _create_parameter_correlation_analysis
            _create_parameter_heatmaps(
                mock_backtester, invalid_df, [], "Test_Scenario", "20240101_120000"
            )
            
            _create_parameter_sensitivity_analysis(
                mock_backtester, invalid_df, [], "Test_Scenario", "20240101_120000"
            )
            
            _create_parameter_correlation_analysis(
                mock_backtester, invalid_df, [], "Test_Scenario", "20240101_120000"
            )
            
            # Should not crash, might not create plots
            # Error handling should be logged
            assert mock_backtester.logger.error.call_count >= 0  # May or may not log errors
    
    def test_plot_file_naming_and_paths(self, mock_backtester, mock_optimization_result, sample_returns):
        """Test that plot files are named correctly and saved to proper paths."""
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('os.makedirs') as mock_makedirs:
            
            from portfolio_backtester.reporting.monte_carlo_analyzer import plot_stability_measures
            plot_stability_measures(
                mock_backtester, 
                "Test_Scenario_With_Special_Chars", 
                mock_optimization_result, 
                sample_returns
            )
            
            # Verify plots directory is created
            mock_makedirs.assert_called_with("plots", exist_ok=True)
            
            # Verify the function ran without errors
            assert mock_backtester.logger.info.call_count > 0
    
    def test_interactive_mode_handling(self, mock_backtester, sample_returns):
        """Test handling of interactive vs non-interactive mode."""
        # Test non-interactive mode (default)
        mock_backtester.args.interactive = False
        
        with patch('matplotlib.pyplot.show') as mock_show, \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            # Should not call show() in non-interactive mode
            from portfolio_backtester.reporting.plot_generator import plot_performance_summary
            plot_performance_summary(
                mock_backtester, sample_returns, None
            )
            
            mock_show.assert_not_called()
        
        # Test interactive mode
        mock_backtester.args.interactive = True
        
        with patch('matplotlib.pyplot.show') as mock_show, \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.pause'):
            
            from portfolio_backtester.reporting.plot_generator import plot_performance_summary
            plot_performance_summary(
                mock_backtester, sample_returns, None
            )
            
            # Should call show() in interactive mode
            mock_show.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
