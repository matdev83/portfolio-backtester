"""
Tests for the OptunaParameterGenerator implementation.

This module tests the OptunaParameterGenerator class that provides
Optuna-based optimization functionality through the ParameterGenerator interface.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Test imports with optional dependency handling
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, NopPruner
    from optuna.study import StudyDirection
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.portfolio_backtester.optimization.generators.optuna_generator import OptunaParameterGenerator
from src.portfolio_backtester.optimization.parameter_generator import (
    ParameterGeneratorError,
    ParameterGeneratorNotInitializedError,
    ParameterGeneratorFinishedError
)
from src.portfolio_backtester.optimization.results import EvaluationResult, OptimizationResult


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
class TestOptunaParameterGeneratorInitialization:
    """Test OptunaParameterGenerator initialization and configuration."""
    
    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        generator = OptunaParameterGenerator()
        
        assert generator.random_state is None
        assert generator.study_name is None
        assert generator.storage_url is None
        assert generator.enable_pruning is True
        assert generator.pruning_config == {}
        assert generator.sampler_config == {}
        assert generator.study is None
        assert not generator._initialized
    
    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        pruning_config = {'n_startup_trials': 10, 'n_warmup_steps': 5}
        sampler_config = {'n_startup_trials': 20}
        
        generator = OptunaParameterGenerator(
            random_state=42,
            study_name="test_study",
            storage_url="sqlite:///test.db",
            enable_pruning=False,
            pruning_config=pruning_config,
            sampler_config=sampler_config
        )
        
        assert generator.random_state == 42
        assert generator.study_name == "test_study"
        assert generator.storage_url == "sqlite:///test.db"
        assert generator.enable_pruning is False
        assert generator.pruning_config == pruning_config
        assert generator.sampler_config == sampler_config
    
    def test_initialization_without_optuna(self):
        """Test that initialization fails gracefully when Optuna is not available."""
        with patch('src.portfolio_backtester.optimization.generators.optuna_generator.OPTUNA_AVAILABLE', False):
            with pytest.raises(ImportError, match="requires 'optuna' package"):
                OptunaParameterGenerator()


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
class TestOptunaParameterGeneratorConfiguration:
    """Test OptunaParameterGenerator configuration and study creation."""
    
    def test_initialize_single_objective(self):
        """Test initialization for single-objective optimization."""
        generator = OptunaParameterGenerator(random_state=42)
        
        scenario_config = {'name': 'test_scenario'}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'param2': {'type': 'int', 'low': 1, 'high': 10}
            },
            'metrics_to_optimize': ['sharpe_ratio'],
            'max_evaluations': 50
        }
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            generator.initialize(scenario_config, optimization_config)
            
            assert generator._initialized is True
            assert generator.parameter_space == optimization_config['parameter_space']
            assert generator.metrics_to_optimize == ['sharpe_ratio']
            assert generator.metric_directions == ['maximize']
            assert generator.is_multi_objective is False
            assert generator.max_evaluations == 50
            assert generator.study == mock_study
            
            # Verify study creation parameters
            mock_create_study.assert_called_once()
            call_kwargs = mock_create_study.call_args[1]
            assert call_kwargs['direction'] == StudyDirection.MAXIMIZE
            assert 'directions' not in call_kwargs
    
    def test_initialize_multi_objective(self):
        """Test initialization for multi-objective optimization."""
        generator = OptunaParameterGenerator(random_state=42)
        
        scenario_config = {'name': 'test_scenario'}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },
            'optimization_targets': [
                {'name': 'sharpe_ratio', 'direction': 'maximize'},
                {'name': 'max_drawdown', 'direction': 'minimize'}
            ],
            'max_evaluations': 100
        }
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            generator.initialize(scenario_config, optimization_config)
            
            assert generator.is_multi_objective is True
            assert generator.metrics_to_optimize == ['sharpe_ratio', 'max_drawdown']
            assert generator.metric_directions == ['maximize', 'minimize']
            
            # Verify study creation parameters
            mock_create_study.assert_called_once()
            call_kwargs = mock_create_study.call_args[1]
            assert call_kwargs['directions'] == [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
            assert 'direction' not in call_kwargs
    
    def test_initialize_with_invalid_directions(self):
        """Test initialization handles invalid optimization directions."""
        generator = OptunaParameterGenerator()
        
        optimization_config = {
            'optimization_targets': [
                {'name': 'metric1', 'direction': 'invalid_direction'},
                {'name': 'metric2', 'direction': 'minimize'}
            ],
            'metrics_to_optimize': ['metric1', 'metric2']
        }
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            generator.initialize({}, optimization_config)
            
            # Invalid direction should be corrected to 'maximize'
            assert generator.metric_directions == ['maximize', 'minimize']
    
    def test_study_name_generation(self):
        """Test automatic study name generation."""
        generator = OptunaParameterGenerator(random_state=42)
        
        scenario_config = {'name': 'momentum_strategy'}
        optimization_config = {'metrics_to_optimize': ['sharpe_ratio']}
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            generator.initialize(scenario_config, optimization_config)
            
            expected_name = "momentum_strategy_optuna_seed_42"
            assert generator.study_name == expected_name
            
            call_kwargs = mock_create_study.call_args[1]
            assert call_kwargs['study_name'] == expected_name
    
    def test_pruning_configuration(self):
        """Test pruning configuration."""
        # Test with pruning enabled
        generator = OptunaParameterGenerator(
            enable_pruning=True,
            pruning_config={'n_startup_trials': 10, 'n_warmup_steps': 5}
        )
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            generator.initialize({}, {'metrics_to_optimize': ['sharpe_ratio']})
            
            call_kwargs = mock_create_study.call_args[1]
            pruner = call_kwargs['pruner']
            assert isinstance(pruner, MedianPruner)
        
        # Test with pruning disabled
        generator = OptunaParameterGenerator(enable_pruning=False)
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            generator.initialize({}, {'metrics_to_optimize': ['sharpe_ratio']})
            
            call_kwargs = mock_create_study.call_args[1]
            pruner = call_kwargs['pruner']
            assert isinstance(pruner, NopPruner)
    
    def test_storage_configuration(self):
        """Test storage configuration."""
        generator = OptunaParameterGenerator(storage_url="sqlite:///test.db")
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            generator.initialize({}, {'metrics_to_optimize': ['sharpe_ratio']})
            
            call_kwargs = mock_create_study.call_args[1]
            assert call_kwargs['storage'] == "sqlite:///test.db"
    
    def test_initialize_validation_error(self):
        """Test initialization with invalid configuration."""
        generator = OptunaParameterGenerator()
        
        # Invalid parameter space
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'invalid_type'}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        with pytest.raises(ParameterGeneratorError):
            generator.initialize({}, optimization_config)


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
class TestOptunaParameterGeneratorSuggestion:
    """Test parameter suggestion functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = OptunaParameterGenerator(random_state=42)
        self.scenario_config = {'name': 'test_scenario'}
        self.optimization_config = {
            'parameter_space': {
                'float_param': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'int_param': {'type': 'int', 'low': 1, 'high': 10, 'step': 2},
                'float_step_param': {'type': 'float', 'low': 0.1, 'high': 0.9, 'step': 0.1},
                'log_param': {'type': 'float', 'low': 1e-5, 'high': 1e-1, 'log': True},
                'cat_param': {'type': 'categorical', 'choices': ['A', 'B', 'C']}
            },
            'metrics_to_optimize': ['sharpe_ratio'],
            'max_evaluations': 50
        }
    
    def test_suggest_parameters_not_initialized(self):
        """Test that suggesting parameters fails when not initialized."""
        with pytest.raises(ParameterGeneratorNotInitializedError):
            self.generator.suggest_parameters()
    
    def test_suggest_parameters_when_finished(self):
        """Test that suggesting parameters fails when optimization is finished."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            self.generator.initialize(self.scenario_config, self.optimization_config)
            
            # Mock finished state
            self.generator.current_evaluation = self.generator.max_evaluations
            
            with pytest.raises(ParameterGeneratorFinishedError):
                self.generator.suggest_parameters()
    
    def test_suggest_parameters_float_type(self):
        """Test parameter suggestion for float parameters."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_trial = Mock()
            mock_study.ask.return_value = mock_trial
            mock_create_study.return_value = mock_study
            
            # Configure mock trial suggestions
            mock_trial.suggest_float.side_effect = [0.5, 0.3, 1e-3]
            mock_trial.suggest_int.return_value = 5
            mock_trial.suggest_categorical.return_value = 'B'
            
            self.generator.initialize(self.scenario_config, self.optimization_config)
            parameters = self.generator.suggest_parameters()
            
            assert 'float_param' in parameters
            assert 'float_step_param' in parameters
            assert 'log_param' in parameters
            assert 'int_param' in parameters
            assert 'cat_param' in parameters
            
            # Verify correct suggest methods were called
            assert mock_trial.suggest_float.call_count == 3
            assert mock_trial.suggest_int.call_count == 1
            assert mock_trial.suggest_categorical.call_count == 1
            
            # Verify parameters for float_param (no step, no log)
            float_call = mock_trial.suggest_float.call_args_list[0]
            assert float_call[0] == ('float_param', 0.0, 1.0)
            assert float_call[1] == {'log': False}
            
            # Verify parameters for float_step_param (with step)
            step_call = mock_trial.suggest_float.call_args_list[1]
            assert step_call[0] == ('float_step_param', 0.1, 0.9)
            assert step_call[1] == {'step': 0.1, 'log': False}
            
            # Verify parameters for log_param (with log)
            log_call = mock_trial.suggest_float.call_args_list[2]
            assert log_call[0] == ('log_param', 1e-5, 1e-1)
            assert log_call[1] == {'log': True}
            
            # Verify parameters for int_param
            int_call = mock_trial.suggest_int.call_args_list[0]
            assert int_call[0] == ('int_param', 1, 10)
            assert int_call[1] == {'step': 2, 'log': False}
            
            # Verify parameters for cat_param
            cat_call = mock_trial.suggest_categorical.call_args_list[0]
            assert cat_call[0] == ('cat_param', ['A', 'B', 'C'])
    
    def test_suggest_parameters_invalid_type(self):
        """Test parameter suggestion with invalid parameter type."""
        invalid_config = {
            'parameter_space': {
                'invalid_param': {'type': 'invalid_type'}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_trial = Mock()
            mock_study.ask.return_value = mock_trial
            mock_create_study.return_value = mock_study
            
            self.generator.initialize({}, invalid_config)
            
            with pytest.raises(ParameterGeneratorError, match="Unsupported parameter type"):
                self.generator.suggest_parameters()
    
    def test_suggest_parameters_trial_creation_error(self):
        """Test error handling when trial creation fails."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_study.ask.side_effect = Exception("Trial creation failed")
            mock_create_study.return_value = mock_study
            
            self.generator.initialize(self.scenario_config, self.optimization_config)
            
            with pytest.raises(ParameterGeneratorError, match="Failed to suggest parameters"):
                self.generator.suggest_parameters()


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
class TestOptunaParameterGeneratorReporting:
    """Test result reporting functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = OptunaParameterGenerator(random_state=42)
        self.scenario_config = {'name': 'test_scenario'}
        self.optimization_config = {
            'parameter_space': {
                'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },
            'metrics_to_optimize': ['sharpe_ratio'],
            'max_evaluations': 50
        }
    
    def test_report_result_not_initialized(self):
        """Test that reporting results fails when not initialized."""
        result = EvaluationResult(objective_value=1.0, metrics={}, window_results=[])
        
        with pytest.raises(ParameterGeneratorNotInitializedError):
            self.generator.report_result({}, result)
    
    def test_report_result_no_current_trial(self):
        """Test that reporting results fails when no current trial exists."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            self.generator.initialize(self.scenario_config, self.optimization_config)
            
            result = EvaluationResult(objective_value=1.0, metrics={}, window_results=[])
            
            with pytest.raises(ValueError, match="No current trial"):
                self.generator.report_result({}, result)
    
    def test_report_result_single_objective(self):
        """Test reporting results for single-objective optimization."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_trial = Mock()
            mock_trial.number = 0
            mock_trial.state = Mock()
            mock_trial.state.name = 'COMPLETE'
            mock_study.ask.return_value = mock_trial
            mock_create_study.return_value = mock_study
            
            self.generator.initialize(self.scenario_config, self.optimization_config)
            
            # Suggest parameters to create current trial
            parameters = self.generator.suggest_parameters()
            
            # Report result
            result = EvaluationResult(
                objective_value=1.5,
                metrics={'sharpe_ratio': 1.5, 'max_drawdown': -0.1},
                window_results=[]
            )
            
            self.generator.report_result(parameters, result)
            
            # Verify study.tell was called correctly
            mock_study.tell.assert_called_once_with(mock_trial, 1.5)
            
            # Verify state updates
            assert self.generator.current_evaluation == 1
            assert len(self.generator.optimization_history) == 1
            assert self.generator._current_trial is None
            
            # Verify history entry
            history_entry = self.generator.optimization_history[0]
            assert history_entry['evaluation'] == 1
            assert history_entry['trial_number'] == 0
            assert history_entry['parameters'] == parameters
            assert history_entry['objective_value'] == 1.5
            assert history_entry['metrics'] == {'sharpe_ratio': 1.5, 'max_drawdown': -0.1}
    
    def test_report_result_single_objective_list_value(self):
        """Test reporting results with list objective value for single-objective."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_trial = Mock()
            mock_trial.number = 0
            mock_trial.state = Mock()
            mock_trial.state.name = 'COMPLETE'
            mock_study.ask.return_value = mock_trial
            mock_create_study.return_value = mock_study
            
            self.generator.initialize(self.scenario_config, self.optimization_config)
            
            parameters = self.generator.suggest_parameters()
            
            # Report result with list value
            result = EvaluationResult(
                objective_value=[1.5, 2.0],  # List value for single objective
                metrics={'sharpe_ratio': 1.5},
                window_results=[]
            )
            
            self.generator.report_result(parameters, result)
            
            # Should use first value from list
            mock_study.tell.assert_called_once_with(mock_trial, 1.5)
    
    def test_report_result_multi_objective(self):
        """Test reporting results for multi-objective optimization."""
        multi_objective_config = {
            'parameter_space': {
                'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },
            'optimization_targets': [
                {'name': 'sharpe_ratio', 'direction': 'maximize'},
                {'name': 'max_drawdown', 'direction': 'minimize'}
            ]
        }
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_trial = Mock()
            mock_trial.number = 0
            mock_trial.state = Mock()
            mock_trial.state.name = 'COMPLETE'
            mock_study.ask.return_value = mock_trial
            mock_create_study.return_value = mock_study
            
            self.generator.initialize({}, multi_objective_config)
            
            parameters = self.generator.suggest_parameters()
            
            # Report multi-objective result
            result = EvaluationResult(
                objective_value=[1.5, -0.1],
                metrics={'sharpe_ratio': 1.5, 'max_drawdown': -0.1},
                window_results=[]
            )
            
            self.generator.report_result(parameters, result)
            
            # Verify study.tell was called with list of values
            mock_study.tell.assert_called_once_with(mock_trial, [1.5, -0.1])
    
    def test_report_result_multi_objective_single_value(self):
        """Test reporting single value for multi-objective optimization."""
        multi_objective_config = {
            'parameter_space': {
                'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },
            'optimization_targets': [
                {'name': 'sharpe_ratio', 'direction': 'maximize'},
                {'name': 'max_drawdown', 'direction': 'minimize'}
            ]
        }
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_trial = Mock()
            mock_trial.number = 0
            mock_trial.state = Mock()
            mock_trial.state.name = 'COMPLETE'
            mock_study.ask.return_value = mock_trial
            mock_create_study.return_value = mock_study
            
            self.generator.initialize({}, multi_objective_config)
            
            parameters = self.generator.suggest_parameters()
            
            # Report single value for multi-objective
            result = EvaluationResult(
                objective_value=1.5,  # Single value
                metrics={'sharpe_ratio': 1.5},
                window_results=[]
            )
            
            self.generator.report_result(parameters, result)
            
            # Should replicate value for all objectives
            mock_study.tell.assert_called_once_with(mock_trial, [1.5, 1.5])
    
    def test_report_result_wrong_number_of_values(self):
        """Test reporting wrong number of values for multi-objective."""
        multi_objective_config = {
            'parameter_space': {
                'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },
            'optimization_targets': [
                {'name': 'metric1', 'direction': 'maximize'},
                {'name': 'metric2', 'direction': 'minimize'}
            ]
        }
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_trial = Mock()
            mock_trial.number = 0
            mock_trial.state = Mock()
            mock_trial.state.name = 'COMPLETE'
            mock_study.ask.return_value = mock_trial
            mock_create_study.return_value = mock_study
            
            self.generator.initialize({}, multi_objective_config)
            
            parameters = self.generator.suggest_parameters()
            
            # Report wrong number of values
            result = EvaluationResult(
                objective_value=[1.5, 2.0, 3.0],  # 3 values for 2 objectives
                metrics={},
                window_results=[]
            )
            
            self.generator.report_result(parameters, result)
            
            # Should use first value for all objectives
            mock_study.tell.assert_called_once_with(mock_trial, [1.5, 1.5])
    
    def test_report_result_error_handling(self):
        """Test error handling during result reporting."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_trial = Mock()
            mock_study.ask.return_value = mock_trial
            mock_study.tell.side_effect = Exception("Tell failed")
            mock_create_study.return_value = mock_study
            
            self.generator.initialize(self.scenario_config, self.optimization_config)
            
            parameters = self.generator.suggest_parameters()
            result = EvaluationResult(objective_value=1.0, metrics={}, window_results=[])
            
            with pytest.raises(ParameterGeneratorError, match="Failed to report result"):
                self.generator.report_result(parameters, result)
            
            # Current trial should be cleared on error
            assert self.generator._current_trial is None


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
class TestOptunaParameterGeneratorCompletion:
    """Test optimization completion and result retrieval."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = OptunaParameterGenerator(random_state=42)
        self.scenario_config = {'name': 'test_scenario'}
        self.optimization_config = {
            'parameter_space': {
                'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },
            'metrics_to_optimize': ['sharpe_ratio'],
            'max_evaluations': 2  # Small number for testing
        }
    
    def test_is_finished_not_initialized(self):
        """Test is_finished returns False when not initialized."""
        assert not self.generator.is_finished()
    
    def test_is_finished_max_evaluations(self):
        """Test is_finished based on max evaluations."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            self.generator.initialize(self.scenario_config, self.optimization_config)
            
            assert not self.generator.is_finished()
            
            # Simulate reaching max evaluations
            self.generator.current_evaluation = 2
            assert self.generator.is_finished()
    
    def test_is_finished_study_stopped(self):
        """Test is_finished when study is stopped."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_study._stop_flag = True
            mock_create_study.return_value = mock_study
            
            self.generator.initialize(self.scenario_config, self.optimization_config)
            
            assert self.generator.is_finished()
    
    def test_get_best_result_not_initialized(self):
        """Test get_best_result fails when not initialized."""
        with pytest.raises(ParameterGeneratorNotInitializedError):
            self.generator.get_best_result()
    
    def test_get_best_result_single_objective_no_trials(self):
        """Test get_best_result for single-objective with no completed trials."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_study.trials = []
            mock_create_study.return_value = mock_study
            
            self.generator.initialize(self.scenario_config, self.optimization_config)
            
            result = self.generator.get_best_result()
            
            assert result.best_parameters == {}
            assert result.best_value == -1e9
            assert result.n_evaluations == 0
            assert result.optimization_history == []
            assert result.best_trial is None
    
    def test_get_best_result_single_objective_with_trials(self):
        """Test get_best_result for single-objective with completed trials."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_trial = Mock()
            mock_trial.state = optuna.trial.TrialState.COMPLETE
            mock_trial.params = {'param1': 0.5}
            mock_trial.value = 1.5
            mock_study.trials = [mock_trial]
            mock_study.best_trial = mock_trial
            mock_create_study.return_value = mock_study
            
            self.generator.initialize(self.scenario_config, self.optimization_config)
            self.generator.current_evaluation = 1
            
            result = self.generator.get_best_result()
            
            assert result.best_parameters == {'param1': 0.5}
            assert result.best_value == 1.5
            assert result.n_evaluations == 1
            assert result.best_trial == mock_trial
    
    def test_get_best_result_multi_objective_no_trials(self):
        """Test get_best_result for multi-objective with no trials."""
        multi_objective_config = {
            'parameter_space': {
                'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },
            'optimization_targets': [
                {'name': 'sharpe_ratio', 'direction': 'maximize'},
                {'name': 'max_drawdown', 'direction': 'minimize'}
            ]
        }
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_study.best_trials = []
            mock_create_study.return_value = mock_study
            
            self.generator.initialize({}, multi_objective_config)
            
            result = self.generator.get_best_result()
            
            assert result.best_parameters == {}
            assert result.best_value == [-1e9, -1e9]
            assert result.n_evaluations == 0
            assert result.best_trial is None
    
    def test_get_best_result_multi_objective_with_trials(self):
        """Test get_best_result for multi-objective with trials."""
        multi_objective_config = {
            'parameter_space': {
                'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },
            'optimization_targets': [
                {'name': 'sharpe_ratio', 'direction': 'maximize'},
                {'name': 'max_drawdown', 'direction': 'minimize'}
            ]
        }
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_trial = Mock()
            mock_trial.params = {'param1': 0.7}
            mock_trial.values = [1.8, -0.05]
            mock_trial.value = 1.8
            mock_study.best_trials = [mock_trial]
            mock_create_study.return_value = mock_study
            
            self.generator.initialize({}, multi_objective_config)
            self.generator.current_evaluation = 1
            
            result = self.generator.get_best_result()
            
            assert result.best_parameters == {'param1': 0.7}
            assert result.best_value == [1.8, -0.05]
            assert result.n_evaluations == 1
            assert result.best_trial == mock_trial
    
    def test_get_best_result_error_handling(self):
        """Test get_best_result error handling."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_study.best_trial = Mock()
            mock_study.best_trial.params = Mock()
            mock_study.best_trial.params.copy.side_effect = Exception("Error")
            mock_study.trials = [Mock()]
            mock_study.trials[0].state = optuna.trial.TrialState.COMPLETE
            mock_create_study.return_value = mock_study
            
            self.generator.initialize(self.scenario_config, self.optimization_config)
            
            result = self.generator.get_best_result()
            
            # Should return empty result on error
            assert result.best_parameters == {}
            assert result.best_value == -1e9


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
class TestOptunaParameterGeneratorOptionalMethods:
    """Test optional methods and additional functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = OptunaParameterGenerator(random_state=42)
        self.optimization_config = {
            'parameter_space': {
                'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
    
    def test_supports_multi_objective(self):
        """Test supports_multi_objective method."""
        assert self.generator.supports_multi_objective() is True
    
    def test_supports_pruning(self):
        """Test supports_pruning method."""
        assert self.generator.supports_pruning() is True
    
    def test_get_optimization_history(self):
        """Test get_optimization_history method."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            self.generator.initialize({}, self.optimization_config)
            
            # Add some history
            self.generator.optimization_history = [
                {'evaluation': 1, 'parameters': {'param1': 0.5}, 'objective_value': 1.0}
            ]
            
            history = self.generator.get_optimization_history()
            
            assert len(history) == 1
            assert history[0]['evaluation'] == 1
            assert history is not self.generator.optimization_history  # Should be a copy
    
    def test_get_parameter_importance_not_initialized(self):
        """Test get_parameter_importance when not initialized."""
        assert self.generator.get_parameter_importance() is None
    
    def test_get_parameter_importance_insufficient_trials(self):
        """Test get_parameter_importance with insufficient trials."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_study.trials = [Mock()]  # Only 1 trial
            mock_study.trials[0].state = optuna.trial.TrialState.COMPLETE
            mock_create_study.return_value = mock_study
            
            self.generator.initialize({}, self.optimization_config)
            
            assert self.generator.get_parameter_importance() is None
    
    def test_get_parameter_importance_single_objective(self):
        """Test get_parameter_importance for single-objective."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_study.trials = [Mock(), Mock()]  # 2 trials
            for trial in mock_study.trials:
                trial.state = optuna.trial.TrialState.COMPLETE
            mock_create_study.return_value = mock_study
            
            self.generator.initialize({}, self.optimization_config)
            
            with patch('optuna.importance.get_param_importances', return_value={'param1': 0.8}) as mock_get_importance:
                importance = self.generator.get_parameter_importance()
                assert importance == {'param1': 0.8}
    
    def test_get_parameter_importance_multi_objective(self):
        """Test get_parameter_importance for multi-objective."""
        multi_objective_config = {
            'parameter_space': {
                'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },
            'optimization_targets': [
                {'name': 'sharpe_ratio', 'direction': 'maximize'},
                {'name': 'max_drawdown', 'direction': 'minimize'}
            ]
        }
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_study.trials = [Mock(), Mock()]  # 2 trials
            for trial in mock_study.trials:
                trial.state = optuna.trial.TrialState.COMPLETE
            mock_create_study.return_value = mock_study
            
            # Patch the method directly on the module where it's used
            with patch('src.portfolio_backtester.optimization.generators.optuna_generator.optuna.importance.get_param_importances') as mock_importance:
                mock_importance.return_value = {'param1': 0.9}
                
                self.generator.initialize({}, multi_objective_config)
                
                importance = self.generator.get_parameter_importance()
                
                assert importance == {'param1': 0.9}
                # Should be called with target function for first objective
                mock_importance.assert_called_once()
                call_args = mock_importance.call_args
                assert call_args[0][0] == mock_study
                assert 'target' in call_args[1]

    def test_get_parameter_importance_error(self):
        """Test get_parameter_importance error handling."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_study.trials = [Mock(), Mock()]
            for trial in mock_study.trials:
                trial.state = optuna.trial.TrialState.COMPLETE
            mock_create_study.return_value = mock_study
            
            with patch('optuna.importance.get_param_importances') as mock_importance:
                mock_importance.side_effect = Exception("Importance calculation failed")
                
                self.generator.initialize({}, self.optimization_config)
                
                importance = self.generator.get_parameter_importance()
                
                assert importance is None
    
    def test_set_random_state(self):
        """Test set_random_state method."""
        self.generator.set_random_state(123)
        assert self.generator.random_state == 123
    
    def test_get_current_evaluation_count(self):
        """Test get_current_evaluation_count method."""
        assert self.generator.get_current_evaluation_count() == 0
        
        self.generator.current_evaluation = 5
        assert self.generator.get_current_evaluation_count() == 5
    
    def test_can_suggest_parameters(self):
        """Test can_suggest_parameters method."""
        # Not initialized
        assert not self.generator.can_suggest_parameters()
        
        # Initialized but not finished
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            self.generator.initialize({}, self.optimization_config)
            assert self.generator.can_suggest_parameters()
            
            # Finished
            self.generator.current_evaluation = self.generator.max_evaluations
            assert not self.generator.can_suggest_parameters()
    
    def test_get_study(self):
        """Test get_study method."""
        assert self.generator.get_study() is None
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            self.generator.initialize({}, self.optimization_config)
            assert self.generator.get_study() == mock_study
    
    def test_get_study_name(self):
        """Test get_study_name method."""
        assert self.generator.get_study_name() is None
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            mock_create_study.return_value = mock_study
            
            self.generator.initialize({'name': 'test'}, self.optimization_config)
            assert self.generator.get_study_name() == "test_optuna_seed_42"
    
    def test_trial_count_methods(self):
        """Test trial counting methods."""
        with patch('optuna.create_study') as mock_create_study:
            mock_study = Mock()
            
            # Create mock trials with different states
            completed_trial = Mock()
            completed_trial.state = optuna.trial.TrialState.COMPLETE
            
            pruned_trial = Mock()
            pruned_trial.state = optuna.trial.TrialState.PRUNED
            
            failed_trial = Mock()
            failed_trial.state = optuna.trial.TrialState.FAIL
            
            mock_study.trials = [completed_trial, pruned_trial, failed_trial, completed_trial]
            mock_create_study.return_value = mock_study
            
            self.generator.initialize({}, self.optimization_config)
            
            assert self.generator.get_completed_trials_count() == 2
            assert self.generator.get_pruned_trials_count() == 1
            assert self.generator.get_failed_trials_count() == 1


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
class TestOptunaParameterGeneratorSeparationOfConcerns:
    """Test separation of concerns - Optuna works without PyGAD dependencies."""
    
    def test_optuna_works_without_pygad(self):
        """Test that Optuna parameter generator works without PyGAD dependencies."""
        # Mock PyGAD as unavailable
        with patch.dict('sys.modules', {'pygad': None}):
            # Should still be able to create and use Optuna generator
            generator = OptunaParameterGenerator(random_state=42)
            
            optimization_config = {
                'parameter_space': {
                    'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}
                },
                'metrics_to_optimize': ['sharpe_ratio'],
                'max_evaluations': 10
            }
            
            with patch('optuna.create_study') as mock_create_study:
                mock_study = Mock()
                mock_trial = Mock()
                mock_trial.suggest_float.return_value = 0.5
                mock_study.ask.return_value = mock_trial
                mock_create_study.return_value = mock_study
                
                # Initialize and use generator
                generator.initialize({}, optimization_config)
                parameters = generator.suggest_parameters()
                
                assert 'param1' in parameters
                assert generator.supports_multi_objective()
                assert generator.supports_pruning()
    
    def test_optuna_import_error_handling(self):
        """Test proper error handling when Optuna is not available."""
        with patch('src.portfolio_backtester.optimization.generators.optuna_generator.OPTUNA_AVAILABLE', False):
            with pytest.raises(ImportError, match="requires 'optuna' package"):
                OptunaParameterGenerator()


if __name__ == "__main__":
    pytest.main([__file__])
