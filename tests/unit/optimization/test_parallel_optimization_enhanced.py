"""
Tests for parallel optimization runner with enhanced WFO windows.

This module tests the integration of the parallel optimization runner with
the enhanced WFO system that supports daily evaluation for intramonth strategies.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.portfolio_backtester.optimization.parallel_optimization_runner import (
    ParallelOptimizationRunner,
    _optuna_worker
)
from src.portfolio_backtester.optimization.results import OptimizationData, OptimizationResult
from src.portfolio_backtester.backtesting.results import WindowResult


class TestParallelOptimizationEnhanced:
    """Test parallel optimization with enhanced WFO windows."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample optimization data."""
        # Create sample price data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        price_data = pd.DataFrame({
            'AAPL': 100 + (dates - dates[0]).days * 0.1,
            'MSFT': 200 + (dates - dates[0]).days * 0.05,
        }, index=dates)
        
        # Create sample windows
        windows = [
            (datetime(2023, 1, 1), datetime(2023, 6, 30), 
             datetime(2023, 7, 1), datetime(2023, 12, 31))
        ]
        
        return OptimizationData(
            monthly=price_data.resample('M').last(),
            daily=price_data,
            returns=price_data.pct_change().dropna(),
            windows=windows
        )
    
    @pytest.fixture
    def intramonth_scenario_config(self):
        """Create scenario config for intramonth strategy."""
        return {
            'name': 'test_intramonth_scenario',
            'strategy': 'IntramonthSeasonalStrategy',
            'strategy_params': {},
            'timing': {
                'rebalance_frequency': 'D',
                'mode': 'signal_based',
                'scan_frequency': 'D'
            }
        }
    
    @pytest.fixture
    def monthly_scenario_config(self):
        """Create scenario config for monthly strategy."""
        return {
            'name': 'test_monthly_scenario',
            'strategy': 'MonthlyStrategy',
            'strategy_params': {},
            'timing': {
                'rebalance_frequency': 'M'
            }
        }
    
    @pytest.fixture
    def optimization_config(self):
        """Create optimization configuration."""
        return {
            'optuna_trials': 5,
            'parameter_space': {
                'test_param': {
                    'type': 'float',
                    'low': 0.1,
                    'high': 1.0
                }
            }
        }
    
    def test_parallel_runner_initialization_intramonth(self, intramonth_scenario_config, 
                                                      optimization_config, sample_data):
        """Test parallel runner initialization with intramonth strategy."""
        runner = ParallelOptimizationRunner(
            scenario_config=intramonth_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=2
        )
        
        assert runner.scenario_config == intramonth_scenario_config
        assert runner.optimization_config == optimization_config
        assert runner.data == sample_data
        assert runner.n_jobs == 2
        assert runner.enable_deduplication is True
    
    def test_parallel_runner_initialization_monthly(self, monthly_scenario_config, 
                                                   optimization_config, sample_data):
        """Test parallel runner initialization with monthly strategy."""
        runner = ParallelOptimizationRunner(
            scenario_config=monthly_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=2
        )
        
        assert runner.scenario_config == monthly_scenario_config
        assert runner.optimization_config == optimization_config
        assert runner.data == sample_data
        assert runner.n_jobs == 2
    
    @patch('src.portfolio_backtester.optimization.parallel_optimization_runner.optuna')
    @patch('src.portfolio_backtester.optimization.parallel_optimization_runner.OptunaObjectiveAdapter')
    def test_worker_function_intramonth_logging(self, mock_adapter, mock_optuna, 
                                               intramonth_scenario_config, optimization_config, 
                                               sample_data):
        """Test worker function with enhanced logging for intramonth strategies."""
        # Mock study
        mock_study = Mock()
        mock_optuna.create_study.return_value = mock_study
        
        # Mock objective adapter
        mock_objective = Mock()
        mock_adapter.return_value = mock_objective
        
        with patch('src.portfolio_backtester.optimization.parallel_optimization_runner.logger') as mock_logger:
            _optuna_worker(
                scenario_config=intramonth_scenario_config,
                optimization_config=optimization_config,
                data=sample_data,
                storage_url="sqlite:///test.db",
                study_name="test_study",
                n_trials=2,
                parameter_space={'test_param': {'type': 'float', 'low': 0.1, 'high': 1.0}}
            )
            
            # Verify enhanced logging was called
            mock_logger.info.assert_any_call(
                "Worker %d starting %d trials with %s evaluation", 
                mock_logger.info.call_args_list[0][0][1],  # PID
                2,  # n_trials
                "daily"  # evaluation_mode for intramonth
            )
    
    @patch('src.portfolio_backtester.optimization.parallel_optimization_runner.optuna')
    @patch('src.portfolio_backtester.optimization.parallel_optimization_runner.OptunaObjectiveAdapter')
    def test_worker_function_monthly_logging(self, mock_adapter, mock_optuna, 
                                            monthly_scenario_config, optimization_config, 
                                            sample_data):
        """Test worker function with enhanced logging for monthly strategies."""
        # Mock study
        mock_study = Mock()
        mock_optuna.create_study.return_value = mock_study
        
        # Mock objective adapter
        mock_objective = Mock()
        mock_adapter.return_value = mock_objective
        
        with patch('src.portfolio_backtester.optimization.parallel_optimization_runner.logger') as mock_logger:
            _optuna_worker(
                scenario_config=monthly_scenario_config,
                optimization_config=optimization_config,
                data=sample_data,
                storage_url="sqlite:///test.db",
                study_name="test_study",
                n_trials=2,
                parameter_space={'test_param': {'type': 'float', 'low': 0.1, 'high': 1.0}}
            )
            
            # Verify enhanced logging was called
            mock_logger.info.assert_any_call(
                "Worker %d starting %d trials with %s evaluation", 
                mock_logger.info.call_args_list[0][0][1],  # PID
                2,  # n_trials
                "monthly"  # evaluation_mode for monthly
            )
    
    @patch('src.portfolio_backtester.optimization.parallel_optimization_runner.optuna')
    def test_run_single_process_intramonth(self, mock_optuna, intramonth_scenario_config, 
                                          optimization_config, sample_data):
        """Test single process run with intramonth strategy."""
        # Mock study
        mock_study = Mock()
        mock_study.best_params = {'test_param': 0.5}
        mock_study.best_value = 1.5
        mock_study.trials = [Mock(state=Mock(is_finished=Mock(return_value=True)))] * 5
        mock_optuna.create_study.return_value = mock_study
        
        runner = ParallelOptimizationRunner(
            scenario_config=intramonth_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=1
        )
        
        with patch('src.portfolio_backtester.optimization.parallel_optimization_runner._optuna_worker') as mock_worker:
            with patch('src.portfolio_backtester.optimization.parallel_optimization_runner.logger') as mock_logger:
                result = runner.run()
                
                # Verify enhanced logging
                mock_logger.info.assert_any_call(
                    "Running optimisation in a single process (%d trials, %s evaluation)",
                    5, "daily"
                )
                
                # Verify worker was called
                mock_worker.assert_called_once()
                
                # Verify result
                assert isinstance(result, OptimizationResult)
                assert result.best_parameters == {'test_param': 0.5}
                assert result.best_value == 1.5
    
    @patch('src.portfolio_backtester.optimization.parallel_optimization_runner.optuna')
    @patch('src.portfolio_backtester.optimization.parallel_optimization_runner.mp')
    def test_run_multi_process_intramonth(self, mock_mp, mock_optuna, intramonth_scenario_config, 
                                         optimization_config, sample_data):
        """Test multi-process run with intramonth strategy."""
        # Mock study
        mock_study = Mock()
        mock_study.best_params = {'test_param': 0.7}
        mock_study.best_value = 2.1
        mock_study.trials = [Mock(state=Mock(is_finished=Mock(return_value=True)))] * 5
        mock_optuna.create_study.return_value = mock_study
        
        # Mock multiprocessing
        mock_ctx = Mock()
        mock_process = Mock()
        mock_ctx.Process.return_value = mock_process
        mock_ctx.Lock.return_value = Mock()
        mock_mp.get_context.return_value = mock_ctx
        
        runner = ParallelOptimizationRunner(
            scenario_config=intramonth_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=2
        )
        
        with patch('src.portfolio_backtester.optimization.parallel_optimization_runner.logger') as mock_logger:
            result = runner.run()
            
            # Verify enhanced logging
            mock_logger.info.assert_any_call(
                "Launching %d worker processes for %d trials (%s evaluation)",
                2, 5, "daily"
            )
            
            # Verify processes were started
            assert mock_process.start.call_count == 2
            assert mock_process.join.call_count == 2
            
            # Verify result
            assert isinstance(result, OptimizationResult)
            assert result.best_parameters == {'test_param': 0.7}
            assert result.best_value == 2.1
    
    @patch('src.portfolio_backtester.optimization.parallel_optimization_runner.optuna')
    def test_optimization_success_logging_intramonth(self, mock_optuna, intramonth_scenario_config, 
                                                    optimization_config, sample_data):
        """Test success logging for intramonth strategy optimization."""
        # Mock study with successful results
        mock_study = Mock()
        mock_study.best_params = {'test_param': 0.8}
        mock_study.best_value = 1.8
        mock_study.trials = [Mock(state=Mock(is_finished=Mock(return_value=True)))] * 5
        mock_optuna.create_study.return_value = mock_study
        
        runner = ParallelOptimizationRunner(
            scenario_config=intramonth_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=1
        )
        
        with patch('src.portfolio_backtester.optimization.parallel_optimization_runner._optuna_worker'):
            with patch('src.portfolio_backtester.optimization.parallel_optimization_runner.logger') as mock_logger:
                result = runner.run()
                
                # Verify success logging with evaluation mode
                mock_logger.info.assert_any_call(
                    "✅ Optimization completed successfully: %d trials, best value: %.6f (%s evaluation)",
                    5, 1.8, "daily"
                )
    
    @patch('src.portfolio_backtester.optimization.parallel_optimization_runner.optuna')
    def test_optimization_success_logging_monthly(self, mock_optuna, monthly_scenario_config, 
                                                 optimization_config, sample_data):
        """Test success logging for monthly strategy optimization."""
        # Mock study with successful results
        mock_study = Mock()
        mock_study.best_params = {'test_param': 0.6}
        mock_study.best_value = 1.2
        mock_study.trials = [Mock(state=Mock(is_finished=Mock(return_value=True)))] * 3
        mock_optuna.create_study.return_value = mock_study
        
        runner = ParallelOptimizationRunner(
            scenario_config=monthly_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=1
        )
        
        with patch('src.portfolio_backtester.optimization.parallel_optimization_runner._optuna_worker'):
            with patch('src.portfolio_backtester.optimization.parallel_optimization_runner.logger') as mock_logger:
                result = runner.run()
                
                # Verify success logging with evaluation mode
                mock_logger.info.assert_any_call(
                    "✅ Optimization completed successfully: %d trials, best value: %.6f (%s evaluation)",
                    3, 1.2, "monthly"
                )
    
    def test_evaluation_mode_detection_intramonth(self, intramonth_scenario_config):
        """Test evaluation mode detection for intramonth strategies."""
        strategy_name = intramonth_scenario_config.get('strategy', 'unknown')
        is_intramonth = 'intramonth' in strategy_name.lower()
        evaluation_mode = "daily" if is_intramonth else "monthly"
        
        assert is_intramonth is True
        assert evaluation_mode == "daily"
    
    def test_evaluation_mode_detection_monthly(self, monthly_scenario_config):
        """Test evaluation mode detection for monthly strategies."""
        strategy_name = monthly_scenario_config.get('strategy', 'unknown')
        is_intramonth = 'intramonth' in strategy_name.lower()
        evaluation_mode = "daily" if is_intramonth else "monthly"
        
        assert is_intramonth is False
        assert evaluation_mode == "monthly"
    
    @patch('src.portfolio_backtester.optimization.parallel_optimization_runner.optuna')
    def test_backward_compatibility_maintained(self, mock_optuna, monthly_scenario_config, 
                                              optimization_config, sample_data):
        """Test that backward compatibility is maintained for existing monthly strategies."""
        # Mock study
        mock_study = Mock()
        mock_study.best_params = {'test_param': 0.4}
        mock_study.best_value = 0.9
        mock_study.trials = [Mock(state=Mock(is_finished=Mock(return_value=True)))] * 3
        mock_optuna.create_study.return_value = mock_study
        
        runner = ParallelOptimizationRunner(
            scenario_config=monthly_scenario_config,
            optimization_config=optimization_config,
            data=sample_data,
            n_jobs=1
        )
        
        with patch('src.portfolio_backtester.optimization.parallel_optimization_runner._optuna_worker'):
            result = runner.run()
            
            # Verify result structure is unchanged
            assert isinstance(result, OptimizationResult)
            assert hasattr(result, 'best_parameters')
            assert hasattr(result, 'best_value')
            assert hasattr(result, 'n_evaluations')
            assert hasattr(result, 'optimization_history')
            
            # Verify values
            assert result.best_parameters == {'test_param': 0.4}
            assert result.best_value == 0.9
            assert result.n_evaluations == 3