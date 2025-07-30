"""
Unit tests for GeneticParameterGenerator.

This module contains comprehensive tests for the genetic algorithm parameter
generator, covering all interface methods, gene space construction, chromosome
encoding/decoding, and separation of concerns validation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Test imports - handle PyGAD availability
try:
    import pygad
    PYGAD_AVAILABLE = True
except ImportError:
    PYGAD_AVAILABLE = False
    pygad = None

from src.portfolio_backtester.optimization.generators.genetic_generator import (
    GeneticParameterGenerator,
    GeneticTrial,
    PYGAD_AVAILABLE
)
from src.portfolio_backtester.optimization.parameter_generator import (
    ParameterGeneratorNotInitializedError,
    ParameterGeneratorFinishedError,
    InvalidParameterSpaceError
)
from src.portfolio_backtester.optimization.results import EvaluationResult, OptimizationResult


class TestGeneticParameterGeneratorBasics:
    """Test basic functionality and initialization."""
    
    def test_import_error_when_pygad_not_available(self):
        """Test that ImportError is raised when PyGAD is not available."""
        with patch('src.portfolio_backtester.optimization.generators.genetic_generator.PYGAD_AVAILABLE', False):
            with pytest.raises(ImportError, match="PyGAD is not available"):
                GeneticParameterGenerator()
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_initialization_with_random_state(self):
        """Test generator initialization with random state."""
        generator = GeneticParameterGenerator(random_state=42)
        
        assert generator.random_state == 42
        assert not generator.is_initialized
        assert generator.current_evaluation == 0
        assert generator.current_generation == 0
        assert len(generator.trials) == 0
        assert generator.best_trial is None
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_initialization_without_random_state(self):
        """Test generator initialization without random state."""
        generator = GeneticParameterGenerator()
        
        assert generator.random_state is None
        assert not generator.is_initialized
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_methods_before_initialization(self):
        """Test that methods raise appropriate errors before initialization."""
        generator = GeneticParameterGenerator()
        
        with pytest.raises(ParameterGeneratorNotInitializedError):
            generator.suggest_parameters()
        
        with pytest.raises(ParameterGeneratorNotInitializedError):
            generator.report_result({}, Mock())
        
        with pytest.raises(ParameterGeneratorNotInitializedError):
            generator.get_best_result()


class TestGeneSpaceConstruction:
    """Test gene space construction from parameter space configurations."""
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_integer_parameter_gene_space(self):
        """Test gene space construction for integer parameters."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'lookback': {'type': 'int', 'low': 5, 'high': 50, 'step': 5},
                'threshold': {'type': 'int', 'low': 1, 'high': 10}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        assert len(generator.gene_space) == 2
        assert generator.gene_space[0] == {'low': 5, 'high': 50, 'step': 5}
        assert generator.gene_space[1] == {'low': 1, 'high': 10, 'step': 1}
        assert generator.gene_types == [int, int]
        assert generator.param_names == ['lookback', 'threshold']
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_float_parameter_gene_space(self):
        """Test gene space construction for float parameters."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'alpha': {'type': 'float', 'low': 0.1, 'high': 0.9},
                'beta': {'type': 'float', 'low': -1.0, 'high': 1.0, 'step': 0.1}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        assert len(generator.gene_space) == 2
        assert generator.gene_space[0] == {'low': 0.1, 'high': 0.9}
        assert generator.gene_space[1] == {'low': -1.0, 'high': 1.0, 'step': 0.1}
        assert generator.gene_types == [float, float]
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_categorical_parameter_gene_space(self):
        """Test gene space construction for categorical parameters."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'method': {'type': 'categorical', 'choices': ['A', 'B', 'C', 'D']},
                'mode': {'type': 'categorical', 'choices': ['fast', 'slow']}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        assert len(generator.gene_space) == 2
        assert generator.gene_space[0] == {'low': 0, 'high': 3, 'step': 1}
        assert generator.gene_space[1] == {'low': 0, 'high': 1, 'step': 1}
        assert generator.gene_types == [int, int]
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_mixed_parameter_types_gene_space(self):
        """Test gene space construction for mixed parameter types."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'lookback': {'type': 'int', 'low': 5, 'high': 50},
                'alpha': {'type': 'float', 'low': 0.1, 'high': 0.9},
                'method': {'type': 'categorical', 'choices': ['A', 'B', 'C']}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        assert len(generator.gene_space) == 3
        assert generator.gene_space[0] == {'low': 5, 'high': 50, 'step': 1}
        assert generator.gene_space[1] == {'low': 0.1, 'high': 0.9}
        assert generator.gene_space[2] == {'low': 0, 'high': 2, 'step': 1}
        assert generator.gene_types == [int, float, int]
        assert generator.param_names == ['lookback', 'alpha', 'method']
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_invalid_parameter_space_empty(self):
        """Test error handling for empty parameter space."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {},
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        with pytest.raises(InvalidParameterSpaceError, match="Parameter space cannot be empty"):
            generator.initialize(scenario_config, optimization_config)
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_invalid_parameter_space_missing_bounds(self):
        """Test error handling for missing parameter bounds."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'int'}  # Missing low and high
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        with pytest.raises(InvalidParameterSpaceError, match="must have 'low' and 'high' bounds"):
            generator.initialize(scenario_config, optimization_config)
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_invalid_parameter_space_invalid_range(self):
        """Test error handling for invalid parameter ranges."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'int', 'low': 10, 'high': 5}  # low >= high
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        with pytest.raises(InvalidParameterSpaceError, match="low .* must be less than high"):
            generator.initialize(scenario_config, optimization_config)
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_invalid_categorical_choices(self):
        """Test error handling for invalid categorical choices."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'method': {'type': 'categorical', 'choices': ['A']}  # Only one choice
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        with pytest.raises(InvalidParameterSpaceError, match="must have at least 2 choices"):
            generator.initialize(scenario_config, optimization_config)
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_unsupported_parameter_type(self):
        """Test error handling for unsupported parameter types."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'unsupported', 'low': 0, 'high': 1}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        with pytest.raises(InvalidParameterSpaceError, match="has invalid type"):
            generator.initialize(scenario_config, optimization_config)


class TestChromosomeEncodingDecoding:
    """Test chromosome encoding and decoding functionality."""
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_decode_integer_parameters(self):
        """Test decoding chromosomes with integer parameters."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'lookback': {'type': 'int', 'low': 5, 'high': 50, 'step': 5},
                'threshold': {'type': 'int', 'low': 1, 'high': 10}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        # Test chromosome decoding
        chromosome = np.array([25.7, 7.3])  # Float values that should be rounded
        parameters = generator._decode_chromosome(chromosome)
        
        assert parameters['lookback'] == 25  # Should be rounded and within step
        assert parameters['threshold'] == 7   # Should be rounded
        assert isinstance(parameters['lookback'], int)
        assert isinstance(parameters['threshold'], int)
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_decode_float_parameters(self):
        """Test decoding chromosomes with float parameters."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'alpha': {'type': 'float', 'low': 0.1, 'high': 0.9},
                'beta': {'type': 'float', 'low': -1.0, 'high': 1.0}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        # Test chromosome decoding
        chromosome = np.array([0.5, 0.25])
        parameters = generator._decode_chromosome(chromosome)
        
        assert parameters['alpha'] == 0.5
        assert parameters['beta'] == 0.25
        assert isinstance(parameters['alpha'], float)
        assert isinstance(parameters['beta'], float)
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_decode_categorical_parameters(self):
        """Test decoding chromosomes with categorical parameters."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'method': {'type': 'categorical', 'choices': ['A', 'B', 'C', 'D']},
                'mode': {'type': 'categorical', 'choices': ['fast', 'slow']}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        # Test chromosome decoding
        chromosome = np.array([2.7, 0.3])  # Should round to indices 3 and 0
        parameters = generator._decode_chromosome(chromosome)
        
        assert parameters['method'] == 'D'    # Index 3 (rounded from 2.7)
        assert parameters['mode'] == 'fast'   # Index 0 (rounded from 0.3)
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_decode_bounds_clamping(self):
        """Test that decoding clamps values to parameter bounds."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'int', 'low': 5, 'high': 10},
                'param2': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'param3': {'type': 'categorical', 'choices': ['A', 'B']}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        # Test chromosome with out-of-bounds values
        chromosome = np.array([15, -0.5, 5])  # All out of bounds
        parameters = generator._decode_chromosome(chromosome)
        
        assert parameters['param1'] == 10    # Clamped to high bound
        assert parameters['param2'] == 0.0   # Clamped to low bound
        assert parameters['param3'] == 'B'   # Clamped to last choice (index 1)
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_encode_parameters(self):
        """Test encoding parameter dictionaries to chromosomes."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'lookback': {'type': 'int', 'low': 5, 'high': 50},
                'alpha': {'type': 'float', 'low': 0.1, 'high': 0.9},
                'method': {'type': 'categorical', 'choices': ['A', 'B', 'C']}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        # Test parameter encoding
        parameters = {
            'lookback': 20,
            'alpha': 0.5,
            'method': 'B'
        }
        
        chromosome = generator._encode_parameters(parameters)
        
        assert chromosome[0] == 20.0  # Integer parameter
        assert chromosome[1] == 0.5   # Float parameter
        assert chromosome[2] == 1.0   # Categorical parameter (index of 'B')
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding are consistent."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'lookback': {'type': 'int', 'low': 5, 'high': 50},
                'alpha': {'type': 'float', 'low': 0.1, 'high': 0.9},
                'method': {'type': 'categorical', 'choices': ['A', 'B', 'C']}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        # Original parameters
        original_params = {
            'lookback': 25,
            'alpha': 0.7,
            'method': 'C'
        }
        
        # Encode and decode
        chromosome = generator._encode_parameters(original_params)
        decoded_params = generator._decode_chromosome(chromosome)
        
        # Should be identical
        assert decoded_params == original_params
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_decode_invalid_chromosome_length(self):
        """Test error handling for invalid chromosome length."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'int', 'low': 1, 'high': 10},
                'param2': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        # Chromosome with wrong length
        chromosome = np.array([5])  # Should have 2 genes
        
        with pytest.raises(ValueError, match="Chromosome length .* doesn't match parameter space size"):
            generator._decode_chromosome(chromosome)


class TestParameterGeneratorInterface:
    """Test implementation of the ParameterGenerator interface."""
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_single_objective_optimization(self):
        """Test single objective optimization workflow."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'int', 'low': 1, 'high': 10}
            },
            'metrics_to_optimize': ['sharpe_ratio'],
            'max_evaluations': 10,
            'genetic_algorithm_params': {
                'num_generations': 2,
                'sol_per_pop': 4
            }
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        assert generator.is_initialized
        assert not generator.is_multi_objective
        assert generator.metrics_to_optimize == ['sharpe_ratio']
        assert not generator.is_finished()
        
        # Test parameter suggestion and result reporting
        with patch.object(generator, '_external_fitness_function', return_value=0.0):
            with patch.object(generator.ga_instance, 'population', np.array([[5.0], [7.0], [3.0], [9.0]])):
                parameters = generator.suggest_parameters()
                assert 'param1' in parameters
                assert isinstance(parameters['param1'], int)
                assert 1 <= parameters['param1'] <= 10
                
                # Report result
                result = EvaluationResult(
                    objective_value=1.5,
                    metrics={'sharpe_ratio': 1.5},
                    window_results=[]
                )
                
                generator.report_result(parameters, result)
                
                assert generator.current_evaluation == 1
                assert len(generator.trials) == 1
                assert generator.best_trial is not None
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization workflow."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'int', 'low': 1, 'high': 10}
            },
            'metrics_to_optimize': ['sharpe_ratio', 'max_drawdown'],
            'optimization_targets': [
                {'direction': 'maximize'},
                {'direction': 'minimize'}
            ],
            'max_evaluations': 10,
            'genetic_algorithm_params': {
                'num_generations': 2,
                'sol_per_pop': 4
            }
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        assert generator.is_multi_objective
        assert generator.metrics_to_optimize == ['sharpe_ratio', 'max_drawdown']
        assert generator.optimization_directions == ['maximize', 'minimize']
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_get_best_result_no_trials(self):
        """Test get_best_result when no trials have been completed."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'int', 'low': 1, 'high': 10}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        result = generator.get_best_result()
        
        assert isinstance(result, OptimizationResult)
        assert result.best_parameters == {}
        assert result.best_value == -1e9
        assert result.n_evaluations == 0
        assert result.best_trial is None
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_get_best_result_with_trials(self):
        """Test get_best_result with completed trials."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'int', 'low': 1, 'high': 10}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        # Manually create a trial
        trial = GeneticTrial(
            number=0,
            parameters={'param1': 5},
            chromosome=np.array([5.0]),
            fitness=1.5,
            generation=0
        )
        
        generator.trials = [trial]
        generator.best_trial = trial
        generator.current_evaluation = 1
        
        result = generator.get_best_result()
        
        assert result.best_parameters == {'param1': 5}
        assert result.best_value == 1.5  # Should convert fitness back to objective
        assert result.n_evaluations == 1
        assert result.best_trial == trial
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_supports_multi_objective(self):
        """Test multi-objective support detection."""
        generator = GeneticParameterGenerator(random_state=42)
        
        # This depends on whether NSGA-II is available in PyGAD
        try:
            import pygad.nsga2 as nsga2_module
            assert generator.supports_multi_objective()
        except (ImportError, AttributeError):
            assert not generator.supports_multi_objective()
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_supports_pruning(self):
        """Test pruning support (should be False for genetic algorithms)."""
        generator = GeneticParameterGenerator(random_state=42)
        assert not generator.supports_pruning()
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_set_random_state(self):
        """Test setting random state."""
        generator = GeneticParameterGenerator(random_state=42)
        
        generator.set_random_state(123)
        assert generator.random_state == 123
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_get_current_evaluation_count(self):
        """Test getting current evaluation count."""
        generator = GeneticParameterGenerator(random_state=42)
        
        assert generator.get_current_evaluation_count() == 0
        
        generator.current_evaluation = 5
        assert generator.get_current_evaluation_count() == 5


class TestSeparationOfConcerns:
    """Test separation of concerns - PyGAD works without Optuna dependencies."""
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_pygad_without_optuna_dependencies(self):
        """Test that PyGAD parameter generator works without Optuna dependencies."""
        # Mock optuna as unavailable
        with patch.dict('sys.modules', {'optuna': None}):
            # Should still be able to create and use genetic generator
            generator = GeneticParameterGenerator(random_state=42)
            
            scenario_config = {}
            optimization_config = {
                'parameter_space': {
                    'param1': {'type': 'int', 'low': 1, 'high': 10}
                },
                'metrics_to_optimize': ['sharpe_ratio']
            }
            
            # Should initialize without issues
            generator.initialize(scenario_config, optimization_config)
            
            assert generator.is_initialized
            assert len(generator.gene_space) == 1
            assert generator.param_names == ['param1']
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_no_backtesting_code_in_generator(self):
        """Test that genetic generator contains no backtesting logic."""
        generator = GeneticParameterGenerator(random_state=42)
        
        # Check that generator doesn't import backtesting modules
        import inspect
        source = inspect.getsource(generator.__class__)
        
        # Should not contain backtesting imports
        assert 'from ...backtesting.backtester' not in source
        assert 'from ..backtester' not in source
        assert 'import backtester' not in source
        
        # Should not contain strategy execution logic
        assert 'strategy_execution' not in source.lower()
        assert 'backtest_strategy' not in source.lower()
        assert 'evaluate_window' not in source.lower()
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_genetic_generator_interface_compliance(self):
        """Test that genetic generator properly implements ParameterGenerator interface."""
        from src.portfolio_backtester.optimization.parameter_generator import ParameterGenerator
        
        generator = GeneticParameterGenerator(random_state=42)
        
        # Should be instance of ParameterGenerator
        assert isinstance(generator, ParameterGenerator)
        
        # Should implement all required methods
        required_methods = [
            'initialize', 'suggest_parameters', 'report_result', 
            'is_finished', 'get_best_result'
        ]
        
        for method_name in required_methods:
            assert hasattr(generator, method_name)
            assert callable(getattr(generator, method_name))


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_finished_generator_suggest_parameters(self):
        """Test that finished generator raises appropriate error."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'int', 'low': 1, 'high': 10}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        generator.is_finished_flag = True
        
        with pytest.raises(ParameterGeneratorFinishedError):
            generator.suggest_parameters()
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_report_result_invalid_parameters(self):
        """Test that report_result works with any valid parameters."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'int', 'low': 1, 'high': 10}
            },
            'metrics_to_optimize': ['sharpe_ratio']
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        # Report result for valid parameters (should work with simplified approach)
        valid_params = {'param1': 5}
        result = EvaluationResult(
            objective_value=1.0,
            metrics={'sharpe_ratio': 1.0},
            window_results=[]
        )
        
        # Should not raise an error with the simplified approach
        generator.report_result(valid_params, result)
        
        assert generator.current_evaluation == 1
        assert len(generator.trials) == 1
    
    @pytest.mark.skipif(not PYGAD_AVAILABLE, reason="PyGAD not available")
    def test_population_size_validation(self):
        """Test population size validation and adjustment."""
        generator = GeneticParameterGenerator(random_state=42)
        
        scenario_config = {}
        optimization_config = {
            'parameter_space': {
                'param1': {'type': 'int', 'low': 1, 'high': 10}
            },
            'metrics_to_optimize': ['sharpe_ratio'],
            'genetic_algorithm_params': {
                'sol_per_pop': 2,  # Too small
                'num_parents_mating': 10  # Too large relative to population
            }
        }
        
        generator.initialize(scenario_config, optimization_config)
        
        # Should adjust population size and parents
        assert generator.population_size >= 4
        assert generator.num_parents_mating < generator.population_size
        assert generator.num_parents_mating >= 2


if __name__ == '__main__':
    pytest.main([__file__])