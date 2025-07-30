"""
Parameter Generator Interface for optimization backends.

This module defines the ParameterGenerator abstract base class that all
parameter generation strategies must implement. It provides a unified
interface for different optimization backends including Optuna, genetic
algorithms, and other optimization methods.

The interface supports both single and multi-objective optimization and
ensures consistent behavior across all optimization backends.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional

from .results import EvaluationResult, OptimizationResult

logger = logging.getLogger(__name__)


class ParameterGenerator(ABC):
    """Abstract base class for parameter generation strategies.
    
    This class defines the contract that all parameter generators must follow,
    ensuring consistent behavior across different optimization backends. It
    supports both single and multi-objective optimization scenarios.
    
    All parameter generators must implement the five core methods:
    - initialize: Set up the generator with configuration
    - suggest_parameters: Provide the next parameter set to evaluate
    - report_result: Process evaluation results and update internal state
    - is_finished: Check if optimization should continue
    - get_best_result: Return the best optimization result found
    
    The interface is designed to be backend-agnostic, allowing new optimization
    methods to be easily integrated without modifying existing code.
    """
    
    def initialize(
        self, 
        scenario_config: Dict[str, Any],
        optimization_config: Dict[str, Any]
    ) -> None:
        """Initialize the parameter generator with configuration.
        
        This method sets up the parameter generator with the necessary
        configuration for the optimization run. It should extract and
        store all required information from the configuration dictionaries.
        
        Args:
            scenario_config: Scenario configuration including strategy settings,
                           base parameters, and strategy-specific options
            optimization_config: Optimization-specific configuration including
                               parameter space definitions, optimization targets,
                               stopping criteria, and backend-specific settings
        
        Raises:
            ValueError: If the configuration is invalid or incomplete
            NotImplementedError: If required configuration options are not supported
            
        Example:
            >>> generator = SomeParameterGenerator()
            >>> scenario_config = {
            ...     'strategy': 'momentum_strategy',
            ...     'strategy_params': {'base_param': 1.0}
            ... }
            >>> optimization_config = {
            ...     'parameter_space': {
            ...         'lookback': {'type': 'int', 'low': 5, 'high': 50},
            ...         'threshold': {'type': 'float', 'low': 0.1, 'high': 0.9}
            ...     },
            ...     'metrics_to_optimize': ['sharpe_ratio'],
            ...     'max_evaluations': 100
            ... }
            >>> generator.initialize(scenario_config, optimization_config)
        """
        pass
    
    @abstractmethod
    def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest the next parameter set to evaluate.
        
        This method returns the next set of parameters that should be
        evaluated by the backtesting system. The parameters should be
        within the bounds defined in the parameter space and should
        represent a promising candidate based on the optimization strategy.
        
        Returns:
            Dictionary mapping parameter names to their suggested values.
            The keys should match the parameter names defined in the
            parameter space configuration.
            
        Raises:
            RuntimeError: If called before initialize() or after is_finished() returns True
            ValueError: If the parameter generation fails due to invalid state
            
        Example:
            >>> parameters = generator.suggest_parameters()
            >>> # parameters might be: {'lookback': 20, 'threshold': 0.35}
        """
        pass
    
    @abstractmethod
    def report_result(
        self, 
        parameters: Dict[str, Any], 
        result: EvaluationResult
    ) -> None:
        """Report the result of evaluating a parameter set.
        
        This method provides feedback to the parameter generator about
        the performance of a suggested parameter set. The generator should
        use this information to update its internal state and improve
        future parameter suggestions.
        
        Args:
            parameters: The parameter set that was evaluated. This should
                       be the exact dictionary returned by a previous call
                       to suggest_parameters()
            result: The evaluation result containing objective value(s),
                   aggregated metrics, and detailed window results
                   
        Raises:
            ValueError: If the parameters don't match a previously suggested set
            RuntimeError: If called before initialize()
            
        Example:
            >>> parameters = {'lookback': 20, 'threshold': 0.35}
            >>> result = EvaluationResult(
            ...     objective_value=1.25,
            ...     metrics={'sharpe_ratio': 1.25, 'max_drawdown': -0.15},
            ...     window_results=[...]
            ... )
            >>> generator.report_result(parameters, result)
        """
        pass
    
    @abstractmethod
    def is_finished(self) -> bool:
        """Check whether optimization should continue.
        
        This method determines if the optimization process should stop
        based on the generator's internal criteria. This could be due to
        reaching maximum evaluations, convergence, or other stopping
        conditions specific to the optimization algorithm.
        
        Returns:
            True if optimization is complete and no more parameter sets
            should be evaluated, False if optimization should continue
            
        Example:
            >>> while not generator.is_finished():
            ...     params = generator.suggest_parameters()
            ...     result = evaluate_parameters(params)
            ...     generator.report_result(params, result)
        """
        pass
    
    def get_best_result(self) -> OptimizationResult:
        """Get the best optimization result found so far.
        
        This method returns the best parameter set and corresponding
        objective value(s) discovered during the optimization process.
        It should be callable at any time after initialization to get
        the current best result.
        
        Returns:
            OptimizationResult containing the best parameters found,
            the corresponding objective value(s), number of evaluations
            performed, and optimization history
            
        Raises:
            RuntimeError: If called before initialize()
            
        Example:
            >>> result = generator.get_best_result()
            >>> print(f"Best parameters: {result.best_parameters}")
            >>> print(f"Best value: {result.best_value}")
            >>> print(f"Evaluations: {result.n_evaluations}")
        """
        pass

    @abstractmethod
    def get_best_parameters(self) -> Dict[str, Any]:
        """Return the best parameter dictionary discovered so far."""
        ...

    # Optional methods that subclasses can override for additional functionality
    
    def supports_multi_objective(self) -> bool:
        """Check if this generator supports multi-objective optimization.
        
        Returns:
            True if the generator can handle multiple optimization objectives,
            False if it only supports single-objective optimization
        """
        return False
    
    def supports_pruning(self) -> bool:
        """Check if this generator supports early pruning of poor trials.
        
        Returns:
            True if the generator can prune trials early based on intermediate
            results, False if it only evaluates complete trials
        """
        return False
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the complete optimization history.
        
        Returns:
            List of dictionaries containing the history of all parameter
            evaluations performed, including parameters, objective values,
            and any additional metadata
        """
        return []
    
    def get_parameter_importance(self) -> Optional[Dict[str, float]]:
        """Get parameter importance scores if available.
        
        Returns:
            Dictionary mapping parameter names to importance scores (0-1),
            or None if importance analysis is not supported by this generator
        """
        return None
    
    def set_random_state(self, random_state: Optional[int]) -> None:
        """Set the random state for reproducible results.
        
        Args:
            random_state: Random seed for reproducible optimization runs,
                         or None for non-deterministic behavior
        """
        pass
    
    def get_current_evaluation_count(self) -> int:
        """Get the number of parameter evaluations performed so far.
        
        Returns:
            Number of parameter sets that have been evaluated
        """
        return 0
    
    def can_suggest_parameters(self) -> bool:
        """Check if the generator can suggest more parameters.
        
        This is a convenience method that combines the logic of checking
        if the generator is initialized and not finished.
        
        Returns:
            True if suggest_parameters() can be called successfully,
            False otherwise
        """
        try:
            return not self.is_finished()
        except:
            return False


class ParameterGeneratorError(Exception):
    """Base exception for parameter generator errors."""
    pass


class ParameterGeneratorNotInitializedError(ParameterGeneratorError):
    """Raised when a generator method is called before initialization."""
    pass


class ParameterGeneratorFinishedError(ParameterGeneratorError):
    """Raised when trying to suggest parameters from a finished generator."""
    pass


class InvalidParameterSpaceError(ParameterGeneratorError):
    """Raised when the parameter space configuration is invalid."""
    pass


class ParameterEvaluationError(ParameterGeneratorError):
    """Raised when parameter evaluation fails."""
    pass


def validate_parameter_space(parameter_space: Dict[str, Any]) -> bool:
    """Validate a parameter space configuration.
    
    This utility function validates that a parameter space configuration
    is well-formed and contains all required information for parameter
    generation.
    
    Args:
        parameter_space: Dictionary defining the parameter space with
                        parameter names as keys and configuration as values
                        
    Raises:
        InvalidParameterSpaceError: If the parameter space is invalid
        
    Example:
        >>> parameter_space = {
        ...     'lookback': {'type': 'int', 'low': 5, 'high': 50},
        ...     'threshold': {'type': 'float', 'low': 0.1, 'high': 0.9},
        ...     'method': {'type': 'categorical', 'choices': ['A', 'B', 'C']}
        ... }
        >>> validate_parameter_space(parameter_space)  # Should not raise
    """
    if not isinstance(parameter_space, dict):
        raise InvalidParameterSpaceError(
            f"Parameter space must be a dictionary, got {type(parameter_space)}"
        )
    
    if not parameter_space:
        raise InvalidParameterSpaceError("Parameter space cannot be empty")
    
    for param_name, param_config in parameter_space.items():
        if not isinstance(param_name, str):
            raise InvalidParameterSpaceError(
                f"Parameter name must be a string, got {type(param_name)}"
            )
        
        if not isinstance(param_config, dict):
            raise InvalidParameterSpaceError(
                f"Parameter configuration for '{param_name}' must be a dictionary, "
                f"got {type(param_config)}"
            )
        
        param_type = param_config.get('type')
        if param_type not in ['int', 'float', 'categorical']:
            raise InvalidParameterSpaceError(
                f"Parameter '{param_name}' has invalid type '{param_type}'. "
                f"Must be one of: 'int', 'float', 'categorical'"
            )
        
        if param_type in ['int', 'float']:
            if 'low' not in param_config or 'high' not in param_config:
                raise InvalidParameterSpaceError(
                    f"Parameter '{param_name}' of type '{param_type}' must have 'low' and 'high' bounds"
                )
            
            low = param_config['low']
            high = param_config['high']
            
            if param_type == 'int':
                if not isinstance(low, int) or not isinstance(high, int):
                    raise InvalidParameterSpaceError(
                        f"Parameter '{param_name}' bounds must be integers"
                    )
            else:  # float
                if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                    raise InvalidParameterSpaceError(
                        f"Parameter '{param_name}' bounds must be numeric"
                    )
            
            if low >= high:
                raise InvalidParameterSpaceError(
                    f"Parameter '{param_name}' low bound ({low}) must be less than high bound ({high})"
                )
        
        elif param_type == 'categorical':
            if 'choices' not in param_config:
                raise InvalidParameterSpaceError(
                    f"Parameter '{param_name}' of type 'categorical' must have 'choices'"
                )
            
            choices = param_config['choices']
            if not isinstance(choices, (list, tuple)) or len(choices) == 0:
                raise InvalidParameterSpaceError(
                    f"Parameter '{param_name}' choices must be a non-empty list or tuple"
                )
    # If no issues were raised, parameter space is valid
    return True


def validate_optimization_config(optimization_config: Dict[str, Any]) -> bool:
    """Validate an optimization configuration.
    
    This utility function validates that an optimization configuration
    contains all required fields and has valid values.
    
    Args:
        optimization_config: Dictionary containing optimization configuration
        
    Raises:
        ValueError: If the optimization configuration is invalid
    """
    if not isinstance(optimization_config, dict):
        raise ParameterGeneratorError(
            f"Optimization config must be a dictionary, got {type(optimization_config)}"
        )
    
    # Validate parameter space
    if 'parameter_space' in optimization_config:
        validate_parameter_space(optimization_config['parameter_space'])
    
    # Validate metrics to optimize
    metrics_to_optimize = optimization_config.get('metrics_to_optimize', [])
    if not isinstance(metrics_to_optimize, (list, tuple)):
        raise ParameterGeneratorError("metrics_to_optimize must be a list or tuple")
    
    if len(metrics_to_optimize) == 0:
        raise ParameterGeneratorError("At least one metric must be specified for optimization")
    
    for metric in metrics_to_optimize:
        if not isinstance(metric, str):
            raise ParameterGeneratorError(f"Metric name must be a string, got {type(metric)}")
    
    # Validate max evaluations if present
    if 'max_evaluations' in optimization_config:
        max_evals = optimization_config['max_evaluations']
        if not isinstance(max_evals, int) or max_evals <= 0:
            raise ParameterGeneratorError("max_evaluations must be a positive integer")
    # Configuration validated successfully
    return True


# Type aliases for better documentation
ParameterDict = Dict[str, Any]
ObjectiveValue = Union[float, List[float]]
OptimizationConfig = Dict[str, Any]
ScenarioConfig = Dict[str, Any]