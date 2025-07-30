# Extensibility Guide

## Overview

This guide demonstrates how to extend the Portfolio Backtester architecture with new optimization backends, custom parameter generators, and additional functionality. The refactored architecture follows the Open/Closed Principle, making it easy to add new components without modifying existing code.

## Adding a New Parameter Generator

### Step 1: Implement the ParameterGenerator Interface

Create a new parameter generator by implementing the abstract `ParameterGenerator` class:

```python
# src/portfolio_backtester/optimization/generators/custom_generator.py

from typing import Dict, Any, Optional, List
import numpy as np
from ..parameter_generator import ParameterGenerator, OptimizationSpec, EvaluationResult

class BayesianOptimizationGenerator(ParameterGenerator):
    """
    Example implementation using scikit-optimize for Bayesian optimization.
    
    This demonstrates how to integrate a third-party optimization library
    into the Portfolio Backtester architecture.
    """
    
    def __init__(
        self,
        random_state: Optional[int] = None,
        n_initial_points: int = 10,
        acquisition_function: str = 'gp_hedge'
    ):
        """
        Initialize Bayesian optimization parameter generator.
        
        Args:
            random_state: Random seed for reproducibility
            n_initial_points: Number of random points to evaluate before using GP
            acquisition_function: Acquisition function for Bayesian optimization
        """
        self.random_state = random_state
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        
        # Internal state
        self.optimizer = None
        self.parameter_names = []
        self.parameter_specs = {}
        self.trial_count = 0
        self.results_history = []
        self.best_parameters = None
        self.best_objective = float('-inf')
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
    
    def suggest_parameters(self, optimization_spec: OptimizationSpec) -> Dict[str, Any]:
        """
        Suggest next parameter set using Bayesian optimization.
        
        Args:
            optimization_spec: Parameter specifications with ranges and types
            
        Returns:
            Dictionary of suggested parameter values
        """
        # Initialize optimizer on first call
        if self.optimizer is None:
            self._initialize_optimizer(optimization_spec)
        
        # Get next point from Bayesian optimizer
        if self.trial_count < self.n_initial_points:
            # Random sampling for initial points
            suggested_point = self._random_sample(optimization_spec)
        else:
            # Use Bayesian optimization
            suggested_point = self._bayesian_sample()
        
        # Convert to parameter dictionary
        parameters = self._point_to_parameters(suggested_point, optimization_spec)
        
        return parameters
    
    def report_result(
        self, 
        parameters: Dict[str, Any], 
        result: EvaluationResult
    ) -> None:
        """
        Report evaluation result back to the optimizer.
        
        Args:
            parameters: Parameter set that was evaluated
            result: Evaluation result with objective value and metrics
        """
        # Convert parameters to point format
        point = self._parameters_to_point(parameters)
        objective_value = result.objective_value
        
        # Store result
        self.results_history.append({
            'parameters': parameters.copy(),
            'point': point,
            'objective': objective_value,
            'result': result
        })
        
        # Update best result
        if objective_value > self.best_objective:
            self.best_objective = objective_value
            self.best_parameters = parameters.copy()
        
        # Tell optimizer about the result
        if self.trial_count >= self.n_initial_points:
            # Convert to minimization problem (skopt minimizes)
            self.optimizer.tell(point, -objective_value)
        
        self.trial_count += 1
    
    def is_complete(self) -> bool:
        """
        Check if optimization should continue.
        
        Returns:
            True if optimization should stop, False otherwise
        """
        # Could implement convergence criteria here
        # For now, let the orchestrator control stopping
        return False
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Get the best parameters found so far.
        
        Returns:
            Dictionary of best parameter values
        """
        return self.best_parameters or {}
    
    def _initialize_optimizer(self, optimization_spec: OptimizationSpec) -> None:
        """Initialize the scikit-optimize optimizer."""
        try:
            from skopt import Optimizer
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            raise ImportError(
                "scikit-optimize is required for BayesianOptimizationGenerator. "
                "Install with: pip install scikit-optimize"
            )
        
        # Build search space
        dimensions = []
        self.parameter_names = []
        self.parameter_specs = optimization_spec.copy()
        
        for param_name, spec in optimization_spec.items():
            self.parameter_names.append(param_name)
            
            if spec['type'] == 'int':
                dimensions.append(Integer(spec['low'], spec['high'], name=param_name))
            elif spec['type'] == 'float':
                if spec.get('log', False):
                    dimensions.append(Real(
                        spec['low'], spec['high'], 
                        prior='log-uniform', name=param_name
                    ))
                else:
                    dimensions.append(Real(spec['low'], spec['high'], name=param_name))
            elif spec['type'] == 'categorical':
                dimensions.append(Categorical(spec['choices'], name=param_name))
            else:
                raise ValueError(f"Unsupported parameter type: {spec['type']}")
        
        # Create optimizer
        self.optimizer = Optimizer(
            dimensions=dimensions,
            random_state=self.random_state,
            acq_func=self.acquisition_function
        )
    
    def _random_sample(self, optimization_spec: OptimizationSpec) -> List[Any]:
        """Generate random sample for initial exploration."""
        point = []
        
        for param_name in self.parameter_names:
            spec = optimization_spec[param_name]
            
            if spec['type'] == 'int':
                value = np.random.randint(spec['low'], spec['high'] + 1)
            elif spec['type'] == 'float':
                if spec.get('log', False):
                    log_low = np.log(spec['low'])
                    log_high = np.log(spec['high'])
                    value = np.exp(np.random.uniform(log_low, log_high))
                else:
                    value = np.random.uniform(spec['low'], spec['high'])
            elif spec['type'] == 'categorical':
                value = np.random.choice(spec['choices'])
            
            point.append(value)
        
        return point
    
    def _bayesian_sample(self) -> List[Any]:
        """Get next point from Bayesian optimizer."""
        return self.optimizer.ask()
    
    def _point_to_parameters(
        self, 
        point: List[Any], 
        optimization_spec: OptimizationSpec
    ) -> Dict[str, Any]:
        """Convert optimizer point to parameter dictionary."""
        parameters = {}
        
        for i, param_name in enumerate(self.parameter_names):
            parameters[param_name] = point[i]
        
        return parameters
    
    def _parameters_to_point(self, parameters: Dict[str, Any]) -> List[Any]:
        """Convert parameter dictionary to optimizer point."""
        point = []
        
        for param_name in self.parameter_names:
            point.append(parameters[param_name])
        
        return point
```

### Step 2: Register the New Generator in the Factory

Update the factory function to support the new generator:

```python
# src/portfolio_backtester/optimization/factory.py

def create_parameter_generator(
    optimizer_type: str,
    random_state: Optional[int] = None,
    **kwargs
) -> ParameterGenerator:
    """
    Factory function to create parameter generators.
    
    Args:
        optimizer_type: Type of optimizer ('optuna', 'genetic', 'bayesian')
        random_state: Random seed for reproducibility
        **kwargs: Additional optimizer-specific parameters
        
    Returns:
        Configured parameter generator instance
    """
    if optimizer_type == "optuna":
        from .generators.optuna_generator import OptunaParameterGenerator
        return OptunaParameterGenerator(random_state=random_state, **kwargs)
    
    elif optimizer_type == "genetic":
        from .generators.genetic_generator import GeneticParameterGenerator
        return GeneticParameterGenerator(random_state=random_state, **kwargs)
    
    elif optimizer_type == "bayesian":
        from .generators.custom_generator import BayesianOptimizationGenerator
        return BayesianOptimizationGenerator(random_state=random_state, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Supported types: optuna, genetic, bayesian"
        )
```

### Step 3: Use the New Generator

```python
# Example usage of the new Bayesian optimization generator

from portfolio_backtester.optimization.factory import create_parameter_generator
from portfolio_backtester.optimization.orchestrator import OptimizationOrchestrator
from portfolio_backtester.optimization.evaluator import BacktestEvaluator

# Create Bayesian optimization generator
bayesian_generator = create_parameter_generator(
    "bayesian",
    random_state=42,
    n_initial_points=15,
    acquisition_function='EI'  # Expected Improvement
)

# Setup orchestrator
evaluator = BacktestEvaluator(strategy_backtester, n_jobs=4)
orchestrator = OptimizationOrchestrator(bayesian_generator, evaluator)

# Run optimization
result = orchestrator.optimize(optimization_data, n_trials=100)
print(f"Bayesian optimization best objective: {result.best_objective:.4f}")
```

## Adding Custom Evaluation Metrics

### Custom Metric Calculator

```python
# src/portfolio_backtester/optimization/custom_metrics.py

import pandas as pd
import numpy as np
from typing import Dict, Any

class CustomMetricsCalculator:
    """
    Custom metrics calculator for specialized performance measures.
    
    This example shows how to add domain-specific metrics to the evaluation process.
    """
    
    @staticmethod
    def calculate_tail_ratio(returns: pd.Series, percentile: float = 0.05) -> float:
        """
        Calculate tail ratio: average of top percentile / average of bottom percentile.
        
        Args:
            returns: Portfolio returns series
            percentile: Percentile threshold (e.g., 0.05 for 5%)
            
        Returns:
            Tail ratio value
        """
        if len(returns) == 0:
            return 0.0
        
        sorted_returns = returns.sort_values()
        n = len(sorted_returns)
        
        bottom_n = max(1, int(n * percentile))
        top_n = max(1, int(n * percentile))
        
        bottom_avg = sorted_returns.iloc[:bottom_n].mean()
        top_avg = sorted_returns.iloc[-top_n:].mean()
        
        if bottom_avg == 0:
            return np.inf if top_avg > 0 else 0.0
        
        return abs(top_avg / bottom_avg)
    
    @staticmethod
    def calculate_pain_index(returns: pd.Series) -> float:
        """
        Calculate pain index: average drawdown over the period.
        
        Args:
            returns: Portfolio returns series
            
        Returns:
            Pain index value
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        
        return abs(drawdowns.mean())
    
    @staticmethod
    def calculate_ulcer_index(returns: pd.Series) -> float:
        """
        Calculate Ulcer Index: RMS of drawdowns.
        
        Args:
            returns: Portfolio returns series
            
        Returns:
            Ulcer index value
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        
        return np.sqrt((drawdowns ** 2).mean())
    
    @classmethod
    def calculate_all_custom_metrics(cls, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate all custom metrics for a returns series.
        
        Args:
            returns: Portfolio returns series
            
        Returns:
            Dictionary of custom metric values
        """
        return {
            'tail_ratio': cls.calculate_tail_ratio(returns),
            'pain_index': cls.calculate_pain_index(returns),
            'ulcer_index': cls.calculate_ulcer_index(returns)
        }
```

### Extended Evaluator with Custom Metrics

```python
# src/portfolio_backtester/optimization/extended_evaluator.py

from typing import Dict, Any
from .evaluator import BacktestEvaluator
from .custom_metrics import CustomMetricsCalculator

class ExtendedBacktestEvaluator(BacktestEvaluator):
    """
    Extended evaluator that includes custom metrics in evaluation results.
    
    This shows how to extend the evaluation process with additional metrics
    without modifying the core architecture.
    """
    
    def __init__(self, strategy_backtester, n_jobs: int = 1, include_custom_metrics: bool = True):
        """
        Initialize extended evaluator.
        
        Args:
            strategy_backtester: Pure backtesting engine
            n_jobs: Number of parallel jobs
            include_custom_metrics: Whether to calculate custom metrics
        """
        super().__init__(strategy_backtester, n_jobs)
        self.include_custom_metrics = include_custom_metrics
        self.custom_calculator = CustomMetricsCalculator()
    
    def _enhance_window_result(self, window_result, returns_series):
        """
        Enhance window result with custom metrics.
        
        Args:
            window_result: Original window result
            returns_series: Portfolio returns for the window
            
        Returns:
            Enhanced window result with custom metrics
        """
        if not self.include_custom_metrics:
            return window_result
        
        # Calculate custom metrics
        custom_metrics = self.custom_calculator.calculate_all_custom_metrics(returns_series)
        
        # Add to existing metrics
        enhanced_metrics = window_result.metrics.copy()
        enhanced_metrics.update(custom_metrics)
        
        # Create new window result with enhanced metrics
        from ..backtesting.results import WindowResult
        
        return WindowResult(
            start_date=window_result.start_date,
            end_date=window_result.end_date,
            returns=window_result.returns,
            metrics=enhanced_metrics,
            trades=window_result.trades
        )
    
    def evaluate_parameters(self, parameters: Dict[str, Any], optimization_data) -> 'EvaluationResult':
        """
        Evaluate parameters with custom metrics included.
        
        Args:
            parameters: Parameter set to evaluate
            optimization_data: Data container with all required information
            
        Returns:
            Evaluation result with custom metrics
        """
        # Get base evaluation result
        base_result = super().evaluate_parameters(parameters, optimization_data)
        
        if not self.include_custom_metrics:
            return base_result
        
        # Enhance window results with custom metrics
        enhanced_window_results = []
        
        for window_result in base_result.window_results:
            enhanced_result = self._enhance_window_result(
                window_result, 
                window_result.returns
            )
            enhanced_window_results.append(enhanced_result)
        
        # Recalculate aggregated metrics including custom ones
        aggregated_metrics = self._aggregate_window_metrics(enhanced_window_results)
        
        # Create enhanced evaluation result
        from .parameter_generator import EvaluationResult
        
        return EvaluationResult(
            parameters=base_result.parameters,
            objective_value=base_result.objective_value,
            metrics=aggregated_metrics,
            window_results=enhanced_window_results,
            evaluation_time=base_result.evaluation_time
        )
```

## Plugin Architecture Example

### Plugin Interface

```python
# src/portfolio_backtester/plugins/base_plugin.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class OptimizationPlugin(ABC):
    """
    Base class for optimization plugins.
    
    Plugins can extend the optimization process with additional functionality
    such as custom stopping criteria, result post-processing, or specialized
    parameter generation strategies.
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the plugin name."""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Plugin configuration dictionary
        """
        pass
    
    def pre_optimization(self, optimization_data) -> Optional[Dict[str, Any]]:
        """
        Called before optimization starts.
        
        Args:
            optimization_data: Optimization data container
            
        Returns:
            Optional modifications to optimization data
        """
        return None
    
    def post_trial(self, trial_result, trial_number: int) -> Optional[Dict[str, Any]]:
        """
        Called after each trial completes.
        
        Args:
            trial_result: Result from the trial
            trial_number: Trial number
            
        Returns:
            Optional additional data to store
        """
        return None
    
    def should_stop(self, optimization_history: list) -> bool:
        """
        Check if optimization should stop early.
        
        Args:
            optimization_history: List of completed trials
            
        Returns:
            True if optimization should stop
        """
        return False
    
    def post_optimization(self, optimization_result) -> Optional[Dict[str, Any]]:
        """
        Called after optimization completes.
        
        Args:
            optimization_result: Final optimization result
            
        Returns:
            Optional additional result data
        """
        return None
```

### Example Plugin Implementation

```python
# src/portfolio_backtester/plugins/early_stopping_plugin.py

import numpy as np
from typing import Dict, Any, List, Optional
from .base_plugin import OptimizationPlugin

class EarlyStoppingPlugin(OptimizationPlugin):
    """
    Plugin that implements early stopping based on convergence criteria.
    
    This plugin monitors optimization progress and stops early if no
    improvement is seen for a specified number of trials.
    """
    
    def __init__(self):
        self.patience = 20
        self.min_improvement = 0.001
        self.best_objective = float('-inf')
        self.trials_without_improvement = 0
        self.improvement_history = []
    
    def get_name(self) -> str:
        """Return the plugin name."""
        return "early_stopping"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize plugin with configuration.
        
        Args:
            config: Configuration with 'patience' and 'min_improvement' keys
        """
        self.patience = config.get('patience', 20)
        self.min_improvement = config.get('min_improvement', 0.001)
        self.best_objective = float('-inf')
        self.trials_without_improvement = 0
        self.improvement_history = []
    
    def post_trial(self, trial_result, trial_number: int) -> Optional[Dict[str, Any]]:
        """
        Check for improvement after each trial.
        
        Args:
            trial_result: Result from the completed trial
            trial_number: Trial number
            
        Returns:
            Dictionary with convergence information
        """
        current_objective = trial_result.objective_value
        
        # Check for improvement
        if current_objective > self.best_objective + self.min_improvement:
            improvement = current_objective - self.best_objective
            self.best_objective = current_objective
            self.trials_without_improvement = 0
            self.improvement_history.append(improvement)
        else:
            self.trials_without_improvement += 1
            self.improvement_history.append(0.0)
        
        return {
            'trials_without_improvement': self.trials_without_improvement,
            'best_objective': self.best_objective,
            'improvement': self.improvement_history[-1]
        }
    
    def should_stop(self, optimization_history: List) -> bool:
        """
        Check if optimization should stop due to lack of improvement.
        
        Args:
            optimization_history: List of completed trials
            
        Returns:
            True if no improvement for 'patience' trials
        """
        return self.trials_without_improvement >= self.patience
```

### Plugin-Enabled Orchestrator

```python
# src/portfolio_backtester/optimization/plugin_orchestrator.py

from typing import List, Optional, Dict, Any
from .orchestrator import OptimizationOrchestrator
from ..plugins.base_plugin import OptimizationPlugin

class PluginEnabledOrchestrator(OptimizationOrchestrator):
    """
    Orchestrator that supports plugins for extended functionality.
    
    This shows how to extend the core orchestrator to support
    a plugin architecture without modifying the base implementation.
    """
    
    def __init__(
        self,
        parameter_generator,
        evaluator,
        progress_tracker=None,
        plugins: Optional[List[OptimizationPlugin]] = None
    ):
        """
        Initialize orchestrator with plugins.
        
        Args:
            parameter_generator: Parameter generation strategy
            evaluator: Backtest evaluation component
            progress_tracker: Progress tracking component
            plugins: List of optimization plugins
        """
        super().__init__(parameter_generator, evaluator, progress_tracker)
        self.plugins = plugins or []
        self.plugin_data = {}
    
    def optimize(self, optimization_data, n_trials: int = 100):
        """
        Run optimization with plugin support.
        
        Args:
            optimization_data: All data needed for optimization
            n_trials: Number of optimization trials to run
            
        Returns:
            Optimization result with plugin data
        """
        # Initialize plugins
        for plugin in self.plugins:
            plugin_config = getattr(optimization_data, f'{plugin.get_name()}_config', {})
            plugin.initialize(plugin_config)
        
        # Pre-optimization plugin hooks
        for plugin in self.plugins:
            modifications = plugin.pre_optimization(optimization_data)
            if modifications:
                # Apply modifications to optimization_data
                for key, value in modifications.items():
                    setattr(optimization_data, key, value)
        
        # Run optimization with plugin hooks
        optimization_history = []
        
        for trial_num in range(n_trials):
            # Get parameters and evaluate
            parameters = self.parameter_generator.suggest_parameters(
                optimization_data.optimization_spec
            )
            
            result = self.evaluator.evaluate_parameters(parameters, optimization_data)
            
            # Report result to generator
            self.parameter_generator.report_result(parameters, result)
            
            # Store trial result
            optimization_history.append(result)
            
            # Post-trial plugin hooks
            for plugin in self.plugins:
                plugin_result = plugin.post_trial(result, trial_num)
                if plugin_result:
                    self.plugin_data[f'{plugin.get_name()}_trial_{trial_num}'] = plugin_result
            
            # Check for early stopping
            should_stop = any(
                plugin.should_stop(optimization_history) 
                for plugin in self.plugins
            )
            
            if should_stop:
                print(f"Early stopping triggered at trial {trial_num}")
                break
            
            # Check if generator wants to stop
            if self.parameter_generator.is_complete():
                break
        
        # Create optimization result
        from .parameter_generator import OptimizationResult
        
        result = OptimizationResult(
            best_parameters=self.parameter_generator.get_best_parameters(),
            best_objective=max(r.objective_value for r in optimization_history),
            optimization_history=optimization_history,
            convergence_data=self.plugin_data,
            total_time=0.0  # Would calculate actual time
        )
        
        # Post-optimization plugin hooks
        for plugin in self.plugins:
            plugin_result = plugin.post_optimization(result)
            if plugin_result:
                self.plugin_data[f'{plugin.get_name()}_final'] = plugin_result
        
        return result
```

### Using the Plugin System

```python
# Example usage of the plugin system

from portfolio_backtester.optimization.plugin_orchestrator import PluginEnabledOrchestrator
from portfolio_backtester.plugins.early_stopping_plugin import EarlyStoppingPlugin

# Create plugins
early_stopping = EarlyStoppingPlugin()

# Create orchestrator with plugins
plugin_orchestrator = PluginEnabledOrchestrator(
    parameter_generator=generator,
    evaluator=evaluator,
    plugins=[early_stopping]
)

# Configure plugin through optimization data
optimization_data.early_stopping_config = {
    'patience': 15,
    'min_improvement': 0.005
}

# Run optimization with plugins
result = plugin_orchestrator.optimize(optimization_data, n_trials=200)

# Access plugin data
convergence_info = result.convergence_data
print(f"Optimization stopped early: {len(result.optimization_history) < 200}")
```

## Best Practices for Extensions

### 1. Follow the Interface Contracts

Always implement the required abstract methods and maintain the expected behavior:

```python
# Good: Proper interface implementation
class MyParameterGenerator(ParameterGenerator):
    def suggest_parameters(self, optimization_spec):
        # Always return a valid parameter dictionary
        return self._generate_parameters(optimization_spec)
    
    def report_result(self, parameters, result):
        # Always handle the result, even if just storing it
        self._store_result(parameters, result)
    
    # ... implement other required methods
```

### 2. Handle Edge Cases Gracefully

```python
# Good: Robust error handling
def suggest_parameters(self, optimization_spec):
    try:
        if not optimization_spec:
            raise ValueError("Empty optimization specification")
        
        parameters = self._generate_parameters(optimization_spec)
        
        # Validate generated parameters
        self._validate_parameters(parameters, optimization_spec)
        
        return parameters
        
    except Exception as e:
        logger.error(f"Parameter generation failed: {e}")
        # Return fallback parameters or re-raise
        raise
```

### 3. Maintain Backward Compatibility

```python
# Good: Backward compatible factory extension
def create_parameter_generator(optimizer_type, random_state=None, **kwargs):
    # Support legacy parameter names
    if 'seed' in kwargs and random_state is None:
        random_state = kwargs.pop('seed')
    
    # Handle new optimizer types
    if optimizer_type in ['optuna', 'genetic']:
        # Existing implementation
        pass
    elif optimizer_type in ['bayesian', 'skopt']:
        # New implementation with alias support
        return BayesianOptimizationGenerator(random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
```

### 4. Document Your Extensions

```python
class CustomParameterGenerator(ParameterGenerator):
    """
    Custom parameter generator using [specific algorithm].
    
    This generator implements [algorithm description] and is particularly
    well-suited for [use cases].
    
    Args:
        random_state: Random seed for reproducibility
        custom_param: Description of custom parameter
        
    Example:
        >>> generator = CustomParameterGenerator(random_state=42)
        >>> orchestrator = OptimizationOrchestrator(generator, evaluator)
        >>> result = orchestrator.optimize(data, n_trials=100)
    
    References:
        - [Paper or documentation reference]
        - [Implementation reference]
    """
```

This extensibility guide provides comprehensive examples for extending the Portfolio Backtester architecture while maintaining clean separation of concerns and following established patterns.