# API Documentation

## Overview

This document provides comprehensive API documentation for all public interfaces in the refactored Portfolio Backtester architecture. Each component's public methods, parameters, return types, and usage examples are documented.

## Core Components

### StrategyBacktester

The pure backtesting engine that handles strategy execution and performance calculation.

**Location**: `src/portfolio_backtester/core.py`

#### Constructor

```python
def __init__(self, global_config: Dict[str, Any], data_source: Any) -> None
```

**Parameters**:
- `global_config` (Dict[str, Any]): Global configuration dictionary containing system-wide settings
- `data_source` (Any): Data source object for loading market data

**Example**:
```python
from portfolio_backtester.core import StrategyBacktester

config = {
    'data_path': 'data/',
    'cache_enabled': True,
    'transaction_costs': 0.001
}
backtester = StrategyBacktester(config, data_source)
```

#### backtest_strategy

```python
def backtest_strategy(
    self,
    strategy_config: Dict[str, Any],
    monthly_data: pd.DataFrame,
    daily_data: pd.DataFrame,
    rets_full: pd.DataFrame
) -> BacktestResult
```

Execute a complete backtest for a strategy with given parameters.

**Parameters**:
- `strategy_config` (Dict[str, Any]): Strategy configuration including parameters and settings
- `monthly_data` (pd.DataFrame): Monthly market data with OHLCV columns
- `daily_data` (pd.DataFrame): Daily market data with OHLCV columns  
- `rets_full` (pd.DataFrame): Full returns data for the universe

**Returns**:
- `BacktestResult`: Complete backtest results with metrics, trades, and charts

**Example**:
```python
strategy_config = {
    'strategy_name': 'momentum_strategy',
    'lookback_period': 12,
    'rebalance_frequency': 'monthly',
    'position_size': 0.1
}

result = backtester.backtest_strategy(
    strategy_config, monthly_data, daily_data, returns_data
)

print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.3f}")
print(f"Total Return: {result.metrics['total_return']:.2%}")
```

#### evaluate_window

```python
def evaluate_window(
    self,
    strategy_config: Dict[str, Any],
    window: Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp],
    monthly_data: pd.DataFrame,
    daily_data: pd.DataFrame,
    rets_full: pd.DataFrame
) -> WindowResult
```

Evaluate strategy performance for a single walk-forward window.

**Parameters**:
- `strategy_config` (Dict[str, Any]): Strategy configuration
- `window` (Tuple[pd.Timestamp, ...]): Window definition (train_start, train_end, test_start, test_end)
- `monthly_data` (pd.DataFrame): Monthly market data
- `daily_data` (pd.DataFrame): Daily market data
- `rets_full` (pd.DataFrame): Returns data

**Returns**:
- `WindowResult`: Single window evaluation results

**Example**:
```python
window = (
    pd.Timestamp('2020-01-01'),  # train_start
    pd.Timestamp('2021-12-31'),  # train_end
    pd.Timestamp('2022-01-01'),  # test_start
    pd.Timestamp('2022-12-31')   # test_end
)

window_result = backtester.evaluate_window(
    strategy_config, window, monthly_data, daily_data, returns_data
)

print(f"Window Sharpe: {window_result.metrics['sharpe_ratio']:.3f}")
```

### OptimizationOrchestrator

Coordinates the optimization process between parameter generators and evaluators.

**Location**: `src/portfolio_backtester/optimization/orchestrator.py`

#### Constructor

```python
def __init__(
    self,
    parameter_generator: ParameterGenerator,
    evaluator: BacktestEvaluator
) -> None
```

**Parameters**:
- `parameter_generator` (ParameterGenerator): Parameter generation strategy
- `evaluator` (BacktestEvaluator): Backtest evaluation component

**Example**:
```python
from portfolio_backtester.optimization.orchestrator import OptimizationOrchestrator
from portfolio_backtester.optimization.factory import create_parameter_generator
from portfolio_backtester.optimization.evaluator import BacktestEvaluator

generator = create_parameter_generator("optuna", random_state=42)
evaluator = BacktestEvaluator(["sharpe_ratio"], is_multi_objective=False)
orchestrator = OptimizationOrchestrator(generator, evaluator)
```

#### optimize

```python
def optimize(
    self,
    scenario_config: Dict[str, Any],
    optimization_config: Dict[str, Any],
    data: OptimizationData
) -> OptimizationResult
```

Execute the complete optimization process.

**Parameters**:
- `scenario_config` (Dict[str, Any]): Scenario configuration with strategy settings
- `optimization_config` (Dict[str, Any]): Optimization settings and parameter bounds
- `data` (OptimizationData): Market data and walk-forward windows

**Returns**:
- `OptimizationResult`: Final optimization results with best parameters

**Example**:
```python
scenario_config = {
    'strategy_name': 'momentum_strategy',
    'universe': 'sp500',
    'rebalance_frequency': 'monthly'
}

optimization_config = {
    'n_trials': 100,
    'timeout': 3600,
    'parameters': {
        'lookback_period': {'type': 'int', 'low': 3, 'high': 24},
        'position_size': {'type': 'float', 'low': 0.05, 'high': 0.2}
    },
    'objective': 'sharpe_ratio',
    'direction': 'maximize'
}

result = orchestrator.optimize(scenario_config, optimization_config, data)
print(f"Best parameters: {result.best_parameters}")
print(f"Best value: {result.best_value:.3f}")
```

### ParameterGenerator Interface

Abstract base class defining the contract for parameter generation strategies.

**Location**: `src/portfolio_backtester/optimization/parameter_generator.py`

#### initialize

```python
@abstractmethod
def initialize(
    self,
    scenario_config: Dict[str, Any],
    optimization_config: Dict[str, Any]
) -> None
```

Initialize the parameter generator with configuration.

**Parameters**:
- `scenario_config` (Dict[str, Any]): Scenario configuration
- `optimization_config` (Dict[str, Any]): Optimization configuration with parameter bounds

#### suggest_parameters

```python
@abstractmethod
def suggest_parameters(self) -> Dict[str, Any]
```

Suggest the next parameter set to evaluate.

**Returns**:
- `Dict[str, Any]`: Parameter dictionary with suggested values

#### report_result

```python
@abstractmethod
def report_result(
    self,
    parameters: Dict[str, Any],
    result: EvaluationResult
) -> None
```

Report evaluation result back to the parameter generator.

**Parameters**:
- `parameters` (Dict[str, Any]): Parameters that were evaluated
- `result` (EvaluationResult): Evaluation result

#### is_finished

```python
@abstractmethod
def is_finished(self) -> bool
```

Check if optimization should continue.

**Returns**:
- `bool`: True if optimization is complete, False otherwise

#### get_best_result

```python
@abstractmethod
def get_best_result(self) -> OptimizationResult
```

Get the best result found so far.

**Returns**:
- `OptimizationResult`: Best optimization result

### OptunaParameterGenerator

Optuna-based parameter generator implementation.

**Location**: `src/portfolio_backtester/optimization/generators/optuna_generator.py`

#### Constructor

```python
def __init__(self, random_state: Optional[int] = None) -> None
```

**Parameters**:
- `random_state` (Optional[int]): Random seed for reproducible results

**Example**:
```python
from portfolio_backtester.optimization.generators.optuna_generator import OptunaParameterGenerator

generator = OptunaParameterGenerator(random_state=42)
```

#### Usage Example

```python
# Initialize with optimization configuration
optimization_config = {
    'n_trials': 100,
    'parameters': {
        'lookback_period': {'type': 'int', 'low': 3, 'high': 24},
        'momentum_threshold': {'type': 'float', 'low': 0.0, 'high': 0.1}
    },
    'objective': 'sharpe_ratio',
    'direction': 'maximize'
}

generator.initialize(scenario_config, optimization_config)

# Optimization loop
while not generator.is_finished():
    parameters = generator.suggest_parameters()
    # Evaluate parameters...
    result = evaluate_parameters(parameters)
    generator.report_result(parameters, result)

best_result = generator.get_best_result()
```

### GeneticParameterGenerator

PyGAD-based genetic algorithm parameter generator.

**Location**: `src/portfolio_backtester/optimization/generators/genetic_generator.py`

#### Constructor

```python
def __init__(self, random_state: Optional[int] = None) -> None
```

**Parameters**:
- `random_state` (Optional[int]): Random seed for reproducible results

**Example**:
```python
from portfolio_backtester.optimization.generators.genetic_generator import GeneticParameterGenerator

generator = GeneticParameterGenerator(random_state=42)
```

#### Usage Example

```python
# Initialize with genetic algorithm configuration
optimization_config = {
    'num_generations': 50,
    'population_size': 20,
    'parameters': {
        'lookback_period': {'type': 'int', 'low': 3, 'high': 24},
        'position_size': {'type': 'float', 'low': 0.05, 'high': 0.2}
    },
    'objective': 'sharpe_ratio',
    'direction': 'maximize'
}

generator.initialize(scenario_config, optimization_config)

# Optimization loop (same interface as Optuna)
while not generator.is_finished():
    parameters = generator.suggest_parameters()
    result = evaluate_parameters(parameters)
    generator.report_result(parameters, result)

best_result = generator.get_best_result()
```

### BacktestEvaluator

Performs walk-forward analysis for parameter sets.

**Location**: `src/portfolio_backtester/optimization/evaluator.py`

#### Constructor

```python
def __init__(
    self,
    metrics_to_optimize: List[str],
    is_multi_objective: bool = False
) -> None
```

**Parameters**:
- `metrics_to_optimize` (List[str]): List of metrics to optimize (e.g., ['sharpe_ratio'])
- `is_multi_objective` (bool): Whether this is multi-objective optimization

**Example**:
```python
from portfolio_backtester.optimization.evaluator import BacktestEvaluator

# Single objective
evaluator = BacktestEvaluator(["sharpe_ratio"], is_multi_objective=False)

# Multi-objective
evaluator = BacktestEvaluator(
    ["sharpe_ratio", "max_drawdown"], 
    is_multi_objective=True
)
```

#### evaluate_parameters

```python
def evaluate_parameters(
    self,
    parameters: Dict[str, Any],
    scenario_config: Dict[str, Any],
    data: OptimizationData,
    backtester: StrategyBacktester
) -> EvaluationResult
```

Evaluate a parameter set across all walk-forward windows.

**Parameters**:
- `parameters` (Dict[str, Any]): Parameters to evaluate
- `scenario_config` (Dict[str, Any]): Scenario configuration
- `data` (OptimizationData): Market data and windows
- `backtester` (StrategyBacktester): Backtesting engine

**Returns**:
- `EvaluationResult`: Aggregated evaluation results

**Example**:
```python
parameters = {
    'lookback_period': 12,
    'position_size': 0.1,
    'rebalance_frequency': 'monthly'
}

result = evaluator.evaluate_parameters(
    parameters, scenario_config, data, backtester
)

print(f"Objective value: {result.objective_value:.3f}")
print(f"Metrics: {result.metrics}")
```

## Data Classes

### BacktestResult

Complete backtest results with all performance data.

**Location**: `src/portfolio_backtester/optimization/evaluator.py`

```python
@dataclass
class BacktestResult:
    returns: pd.Series              # Strategy returns time series
    metrics: Dict[str, float]       # Performance metrics
    trade_history: pd.DataFrame     # Trade execution history
    performance_stats: Dict[str, Any]  # Additional performance statistics
    charts_data: Dict[str, Any]     # Data for generating charts
```

**Example**:
```python
result = backtester.backtest_strategy(config, monthly, daily, returns)

# Access returns
print(f"Final return: {result.returns.iloc[-1]:.2%}")

# Access metrics
sharpe = result.metrics['sharpe_ratio']
max_dd = result.metrics['max_drawdown']

# Access trade history
print(f"Number of trades: {len(result.trade_history)}")
```

### WindowResult

Single walk-forward window evaluation result.

```python
@dataclass
class WindowResult:
    window_returns: pd.Series       # Returns for this window
    metrics: Dict[str, float]       # Window-specific metrics
    train_start: pd.Timestamp       # Training period start
    train_end: pd.Timestamp         # Training period end
    test_start: pd.Timestamp        # Test period start
    test_end: pd.Timestamp          # Test period end
```

### EvaluationResult

Result of evaluating a parameter set across all windows.

```python
@dataclass
class EvaluationResult:
    objective_value: Union[float, List[float]]  # Single or multi-objective value
    metrics: Dict[str, float]                   # Aggregated metrics
    window_results: List[WindowResult]          # Individual window results
```

### OptimizationResult

Final optimization result with best parameters.

```python
@dataclass
class OptimizationResult:
    best_parameters: Dict[str, Any]             # Best parameter set found
    best_value: Union[float, List[float]]       # Best objective value(s)
    n_evaluations: int                          # Number of evaluations performed
    optimization_history: List[Dict[str, Any]]  # History of all evaluations
    best_trial: Any = None                      # Backend-specific best trial object
```

### OptimizationData

Data container for optimization process.

```python
@dataclass
class OptimizationData:
    monthly: pd.DataFrame           # Monthly market data
    daily: pd.DataFrame             # Daily market data
    returns: pd.DataFrame           # Returns data
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]  # Walk-forward windows
```

## Factory Functions

### create_parameter_generator

Factory function to create parameter generators.

**Location**: `src/portfolio_backtester/optimization/factory.py`

```python
def create_parameter_generator(
    optimizer_type: str,
    random_state: Optional[int] = None
) -> ParameterGenerator
```

**Parameters**:
- `optimizer_type` (str): Type of optimizer ("optuna", "genetic", "mock")
- `random_state` (Optional[int]): Random seed for reproducible results

**Returns**:
- `ParameterGenerator`: Configured parameter generator instance

**Raises**:
- `ValueError`: If optimizer_type is not recognized

**Example**:
```python
from portfolio_backtester.optimization.factory import create_parameter_generator

# Create Optuna generator
optuna_gen = create_parameter_generator("optuna", random_state=42)

# Create genetic algorithm generator
genetic_gen = create_parameter_generator("genetic", random_state=42)

# Create mock generator for testing
mock_gen = create_parameter_generator("mock", random_state=42)
```

## Configuration Formats

### Optimization Configuration

```python
optimization_config = {
    # Basic settings
    'n_trials': 100,                    # Number of optimization trials
    'timeout': 3600,                    # Timeout in seconds
    'random_state': 42,                 # Random seed
    
    # Parameter space definition
    'parameters': {
        'lookback_period': {
            'type': 'int',              # Parameter type: 'int', 'float', 'categorical'
            'low': 3,                   # Lower bound (for int/float)
            'high': 24                  # Upper bound (for int/float)
        },
        'position_size': {
            'type': 'float',
            'low': 0.05,
            'high': 0.2,
            'step': 0.01               # Optional step size
        },
        'rebalance_freq': {
            'type': 'categorical',
            'choices': ['monthly', 'quarterly', 'annually']  # Categorical choices
        }
    },
    
    # Optimization objective
    'objective': 'sharpe_ratio',        # Single objective
    'direction': 'maximize',            # 'maximize' or 'minimize'
    
    # Multi-objective (alternative to single objective)
    'objectives': [
        {'metric': 'sharpe_ratio', 'direction': 'maximize'},
        {'metric': 'max_drawdown', 'direction': 'minimize'}
    ],
    
    # Backend-specific settings
    'optuna_settings': {
        'sampler': 'TPE',               # Optuna sampler type
        'pruner': 'MedianPruner',       # Optuna pruner type
        'study_name': 'my_study'        # Study name for persistence
    },
    
    'genetic_settings': {
        'population_size': 20,          # GA population size
        'num_generations': 50,          # Number of generations
        'mutation_rate': 0.1,           # Mutation rate
        'crossover_rate': 0.8           # Crossover rate
    }
}
```

### Scenario Configuration

```python
scenario_config = {
    # Strategy settings
    'strategy_name': 'momentum_strategy',
    'universe': 'sp500',
    'rebalance_frequency': 'monthly',
    
    # Data settings
    'start_date': '2010-01-01',
    'end_date': '2023-12-31',
    'benchmark': 'SPY',
    
    # Walk-forward settings
    'train_period_months': 24,          # Training period length
    'test_period_months': 6,            # Test period length
    'step_months': 3,                   # Step size between windows
    
    # Risk management
    'max_position_size': 0.1,
    'stop_loss': 0.05,
    'transaction_costs': 0.001,
    
    # Strategy-specific parameters (will be optimized)
    'lookback_period': 12,              # Default value
    'momentum_threshold': 0.05,         # Default value
    'position_size': 0.08               # Default value
}
```

## Error Handling

### Exception Types

```python
# Base exceptions
class OptimizationError(Exception):
    """Base exception for optimization-related errors"""
    pass

class ParameterGenerationError(OptimizationError):
    """Raised when parameter generation fails"""
    pass

class EvaluationError(OptimizationError):
    """Raised when parameter evaluation fails"""
    pass

class BacktestError(Exception):
    """Base exception for backtesting-related errors"""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass
```

### Error Handling Examples

```python
try:
    generator = create_parameter_generator("unknown_type")
except ValueError as e:
    print(f"Invalid optimizer type: {e}")

try:
    result = orchestrator.optimize(scenario_config, optimization_config, data)
except OptimizationError as e:
    print(f"Optimization failed: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Type Hints and Validation

### Type Annotations

All public methods include comprehensive type hints:

```python
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd

def backtest_strategy(
    self,
    strategy_config: Dict[str, Any],
    monthly_data: pd.DataFrame,
    daily_data: pd.DataFrame,
    rets_full: pd.DataFrame
) -> BacktestResult:
    """Type hints provide clear interface contracts"""
    pass
```

### Parameter Validation

```python
def validate_optimization_config(config: Dict[str, Any]) -> None:
    """Validate optimization configuration"""
    required_keys = ['parameters', 'objective', 'direction']
    
    for key in required_keys:
        if key not in config:
            raise ConfigurationError(f"Missing required key: {key}")
    
    # Validate parameter specifications
    for param_name, param_spec in config['parameters'].items():
        if 'type' not in param_spec:
            raise ConfigurationError(f"Parameter {param_name} missing type")
        
        param_type = param_spec['type']
        if param_type in ['int', 'float']:
            if 'low' not in param_spec or 'high' not in param_spec:
                raise ConfigurationError(f"Parameter {param_name} missing bounds")
        elif param_type == 'categorical':
            if 'choices' not in param_spec:
                raise ConfigurationError(f"Parameter {param_name} missing choices")
```

## Optimization Components

### OptimizationOrchestrator

The central coordinator for optimization workflows that manages parameter generators and evaluation.

**Location**: `src/portfolio_backtester/optimization/orchestrator.py`

#### Constructor

```python
def __init__(
    self,
    parameter_generator: ParameterGenerator,
    evaluator: BacktestEvaluator,
    progress_tracker: Optional[ProgressTracker] = None
) -> None
```

**Parameters**:
- `parameter_generator` (ParameterGenerator): Parameter generation strategy (Optuna, Genetic, etc.)
- `evaluator` (BacktestEvaluator): Backtest evaluation component
- `progress_tracker` (Optional[ProgressTracker]): Progress tracking component

#### optimize

```python
def optimize(
    self,
    optimization_data: OptimizationData,
    n_trials: int = 100
) -> OptimizationResult
```

Execute the complete optimization workflow.

**Parameters**:
- `optimization_data` (OptimizationData): All data needed for optimization
- `n_trials` (int): Number of optimization trials to run

**Returns**:
- `OptimizationResult`: Complete optimization results with best parameters and metrics

**Example**:
```python
from portfolio_backtester.optimization.orchestrator import OptimizationOrchestrator
from portfolio_backtester.optimization.factory import create_parameter_generator
from portfolio_backtester.optimization.evaluator import BacktestEvaluator

# Create components
generator = create_parameter_generator("optuna", random_state=42)
evaluator = BacktestEvaluator(strategy_backtester)
orchestrator = OptimizationOrchestrator(generator, evaluator)

# Run optimization
result = orchestrator.optimize(optimization_data, n_trials=50)
print(f"Best parameters: {result.best_parameters}")
```

### ParameterGenerator (Abstract Base Class)

Interface for all parameter generation strategies.

**Location**: `src/portfolio_backtester/optimization/parameter_generator.py`

#### Abstract Methods

```python
def suggest_parameters(self, optimization_spec: OptimizationSpec) -> Dict[str, Any]
```

Generate the next set of parameters to evaluate.

```python
def report_result(self, parameters: Dict[str, Any], result: EvaluationResult) -> None
```

Report evaluation results back to the generator.

```python
def is_complete(self) -> bool
```

Check if optimization should continue.

```python
def get_best_parameters(self) -> Dict[str, Any]
```

Get the best parameters found so far.

### OptunaParameterGenerator

Optuna-based parameter generation with TPE sampling and pruning.

**Location**: `src/portfolio_backtester/optimization/generators/optuna_generator.py`

#### Constructor

```python
def __init__(
    self,
    random_state: Optional[int] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None
) -> None
```

**Parameters**:
- `random_state` (Optional[int]): Random seed for reproducibility
- `study_name` (Optional[str]): Name for the Optuna study
- `storage` (Optional[str]): Storage backend for study persistence

**Example**:
```python
from portfolio_backtester.optimization.generators.optuna_generator import OptunaParameterGenerator

# Create with persistence
generator = OptunaParameterGenerator(
    random_state=42,
    study_name="momentum_optimization",
    storage="sqlite:///optimization.db"
)
```

### GeneticParameterGenerator

PyGAD-based genetic algorithm parameter generation.

**Location**: `src/portfolio_backtester/optimization/generators/genetic_generator.py`

#### Constructor

```python
def __init__(
    self,
    random_state: Optional[int] = None,
    population_size: int = 50,
    num_generations: int = 100,
    mutation_probability: float = 0.1
) -> None
```

**Parameters**:
- `random_state` (Optional[int]): Random seed for reproducibility
- `population_size` (int): Size of the genetic algorithm population
- `num_generations` (int): Number of generations to evolve
- `mutation_probability` (float): Probability of mutation for each gene

**Example**:
```python
from portfolio_backtester.optimization.generators.genetic_generator import GeneticParameterGenerator

# Create genetic optimizer
generator = GeneticParameterGenerator(
    random_state=42,
    population_size=100,
    num_generations=50,
    mutation_probability=0.15
)
```

### BacktestEvaluator

Evaluates parameter sets using walk-forward analysis.

**Location**: `src/portfolio_backtester/optimization/evaluator.py`

#### Constructor

```python
def __init__(
    self,
    strategy_backtester: StrategyBacktester,
    n_jobs: int = 1
) -> None
```

**Parameters**:
- `strategy_backtester` (StrategyBacktester): Pure backtesting engine
- `n_jobs` (int): Number of parallel jobs for evaluation

#### evaluate_parameters

```python
def evaluate_parameters(
    self,
    parameters: Dict[str, Any],
    optimization_data: OptimizationData
) -> EvaluationResult
```

Evaluate a parameter set using walk-forward analysis.

**Parameters**:
- `parameters` (Dict[str, Any]): Parameter set to evaluate
- `optimization_data` (OptimizationData): Data container with all required information

**Returns**:
- `EvaluationResult`: Evaluation results with metrics and window details

## Data Classes

### BacktestResult

Complete results from a single backtest run.

**Location**: `src/portfolio_backtester/backtesting/results.py`

**Attributes**:
- `returns` (pd.Series): Portfolio returns time series
- `metrics` (Dict[str, float]): Performance metrics (Sharpe, Sortino, etc.)
- `trade_history` (pd.DataFrame): Complete trade history
- `performance_stats` (Dict[str, Any]): Detailed performance statistics
- `charts_data` (Dict[str, Any]): Data for generating performance charts

### EvaluationResult

Results from evaluating a parameter set across walk-forward windows.

**Attributes**:
- `parameters` (Dict[str, Any]): Parameter set that was evaluated
- `objective_value` (float): Primary optimization objective value
- `metrics` (Dict[str, float]): Aggregated performance metrics
- `window_results` (List[WindowResult]): Results from each walk-forward window
- `evaluation_time` (float): Time taken for evaluation

### OptimizationResult

Final results from a complete optimization run.

**Attributes**:
- `best_parameters` (Dict[str, Any]): Best parameter set found
- `best_objective` (float): Best objective value achieved
- `optimization_history` (List[EvaluationResult]): Complete optimization history
- `convergence_data` (Dict[str, Any]): Convergence analysis data
- `total_time` (float): Total optimization time

## Factory Functions

### create_parameter_generator

Factory function for creating parameter generators.

**Location**: `src/portfolio_backtester/optimization/factory.py`

```python
def create_parameter_generator(
    optimizer_type: str,
    random_state: Optional[int] = None,
    **kwargs
) -> ParameterGenerator
```

**Parameters**:
- `optimizer_type` (str): Type of optimizer ("optuna" or "genetic")
- `random_state` (Optional[int]): Random seed for reproducibility
- `**kwargs`: Additional optimizer-specific parameters

**Returns**:
- `ParameterGenerator`: Configured parameter generator instance

**Example**:
```python
from portfolio_backtester.optimization.factory import create_parameter_generator

# Create Optuna generator
optuna_gen = create_parameter_generator(
    "optuna",
    random_state=42,
    study_name="my_study"
)

# Create genetic generator
genetic_gen = create_parameter_generator(
    "genetic",
    random_state=42,
    population_size=100,
    num_generations=50
)
```

## Configuration Options

### OptimizationSpec

Specification for parameter optimization ranges and types.

**Format**:
```python
optimization_spec = {
    "parameter_name": {
        "type": "int" | "float" | "categorical",
        "low": float,  # For int/float types
        "high": float,  # For int/float types
        "choices": List[Any],  # For categorical type
        "log": bool  # Optional, for log-scale sampling
    }
}
```

**Example**:
```python
optimization_spec = {
    "lookback_period": {
        "type": "int",
        "low": 6,
        "high": 24
    },
    "momentum_threshold": {
        "type": "float",
        "low": 0.01,
        "high": 0.1,
        "log": True
    },
    "rebalance_frequency": {
        "type": "categorical",
        "choices": ["monthly", "quarterly", "semi-annual"]
    }
}
```

### Global Configuration

System-wide configuration options.

**Required Fields**:
- `data_path` (str): Path to data directory
- `start_date` (str): Backtest start date (YYYY-MM-DD)
- `end_date` (str): Backtest end date (YYYY-MM-DD)
- `universe` (str): Trading universe specification

**Optional Fields**:
- `cache_enabled` (bool): Enable data caching (default: True)
- `transaction_costs` (float): Transaction cost in basis points (default: 0.001)
- `benchmark` (str): Benchmark ticker for comparison (default: "SPY")
- `n_jobs` (int): Number of parallel jobs (default: 1)

## Error Handling

### Common Exceptions

- `OptimizationError`: Raised when optimization fails
- `ParameterValidationError`: Raised for invalid parameter specifications
- `DataError`: Raised for data-related issues
- `ConfigurationError`: Raised for configuration problems

### Best Practices

1. **Always validate parameters** before starting optimization
2. **Use try-catch blocks** around optimization calls
3. **Check data quality** before backtesting
4. **Monitor memory usage** for large optimizations
5. **Save intermediate results** for long-running optimizations

This API documentation provides comprehensive coverage of all public interfaces, enabling developers to effectively use and extend the refactored architecture.