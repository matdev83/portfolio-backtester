# API Usage Examples

## Overview

This document provides practical examples of using the Portfolio Backtester API for common tasks. Each example includes complete, runnable code with explanations.

## Basic Backtesting

### Simple Strategy Backtest

```python
import pandas as pd
from portfolio_backtester.core import StrategyBacktester

# Load your data
monthly_data = pd.read_csv('data/monthly_prices.csv', index_col=0, parse_dates=True)
daily_data = pd.read_csv('data/daily_prices.csv', index_col=0, parse_dates=True)
returns_data = pd.read_csv('data/returns.csv', index_col=0, parse_dates=True)

# Configure the backtester
global_config = {
    'data_path': 'data/',
    'cache_enabled': True,
    'transaction_costs': 0.001,
    'benchmark': 'SPY'
}

# Create backtester instance
backtester = StrategyBacktester(global_config, data_source)

# Define strategy configuration
strategy_config = {
    'strategy': 'momentum_strategy',
    'strategy_params': {
        'lookback_period': 12,
        'momentum_threshold': 0.05,
        'position_size': 0.1
    },
    'rebalance_frequency': 'monthly'
}

# Run backtest
result = backtester.backtest_strategy(
    strategy_config,
    monthly_data,
    daily_data,
    returns_data
)

# Access results
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
```

## Parameter Optimization

### Basic Optuna Optimization

```python
from portfolio_backtester.optimization.orchestrator import OptimizationOrchestrator
from portfolio_backtester.optimization.factory import create_parameter_generator
from portfolio_backtester.optimization.evaluator import BacktestEvaluator
from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester

# Setup components
global_config = {
    'data_path': 'data/',
    'start_date': '2020-01-01',
    'end_date': '2023-12-31',
    'universe': 'sp500_top50'
}

backtester = StrategyBacktester(global_config, data_source)
evaluator = BacktestEvaluator(backtester, n_jobs=4)

# Create Optuna parameter generator
generator = create_parameter_generator(
    "optuna",
    random_state=42,
    study_name="momentum_optimization",
    storage="sqlite:///momentum_study.db"
)

# Setup orchestrator
orchestrator = OptimizationOrchestrator(generator, evaluator)

# Define optimization specification
optimization_spec = {
    "lookback_period": {
        "type": "int",
        "low": 6,
        "high": 24
    },
    "momentum_threshold": {
        "type": "float",
        "low": 0.01,
        "high": 0.2,
        "log": True
    },
    "position_size": {
        "type": "float",
        "low": 0.05,
        "high": 0.2
    }
}

# Create optimization data container
from portfolio_backtester.optimization.parameter_generator import OptimizationData

optimization_data = OptimizationData(
    scenario_config=strategy_config,
    optimization_spec=optimization_spec,
    monthly_data=monthly_data,
    daily_data=daily_data,
    rets_full=returns_data,
    walk_forward_config={
        'train_window_months': 36,
        'test_window_months': 6,
        'step_size_months': 3
    }
)

# Run optimization
result = orchestrator.optimize(optimization_data, n_trials=100)

# Access results
print(f"Best parameters: {result.best_parameters}")
print(f"Best objective: {result.best_objective:.4f}")
print(f"Total trials: {len(result.optimization_history)}")
```

### Genetic Algorithm Optimization

```python
# Create genetic algorithm parameter generator
genetic_generator = create_parameter_generator(
    "genetic",
    random_state=42,
    population_size=50,
    num_generations=30,
    mutation_probability=0.1
)

# Setup orchestrator with genetic algorithm
genetic_orchestrator = OptimizationOrchestrator(genetic_generator, evaluator)

# Run genetic optimization
genetic_result = genetic_orchestrator.optimize(optimization_data, n_trials=1500)

print(f"Genetic best parameters: {genetic_result.best_parameters}")
print(f"Genetic best objective: {genetic_result.best_objective:.4f}")
```

### Advanced Usage Examples

```python
# Custom parameter generator example
from portfolio_backtester.optimization.parameter_generator import ParameterGenerator
import random
import math

class RandomSearchGenerator(ParameterGenerator):
    """Simple random search parameter generator example."""
    
    def __init__(self, random_state=None):
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
        
        self.best_parameters = None
        self.best_objective = float('-inf')
        self.trial_count = 0
        self.max_trials = 100
    
    def suggest_parameters(self, optimization_spec):
        """Generate random parameters within specified ranges."""
        parameters = {}
        
        for param_name, spec in optimization_spec.items():
            if spec['type'] == 'int':
                parameters[param_name] = random.randint(spec['low'], spec['high'])
            elif spec['type'] == 'float':
                if spec.get('log', False):
                    log_low = math.log(spec['low'])
                    log_high = math.log(spec['high'])
                    parameters[param_name] = math.exp(random.uniform(log_low, log_high))
                else:
                    parameters[param_name] = random.uniform(spec['low'], spec['high'])
            elif spec['type'] == 'categorical':
                parameters[param_name] = random.choice(spec['choices'])
        
        return parameters
    
    def report_result(self, parameters, result):
        """Update best parameters if this result is better."""
        objective_value = result.objective_value
        
        if objective_value > self.best_objective:
            self.best_objective = objective_value
            self.best_parameters = parameters.copy()
        
        self.trial_count += 1
    
    def is_complete(self):
        """Check if we've reached the maximum number of trials."""
        return self.trial_count >= self.max_trials
    
    def get_best_parameters(self):
        """Return the best parameters found."""
        return self.best_parameters or {}

# Use custom generator
custom_generator = RandomSearchGenerator(random_state=42)
custom_orchestrator = OptimizationOrchestrator(custom_generator, evaluator)
custom_result = custom_orchestrator.optimize(optimization_data, n_trials=100)
```

## Error Handling and Best Practices

```python
import logging
from portfolio_backtester.optimization.exceptions import OptimizationError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_optimization(optimization_data, n_trials=100, max_retries=3):
    """Run optimization with error handling and retries."""
    
    for attempt in range(max_retries):
        try:
            # Create fresh components for each attempt
            generator = create_parameter_generator("optuna", random_state=42 + attempt)
            evaluator = BacktestEvaluator(backtester, n_jobs=1)
            orchestrator = OptimizationOrchestrator(generator, evaluator)
            
            logger.info(f"Starting optimization attempt {attempt + 1}/{max_retries}")
            
            result = orchestrator.optimize(optimization_data, n_trials)
            logger.info(f"Optimization completed successfully on attempt {attempt + 1}")
            return result
                
        except OptimizationError as e:
            logger.error(f"Optimization failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
            continue
            
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
            continue
    
    raise OptimizationError("All optimization attempts failed")

# Use robust optimization
try:
    robust_result = robust_optimization(optimization_data, n_trials=50)
    print(f"Robust optimization succeeded: {robust_result.best_objective:.4f}")
except OptimizationError as e:
    print(f"Optimization failed: {e}")
```

These examples demonstrate comprehensive usage patterns for the refactored architecture, covering basic usage, advanced customization, error handling, and real-world scenarios.