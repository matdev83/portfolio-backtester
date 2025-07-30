# Migration Guide: Architecture Refactoring

## Overview

This guide provides step-by-step instructions for migrating from the legacy Portfolio Backtester architecture to the new refactored system. The migration is designed to be gradual and backward-compatible, allowing you to transition at your own pace while maintaining existing functionality.

## Migration Timeline

The refactored architecture is now available alongside the legacy system. You can:

1. **Immediate**: Start using new components for new projects
2. **Gradual**: Migrate existing code piece by piece using feature flags
3. **Complete**: Remove legacy code paths after full migration

## What Changed

### High-Level Changes

| Component | Legacy | New | Status |
|-----------|--------|-----|--------|
| Backtesting | `Backtester` class (mixed concerns) | `StrategyBacktester` (pure backtesting) | ✅ Available |
| Optimization | Embedded in `Backtester` | `OptimizationOrchestrator` + generators | ✅ Available |
| Parameter Generation | Optuna/PyGAD specific code | `ParameterGenerator` interface | ✅ Available |
| Results | Mixed return types | Structured data classes | ✅ Available |
| Factory Pattern | Direct instantiation | `create_parameter_generator()` | ✅ Available |

### Breaking Changes

⚠️ **Important**: The new architecture maintains backward compatibility through adapters. No immediate breaking changes are required.

**Future Breaking Changes** (will be announced):
- Legacy `Backtester.optimize()` method will be deprecated
- Direct Optuna/PyGAD imports will be discouraged
- Some internal APIs will be removed

## Step-by-Step Migration

### Step 1: Update Imports (Optional)

**Before (Legacy)**:
```python
from portfolio_backtester.backtester import Backtester
```

**After (New)**:
```python
from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
from portfolio_backtester.optimization.orchestrator import OptimizationOrchestrator
from portfolio_backtester.optimization.factory import create_parameter_generator
```

**Transition Strategy**: You can continue using the legacy imports. The new imports are available for new code.

### Step 2: Migrate Simple Backtesting

**Before (Legacy)**:
```python
# Legacy approach
backtester = Backtester(global_config, data_source)
result = backtester.run_backtest(scenario_config)

# Access results
total_return = result['metrics']['total_return']
sharpe_ratio = result['metrics']['sharpe_ratio']
```

**After (New)**:
```python
# New approach
strategy_backtester = StrategyBacktester(global_config, data_source)
result = strategy_backtester.backtest_strategy(
    scenario_config,
    monthly_data,
    daily_data,
    rets_full
)

# Access structured results
total_return = result.metrics['total_return']
sharpe_ratio = result.metrics['sharpe_ratio']
trade_history = result.trade_history
```

**Migration Steps**:
1. Replace `Backtester` with `StrategyBacktester`
2. Update method call from `run_backtest()` to `backtest_strategy()`
3. Provide data parameters explicitly (monthly_data, daily_data, rets_full)
4. Update result access to use structured attributes

### Step 3: Migrate Optimization Code

**Before (Legacy)**:
```python
# Legacy optimization
backtester = Backtester(global_config, data_source)
result = backtester.optimize(
    scenario_config,
    optimizer_type='optuna',
    n_trials=100,
    random_state=42
)

best_params = result['best_parameters']
best_objective = result['best_objective']
```

**After (New)**:
```python
# New optimization approach
from portfolio_backtester.optimization.orchestrator import OptimizationOrchestrator
from portfolio_backtester.optimization.factory import create_parameter_generator
from portfolio_backtester.optimization.evaluator import BacktestEvaluator
from portfolio_backtester.optimization.parameter_generator import OptimizationData

# Create components
strategy_backtester = StrategyBacktester(global_config, data_source)
parameter_generator = create_parameter_generator('optuna', random_state=42)
evaluator = BacktestEvaluator(strategy_backtester, n_jobs=4)
orchestrator = OptimizationOrchestrator(parameter_generator, evaluator)

# Prepare optimization data
optimization_data = OptimizationData(
    scenario_config=scenario_config,
    optimization_spec=optimization_spec,
    monthly_data=monthly_data,
    daily_data=daily_data,
    rets_full=rets_full,
    walk_forward_config=walk_forward_config
)

# Run optimization
result = orchestrator.optimize(optimization_data, n_trials=100)

# Access structured results
best_params = result.best_parameters
best_objective = result.best_objective
optimization_history = result.optimization_history
```

**Migration Steps**:
1. Replace single `backtester.optimize()` call with component-based approach
2. Create separate components: generator, evaluator, orchestrator
3. Use `OptimizationData` container for data management
4. Update result access to use structured attributes

### Step 4: Update Configuration Files

**Before (Legacy)**:
```yaml
# scenario_config.yaml
strategy: momentum_strategy
strategy_params:
  lookback_period: 12
  momentum_threshold: 0.05

# Optimization embedded in strategy config
optimization:
  lookback_period:
    type: int
    low: 6
    high: 24
```

**After (New)**:
```yaml
# scenario_config.yaml - Strategy configuration
strategy: momentum_strategy
strategy_params:
  lookback_period: 12
  momentum_threshold: 0.05

# optimization_spec.yaml - Separate optimization specification
optimization_spec:
  lookback_period:
    type: int
    low: 6
    high: 24
  momentum_threshold:
    type: float
    low: 0.01
    high: 0.2
    log: true

# walk_forward_config.yaml - WFO configuration
walk_forward_config:
  train_window_months: 36
  test_window_months: 6
  step_size_months: 3
```

**Migration Steps**:
1. Separate optimization specifications from strategy configurations
2. Create dedicated configuration files for different concerns
3. Update loading code to handle separate configurations

### Step 5: Feature Flag Migration

The system supports feature flags for gradual migration:

```python
# Enable new architecture gradually
from portfolio_backtester.feature_flags import FeatureFlags

# Enable new backtesting engine
FeatureFlags.enable('new_backtesting_engine')

# Enable new optimization system
FeatureFlags.enable('new_optimization_system')

# Use legacy system for specific components
FeatureFlags.disable('new_parameter_generation')
```

**Available Feature Flags**:
- `new_backtesting_engine`: Use `StrategyBacktester` instead of legacy backtester
- `new_optimization_system`: Use `OptimizationOrchestrator` system
- `new_parameter_generation`: Use factory pattern for parameter generators
- `structured_results`: Return structured result objects

## Common Migration Patterns

### Pattern 1: Wrapper Functions for Backward Compatibility

```python
# Create wrapper functions to ease migration
def legacy_optimize(global_config, scenario_config, optimizer_type='optuna', **kwargs):
    """
    Wrapper function that provides legacy interface using new architecture.
    
    This allows gradual migration without changing all calling code at once.
    """
    # Extract parameters
    n_trials = kwargs.get('n_trials', 100)
    random_state = kwargs.get('random_state', None)
    n_jobs = kwargs.get('n_jobs', 1)
    
    # Create new architecture components
    strategy_backtester = StrategyBacktester(global_config, data_source)
    parameter_generator = create_parameter_generator(optimizer_type, random_state=random_state)
    evaluator = BacktestEvaluator(strategy_backtester, n_jobs=n_jobs)
    orchestrator = OptimizationOrchestrator(parameter_generator, evaluator)
    
    # Load data (you'll need to implement data loading)
    monthly_data, daily_data, rets_full = load_data_for_scenario(scenario_config)
    
    # Create optimization data
    optimization_data = OptimizationData(
        scenario_config=scenario_config,
        optimization_spec=extract_optimization_spec(scenario_config),
        monthly_data=monthly_data,
        daily_data=daily_data,
        rets_full=rets_full,
        walk_forward_config=extract_wfo_config(scenario_config)
    )
    
    # Run optimization
    result = orchestrator.optimize(optimization_data, n_trials=n_trials)
    
    # Convert to legacy format
    return {
        'best_parameters': result.best_parameters,
        'best_objective': result.best_objective,
        'optimization_history': [
            {
                'parameters': trial.parameters,
                'objective': trial.objective_value,
                'metrics': trial.metrics
            }
            for trial in result.optimization_history
        ]
    }

# Usage - looks like legacy code but uses new architecture
result = legacy_optimize(global_config, scenario_config, optimizer_type='optuna', n_trials=50)
```

### Pattern 2: Gradual Component Migration

```python
# Migrate one component at a time
class HybridBacktester:
    """
    Hybrid backtester that allows mixing legacy and new components.
    
    Use this during migration to test new components while keeping
    legacy components for stability.
    """
    
    def __init__(self, global_config, data_source, use_new_backtesting=False):
        self.global_config = global_config
        self.data_source = data_source
        self.use_new_backtesting = use_new_backtesting
        
        if use_new_backtesting:
            self.strategy_backtester = StrategyBacktester(global_config, data_source)
        else:
            self.legacy_backtester = LegacyBacktester(global_config, data_source)
    
    def run_backtest(self, scenario_config):
        """Run backtest using selected engine."""
        if self.use_new_backtesting:
            # Use new engine
            monthly_data, daily_data, rets_full = self._load_data(scenario_config)
            return self.strategy_backtester.backtest_strategy(
                scenario_config, monthly_data, daily_data, rets_full
            )
        else:
            # Use legacy engine
            return self.legacy_backtester.run_backtest(scenario_config)
    
    def optimize(self, scenario_config, **kwargs):
        """Run optimization with hybrid approach."""
        # Always use new optimization system, but allow legacy backtesting
        parameter_generator = create_parameter_generator(
            kwargs.get('optimizer_type', 'optuna'),
            random_state=kwargs.get('random_state')
        )
        
        if self.use_new_backtesting:
            evaluator = BacktestEvaluator(self.strategy_backtester)
        else:
            evaluator = LegacyBacktestEvaluator(self.legacy_backtester)
        
        orchestrator = OptimizationOrchestrator(parameter_generator, evaluator)
        
        # Continue with optimization...
```

### Pattern 3: Configuration Migration

```python
# Utility functions for configuration migration
def migrate_scenario_config(legacy_config):
    """
    Convert legacy scenario configuration to new format.
    
    Args:
        legacy_config: Legacy configuration dictionary
        
    Returns:
        Tuple of (scenario_config, optimization_spec, walk_forward_config)
    """
    # Extract strategy configuration
    scenario_config = {
        'strategy': legacy_config['strategy'],
        'strategy_params': legacy_config.get('strategy_params', {}),
        'rebalance_frequency': legacy_config.get('rebalance_frequency', 'monthly')
    }
    
    # Extract optimization specification
    optimization_spec = legacy_config.get('optimization', {})
    
    # Extract walk-forward configuration
    walk_forward_config = {
        'train_window_months': legacy_config.get('train_window_months', 36),
        'test_window_months': legacy_config.get('test_window_months', 6),
        'step_size_months': legacy_config.get('step_size_months', 3)
    }
    
    return scenario_config, optimization_spec, walk_forward_config

# Usage
legacy_config = load_legacy_config('momentum_strategy.yaml')
scenario_config, optimization_spec, walk_forward_config = migrate_scenario_config(legacy_config)
```

## Testing Your Migration

### 1. Validation Tests

Create tests to ensure new architecture produces identical results:

```python
# test_migration_validation.py
import pytest
from portfolio_backtester.testing.migration_validator import MigrationValidator

class TestMigrationValidation:
    """Test that new architecture produces identical results to legacy."""
    
    def test_backtest_results_identical(self):
        """Test that backtesting results are identical."""
        validator = MigrationValidator()
        
        # Run same backtest with both architectures
        legacy_result = validator.run_legacy_backtest(scenario_config)
        new_result = validator.run_new_backtest(scenario_config)
        
        # Compare results
        assert validator.compare_backtest_results(legacy_result, new_result)
    
    def test_optimization_results_equivalent(self):
        """Test that optimization results are equivalent."""
        validator = MigrationValidator()
        
        # Run same optimization with both architectures
        legacy_result = validator.run_legacy_optimization(scenario_config, random_state=42)
        new_result = validator.run_new_optimization(scenario_config, random_state=42)
        
        # Compare results (allowing for small numerical differences)
        assert validator.compare_optimization_results(
            legacy_result, new_result, tolerance=1e-6
        )
```

### 2. Performance Tests

```python
# test_migration_performance.py
import time
import pytest

class TestMigrationPerformance:
    """Test that new architecture maintains performance."""
    
    def test_backtest_performance(self):
        """Test that backtesting performance is maintained."""
        # Time legacy approach
        start_time = time.time()
        legacy_result = run_legacy_backtest(scenario_config)
        legacy_time = time.time() - start_time
        
        # Time new approach
        start_time = time.time()
        new_result = run_new_backtest(scenario_config)
        new_time = time.time() - start_time
        
        # New approach should be within 10% of legacy performance
        assert new_time <= legacy_time * 1.1
    
    def test_optimization_performance(self):
        """Test that optimization performance is maintained."""
        # Similar performance test for optimization
        pass
```

## Troubleshooting Common Issues

### Issue 1: Import Errors

**Problem**: `ImportError: cannot import name 'StrategyBacktester'`

**Solution**:
```python
# Check if you're using the correct import path
from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester

# If still failing, check if refactored components are available
try:
    from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
    NEW_ARCHITECTURE_AVAILABLE = True
except ImportError:
    NEW_ARCHITECTURE_AVAILABLE = False
    # Fall back to legacy approach
```

### Issue 2: Configuration Format Errors

**Problem**: `KeyError: 'optimization_spec'` when using new architecture

**Solution**:
```python
# Ensure optimization_spec is properly formatted
optimization_spec = {
    "parameter_name": {
        "type": "int" | "float" | "categorical",
        "low": value,  # for int/float
        "high": value,  # for int/float
        "choices": [list],  # for categorical
        "log": bool  # optional, for log-scale sampling
    }
}

# Validate optimization spec before use
def validate_optimization_spec(spec):
    for param_name, param_spec in spec.items():
        assert 'type' in param_spec, f"Missing 'type' for parameter {param_name}"
        
        if param_spec['type'] in ['int', 'float']:
            assert 'low' in param_spec and 'high' in param_spec
        elif param_spec['type'] == 'categorical':
            assert 'choices' in param_spec
```

### Issue 3: Data Format Mismatches

**Problem**: `ValueError: Expected DataFrame with MultiIndex columns`

**Solution**:
```python
# Ensure data is in the correct format for new architecture
def prepare_data_for_new_architecture(data):
    """
    Convert data to format expected by new architecture.
    
    Args:
        data: Raw data from data source
        
    Returns:
        Properly formatted data for StrategyBacktester
    """
    if not isinstance(data.columns, pd.MultiIndex):
        # Convert to MultiIndex if needed
        data = data.stack().unstack(level=0)
    
    # Ensure required columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in data.columns.get_level_values(1):
            raise ValueError(f"Missing required column: {col}")
    
    return data
```

### Issue 4: Performance Degradation

**Problem**: New architecture is significantly slower

**Solution**:
```python
# Check parallel processing configuration
evaluator = BacktestEvaluator(
    strategy_backtester,
    n_jobs=4  # Increase for better performance
)

# Enable caching if available
global_config['cache_enabled'] = True

# Use appropriate data types
# Ensure DataFrames use efficient dtypes (float32 instead of float64 if precision allows)
```

## Migration Checklist

### Pre-Migration
- [ ] Read this migration guide completely
- [ ] Backup existing code and configurations
- [ ] Set up test environment for validation
- [ ] Identify components to migrate first

### During Migration
- [ ] Start with simple backtesting migration
- [ ] Validate results match legacy system
- [ ] Migrate optimization components
- [ ] Update configuration files
- [ ] Test performance impact
- [ ] Update documentation and comments

### Post-Migration
- [ ] Run full test suite
- [ ] Performance benchmark comparison
- [ ] Update team documentation
- [ ] Plan legacy code removal timeline
- [ ] Monitor production performance

## Getting Help

### Resources
- **API Documentation**: `docs/api/README.md`
- **Architecture Documentation**: `docs/architecture/README.md`
- **Extensibility Guide**: `docs/extensibility_guide.md`
- **Example Code**: `docs/api/examples.md`

### Support Channels
- Create GitHub issues for bugs or questions
- Check existing issues for similar problems
- Review test files for usage examples

### Best Practices
1. **Migrate incrementally** - Don't try to migrate everything at once
2. **Test thoroughly** - Validate that results match legacy system
3. **Monitor performance** - Ensure new architecture meets performance requirements
4. **Document changes** - Keep track of what you've migrated
5. **Use feature flags** - Enable gradual rollout and easy rollback

This migration guide provides a comprehensive path from the legacy architecture to the new refactored system while maintaining stability and backward compatibility.