# Advanced Crossover Operators

## Overview

The portfolio backtester's genetic optimizer now supports four advanced crossover operators that provide more sophisticated recombination strategies than the basic crossover types available in PyGAD. These operators can significantly improve convergence speed and solution quality for different types of optimization problems.

## Available Operators

### 1. Simulated Binary Crossover (SBX)

**Purpose**: Designed for continuous optimization problems, SBX creates offspring that are close to the parents, with the spread controlled by a distribution index.

**Best For**: Continuous parameter optimization where you want to maintain similarity to parents while exploring the search space.

**Parameters**:
- `sbx_distribution_index`: Controls how close offspring are to parents (higher values = closer offspring)
  - **Default**: 20.0
  - **Range**: 1.0 - 100.0
  - **Effect**: Higher values create offspring closer to parents; lower values allow for more exploration

**Configuration**:
```yaml
genetic_algorithm_params:
  advanced_crossover_type: simulated_binary
  sbx_distribution_index: 15.0  # Optional, defaults to 20.0
```

### 2. Multi-point Crossover

**Purpose**: Creates offspring by selecting multiple crossover points and alternating between parents for chromosome segments.

**Best For**: Problems where genes are linked or where you want to preserve building blocks from parents.

**Parameters**:
- `num_crossover_points`: Number of crossover points to use
  - **Default**: 3
  - **Range**: 2 - 10
  - **Effect**: More points create more diverse offspring but may break beneficial gene combinations

**Configuration**:
```yaml
genetic_algorithm_params:
  advanced_crossover_type: multi_point
  num_crossover_points: 4  # Optional, defaults to 3
```

### 3. Uniform Crossover Variant

**Purpose**: Unlike standard uniform crossover, this variant allows for a bias parameter to favor one parent over another.

**Best For**: When you want to bias the recombination toward one parent while still maintaining some exploration.

**Parameters**:
- `uniform_crossover_bias`: Probability of selecting from the first parent
  - **Default**: 0.5
  - **Range**: 0.1 - 0.9
  - **Effect**: Values > 0.5 favor the first parent; values < 0.5 favor the second parent

**Configuration**:
```yaml
genetic_algorithm_params:
  advanced_crossover_type: uniform_variant
  uniform_crossover_bias: 0.3  # Optional, defaults to 0.5
```

### 4. Arithmetic Crossover

**Purpose**: Creates offspring as weighted averages of parents, suitable for continuous parameter optimization.

**Best For**: Continuous optimization problems where weighted combinations of parent parameters are meaningful.

**Parameters**: None required

**Configuration**:
```yaml
genetic_algorithm_params:
  advanced_crossover_type: arithmetic
```

## Configuration

### Basic Usage

To use any advanced crossover operator, simply specify the `advanced_crossover_type` parameter:

```yaml
genetic_algorithm_params:
  advanced_crossover_type: simulated_binary  # or multi_point, uniform_variant, arithmetic
```

### Complete Example

Here's a complete genetic algorithm configuration that uses SBX with custom parameters:

```yaml
genetic_algorithm_params:
  num_generations: 100
  sol_per_pop: 30
  num_parents_mating: 15
  parent_selection_type: tournament
  mutation_type: random
  mutation_percent_genes: 10
  
  # Advanced Crossover Operator
  advanced_crossover_type: simulated_binary
  sbx_distribution_index: 12.0
  
  # Optional Adaptive Features
  adaptive_mutation:
    enabled: true
    base_rate: 0.1
    min_rate: 0.01
    max_rate: 0.5
  elite_preservation:
    enabled: true
    max_archive_size: 50
    injection_frequency: 5
```

### Switching Between Operators

You can easily switch between different operators by changing the `advanced_crossover_type`:

```yaml
# For exploration phase
genetic_algorithm_params:
  advanced_crossover_type: multi_point
  num_crossover_points: 5

# For exploitation phase
genetic_algorithm_params:
  advanced_crossover_type: arithmetic

# For biased recombination
genetic_algorithm_params:
  advanced_crossover_type: uniform_variant
  uniform_crossover_bias: 0.7
```

## Performance Considerations

### Computational Overhead

Advanced crossover operators have minimal computational overhead compared to standard PyGAD operators:
- **Arithmetic Crossover**: Fastest, simple weighted average
- **Uniform Crossover Variant**: Very fast, similar to standard uniform
- **Multi-point Crossover**: Moderate, depends on number of points
- **Simulated Binary Crossover**: Slightly slower due to distribution calculations

### When to Use Each Operator

| Operator | Best For | Performance | Exploration Level |
|----------|----------|-------------|-------------------|
| SBX | Continuous optimization | Medium | Medium-High |
| Multi-point | Linked genes, building blocks | Medium | High |
| Uniform Variant | Biased recombination | Fast | Medium |
| Arithmetic | Weighted combinations | Fastest | Low-Medium |

## Backward Compatibility

Advanced crossover operators are fully backward compatible:
- When `advanced_crossover_type` is not specified, standard PyGAD crossover types are used
- All existing scenarios continue to work unchanged
- You can mix advanced operators with other GA enhancements (adaptive parameters, elite preservation)

## Example Scenarios

### Momentum Strategy Optimization

```yaml
name: momentum_sbx_optimization
strategy: momentum
rebalance_frequency: ME
position_sizer: equal_weight
transaction_costs_bps: 10
train_window_months: 36
test_window_months: 12
optimization_metric: Sharpe
genetic_algorithm_params:
  num_generations: 50
  sol_per_pop: 20
  num_parents_mating: 10
  advanced_crossover_type: simulated_binary
  sbx_distribution_index: 18.0
optimize:
- parameter: lookback_months
  type: int
  min_value: 3
  max_value: 12
- parameter: num_holdings
  type: int
  min_value: 10
  max_value: 50
strategy_params:
  long_only: true
  top_decile_fraction: 0.2
  smoothing_lambda: 0.5
  leverage: 1.0
```

### EMA Crossover Strategy

```yaml
name: ema_crossover_multipoint
strategy: ema_crossover
rebalance_frequency: ME
position_sizer: volatility_weighted
transaction_costs_bps: 5
train_window_months: 24
test_window_months: 24
optimization_metric: Calmar
genetic_algorithm_params:
  num_generations: 75
  sol_per_pop: 25
  num_parents_mating: 12
  advanced_crossover_type: multi_point
  num_crossover_points: 3
optimize:
- parameter: fast_period
  type: int
  min_value: 5
  max_value: 30
- parameter: slow_period
  type: int
  min_value: 20
  max_value: 100
strategy_params:
  long_only: true
  atr_multiplier: 2.0
  max_positions: 20
```

## Integration with Other Features

Advanced crossover operators work seamlessly with other GA enhancements:

### With Adaptive Parameter Control
```yaml
genetic_algorithm_params:
  advanced_crossover_type: arithmetic
  adaptive_mutation:
    enabled: true
    base_rate: 0.1
    diversity_threshold: 0.3
```

### With Elite Preservation
```yaml
genetic_algorithm_params:
  advanced_crossover_type: uniform_variant
  uniform_crossover_bias: 0.4
  elite_preservation:
    enabled: true
    max_archive_size: 30
    injection_frequency: 3
```

### Complete Enhanced Configuration
```yaml
genetic_algorithm_params:
  num_generations: 100
  sol_per_pop: 40
  num_parents_mating: 20
  advanced_crossover_type: simulated_binary
  sbx_distribution_index: 15.0
  
  # Adaptive Parameter Control
  adaptive_mutation:
    enabled: true
    base_rate: 0.12
    min_rate: 0.02
    max_rate: 0.40
    diversity_threshold: 0.25
  
  # Elite Preservation
  elite_preservation:
    enabled: true
    max_archive_size: 50
    injection_strategy: tournament
    injection_frequency: 5
```

## Troubleshooting

### Common Issues

1. **Parameters out of bounds**: Ensure parameter values stay within gene space limits
2. **Poor convergence**: Try different distribution index values for SBX
3. **Excessive diversity loss**: Reduce the number of crossover points for Multi-point

### Best Practices

1. **Start with defaults**: Use default parameter values and adjust based on performance
2. **Monitor diversity**: Use `--log-level DEBUG` to see population diversity metrics
3. **Experiment with operators**: Try different operators to see which works best for your problem
4. **Combine with other enhancements**: Advanced crossover works well with adaptive parameters and elite preservation

## Further Reading

- [Genetic Algorithm Configuration Guide](timing_configuration_guide.md)
- [Walk-Forward Optimization Robustness](../tests/TEST_WFO_ROBUSTNESS.md)
- [PyGAD Documentation](https://pygad.readthedocs.io/en/latest/)