# Timing Configuration Guide

This guide explains how to configure the flexible timing system in the portfolio backtester.

## Overview

The timing system allows strategies to use different timing modes:

- **Time-Based**: Traditional scheduled rebalancing (monthly, quarterly, etc.)
- **Signal-Based**: Market-driven timing based on strategy signals
- **Custom**: User-defined timing controllers for advanced use cases

## Configuration Structure

All timing configuration is specified in the `timing_config` section of your strategy configuration:

```yaml
strategy_name:
  strategy_params:
    # Strategy-specific parameters
  timing_config:
    mode: time_based  # or signal_based, custom
    # Mode-specific parameters
```

## Time-Based Timing

Time-based timing rebalances on fixed schedules.

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | `time_based` | Must be "time_based" |
| `rebalance_frequency` | string | `M` | Rebalancing frequency |
| `rebalance_offset` | integer | `0` | Days offset from standard date |
| `enable_logging` | boolean | `false` | Enable detailed logging |
| `log_level` | string | `INFO` | Logging level |

### Frequency Options

| Code | Description | Example |
|------|-------------|---------|
| `D` | Daily | Every trading day |
| `W` | Weekly | Every Monday |
| `M` | Monthly | First trading day of month |
| `ME` | Month End | Last trading day of month |
| `Q` | Quarterly | First trading day of quarter |
| `QE` | Quarter End | Last trading day of quarter |
| `A` | Annual | First trading day of year |
| `Y` | Annual | Alias for `A` |
| `YE` | Year End | Last trading day of year |

### Example Configurations

```yaml
# Monthly rebalancing
monthly_strategy:
  timing_config:
    mode: time_based
    rebalance_frequency: M
    rebalance_offset: 0

# Quarterly rebalancing with 5-day offset
quarterly_offset_strategy:
  timing_config:
    mode: time_based
    rebalance_frequency: Q
    rebalance_offset: 5

# Daily rebalancing
daily_strategy:
  timing_config:
    mode: time_based
    rebalance_frequency: D
```

## Signal-Based Timing

Signal-based timing generates signals based on market conditions.

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | `signal_based` | Must be "signal_based" |
| `scan_frequency` | string | `D` | How often to scan for signals |
| `min_holding_period` | integer | `1` | Minimum days to hold position |
| `max_holding_period` | integer/null | `null` | Maximum days to hold position |
| `enable_logging` | boolean | `false` | Enable detailed logging |
| `log_level` | string | `INFO` | Logging level |

### Scan Frequency Options

| Code | Description |
|------|-------------|
| `D` | Daily scanning |
| `W` | Weekly scanning |
| `M` | Monthly scanning |

### Example Configurations

```yaml
# Daily signal scanning with 1-day holding
short_term_signals:
  timing_config:
    mode: signal_based
    scan_frequency: D
    min_holding_period: 1
    max_holding_period: 5

# Weekly signals with unlimited holding
long_term_signals:
  timing_config:
    mode: signal_based
    scan_frequency: W
    min_holding_period: 7
    max_holding_period: null

# UVXY-style strategy (1-day forced exit)
uvxy_style:
  timing_config:
    mode: signal_based
    scan_frequency: D
    min_holding_period: 1
    max_holding_period: 1
```

## Custom Timing Controllers

Custom timing controllers allow you to implement complex timing logic.

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | `custom` | Must be "custom" |
| `custom_controller_class` | string | Required | Class name or registered alias |
| `custom_controller_params` | object | `{}` | Parameters for controller |
| `enable_logging` | boolean | `false` | Enable detailed logging |
| `log_level` | string | `INFO` | Logging level |

### Built-in Custom Controllers

#### Adaptive Timing Controller

Adjusts frequency based on market volatility.

```yaml
adaptive_strategy:
  timing_config:
    mode: custom
    custom_controller_class: adaptive_timing
    custom_controller_params:
      volatility_threshold: 0.02
      base_frequency: M
      high_vol_frequency: W
      low_vol_frequency: Q
```

#### Momentum Timing Controller

Rebalances based on momentum signals.

```yaml
momentum_strategy:
  timing_config:
    mode: custom
    custom_controller_class: momentum_timing
    custom_controller_params:
      momentum_period: 20
```

### Creating Custom Controllers

To create your own timing controller:

1. **Inherit from TimingController**:

```python
from src.portfolio_backtester.timing.timing_controller import TimingController
from src.portfolio_backtester.timing.custom_timing_registry import register_timing_controller

@register_timing_controller('my_custom_timing')
class MyCustomTimingController(TimingController):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        # Initialize your custom parameters
    
    def should_generate_signal(self, current_date, strategy):
        # Implement your timing logic
        return True  # or False
    
    def get_rebalance_dates(self, start_date, end_date, available_dates, strategy):
        # Return list of rebalance dates
        return []
```

2. **Register and Use**:

```yaml
my_strategy:
  timing_config:
    mode: custom
    custom_controller_class: my_custom_timing
    custom_controller_params:
      param1: value1
```

## Logging Configuration

Enable detailed logging to debug timing decisions:

```yaml
debug_strategy:
  timing_config:
    mode: signal_based
    scan_frequency: D
    enable_logging: true
    log_level: DEBUG  # DEBUG, INFO, WARNING, ERROR
```

### Log Output Example

```
2023-01-15 10:30:00 - timing.global - INFO - [MyStrategy] Signal generation: YES - RSI below threshold
2023-01-15 10:30:00 - timing.global - INFO - [MyStrategy] Position entry: AAPL weight=0.2500 @ $150.00
2023-01-16 10:30:00 - timing.global - INFO - [MyStrategy] Signal generation: NO - holding period constraint
```

## Migration from Legacy Configuration

Legacy configurations are automatically migrated:

```yaml
# Legacy format (still supported)
legacy_strategy:
  strategy_params:
    rebalance_frequency: M  # Will be migrated to timing_config

# Equivalent new format
new_strategy:
  strategy_params:
    # strategy parameters only
  timing_config:
    mode: time_based
    rebalance_frequency: M
```

## Validation and Error Handling

The system provides comprehensive validation with helpful error messages:

### Common Validation Errors

1. **Invalid Mode**:
```
✗ mode: Invalid timing mode 'invalid_mode'
  Suggestion: Use one of: time_based, signal_based, custom
```

2. **Invalid Frequency**:
```
✗ rebalance_frequency: Invalid rebalance frequency 'X'
  Suggestion: Use one of: D, W, M, ME, Q, QE, A, Y, YE
```

3. **Invalid Holding Period**:
```
✗ max_holding_period: max_holding_period (2) cannot be less than min_holding_period (5)
  Suggestion: Set max_holding_period to at least 5 or null
```

### Validation Tools

Use the validation tools to check your configuration:

```python
from src.portfolio_backtester.timing.config_schema import validate_timing_config

config = {
    'timing_config': {
        'mode': 'signal_based',
        'scan_frequency': 'D'
    }
}

errors = validate_timing_config(config, raise_on_error=False)
if errors:
    print("Configuration errors found:")
    for error in errors:
        print(f"- {error.message}")
```

## Performance Considerations

### High-Performance Configuration

For maximum performance:

```yaml
performance_strategy:
  timing_config:
    mode: time_based
    rebalance_frequency: M
    enable_logging: false  # Disable logging
    # Minimal configuration
```

### Debug Configuration

For maximum debugging information:

```yaml
debug_strategy:
  timing_config:
    mode: signal_based
    scan_frequency: D
    enable_logging: true
    log_level: DEBUG  # Maximum verbosity
```

## Best Practices

1. **Start Simple**: Begin with time-based timing before moving to signal-based
2. **Use Validation**: Always validate your configuration before running backtests
3. **Enable Logging**: Use logging during development and debugging
4. **Test Thoroughly**: Test custom timing controllers with known scenarios
5. **Document Custom Controllers**: Provide clear documentation for custom timing logic

## Troubleshooting

### Common Issues

1. **No Signals Generated**:
   - Check `scan_frequency` setting
   - Verify strategy signal generation logic
   - Enable logging to see timing decisions

2. **Unexpected Rebalancing**:
   - Check `rebalance_frequency` and `rebalance_offset`
   - Verify date alignment with trading calendar

3. **Custom Controller Not Found**:
   - Ensure controller is registered or fully qualified class name is correct
   - Check import paths and module availability

4. **Performance Issues**:
   - Disable logging in production
   - Use appropriate scan frequencies
   - Consider caching for expensive operations

### Getting Help

1. Enable debug logging to see detailed timing decisions
2. Use validation tools to check configuration
3. Check the examples in `config/timing_examples.yaml`
4. Review the test cases for usage patterns

## Examples

See `config/timing_examples.yaml` for comprehensive examples of all timing configurations.