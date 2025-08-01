### Monte Carlo Configuration

```yaml
monte_carlo_config:
  enable_synthetic_data: true
  enable_during_optimization: true    # Stage 1: Lightweight MC during optimization
  enable_stage2_stress_testing: true  # Stage 2: Full stress testing after optimization
  replacement_percentage: 0.05        # 5% of assets replaced with synthetic data
  min_historical_observations: 200    # Minimum data for parameter estimation
  
  garch_config:
    model_type: "GARCH"
    p: 1
    q: 1
    distribution: "studentt"
    bounds:
      omega: [1e-6, 1.0]
      alpha: [0.01, 0.3]
      beta: [0.5, 0.99]
      nu: [2.1, 30.0]
  
  generation_config:
    buffer_multiplier: 1.2
    max_attempts: 2
    validation_tolerance: 0.3
  
  validation_config:
    enable_validation: false  # Disabled during optimization for speed
    tolerance: 0.8
```
