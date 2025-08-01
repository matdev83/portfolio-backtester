### WFO Robustness Configuration (`config/parameters.yaml`)

```yaml
wfo_robustness_config:
  enable_window_randomization: true
  enable_start_date_randomization: true
  train_window_randomization:
    min_offset: 3    # Minimum months to add to base train window
    max_offset: 14   # Maximum months to add to base train window
  test_window_randomization:
    min_offset: 3    # Minimum months to add to base test window
    max_offset: 14   # Maximum months to add to base test window
  start_date_randomization:
    min_offset: 0    # Minimum months to offset start date
    max_offset: 12   # Maximum months to offset start date
  stability_metrics:
    enable: true
    worst_percentile: 10
    consistency_threshold: 0.0
  random_seed: null  # Set for reproducible randomization
```
