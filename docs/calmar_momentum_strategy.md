# Calmar Momentum Strategy

## Overview

The Calmar Momentum Strategy is a risk-adjusted momentum strategy that ranks assets based on their Calmar ratio instead of simple price momentum. The Calmar ratio is defined as the annualized return divided by the maximum drawdown, providing a measure of risk-adjusted performance that specifically penalizes large drawdowns.

## Key Features

- **Risk-Adjusted Ranking**: Uses Calmar ratio to identify assets with the best risk-adjusted momentum
- **Drawdown Awareness**: Specifically penalizes assets with large historical drawdowns
- **Configurable Parameters**: Supports various configuration options for customization
- **SMA Filter**: Optional simple moving average filter for market timing

## Calmar Ratio Calculation

The Calmar ratio is calculated as:

```
Calmar Ratio = Annualized Return / Maximum Drawdown
```

Where:
- **Annualized Return**: Rolling average return multiplied by 12 (for monthly data)
- **Maximum Drawdown**: The largest peak-to-trough decline in the rolling window

## Strategy Logic

1. **Calculate Rolling Calmar Ratios**: For each asset, calculate the Calmar ratio over a rolling window
2. **Rank Assets**: Rank all assets by their Calmar ratio (higher is better)
3. **Select Winners**: Choose the top N assets based on the ranking
4. **Weight Allocation**: Allocate equal weights among selected assets
5. **Apply Smoothing**: Smooth transitions between periods using exponential smoothing
6. **Apply Leverage**: Scale positions according to target leverage
7. **Apply SMA Filter**: (Optional) Set weights to zero during risk-off periods

## Configuration Parameters

### Core Parameters

- `rolling_window` (int, default: 6): Number of months for rolling Calmar calculation
- `top_decile_fraction` (float, default: 0.1): Fraction of universe to hold (if num_holdings not specified)
- `num_holdings` (int, optional): Explicit number of holdings (overrides top_decile_fraction)

### Risk Management

- `leverage` (float, default: 1.0): Target leverage (1.0 = 100% invested)
- `long_only` (bool, default: True): Whether to allow short positions
- `smoothing_lambda` (float, default: 0.5): Smoothing parameter (0 = no smoothing, 1 = no changes)

### Market Timing

- `sma_filter_window` (int, optional): SMA window for market timing filter
- `target_return` (float, default: 0.0): Target return for excess return calculation

## Advantages

1. **Drawdown Sensitivity**: Explicitly penalizes assets with large historical drawdowns
2. **Risk-Adjusted**: Considers both return and risk in asset selection
3. **Momentum Capture**: Still captures momentum effects but with risk adjustment
4. **Flexible**: Highly configurable for different market conditions

## Disadvantages

1. **Lookback Dependency**: Requires sufficient history for meaningful Calmar calculation
2. **Drawdown Lag**: May be slow to react to recent drawdowns not yet reflected in rolling window
3. **Complexity**: More complex than simple momentum strategies
4. **Data Requirements**: Requires clean price data for accurate drawdown calculation

## Comparison with Other Strategies

| Strategy | Ranking Metric | Risk Adjustment | Drawdown Focus |
|----------|----------------|-----------------|----------------|
| Momentum | Price Return | None | No |
| Sharpe Momentum | Sharpe Ratio | Volatility | No |
| Sortino Momentum | Sortino Ratio | Downside Deviation | Partial |
| **Calmar Momentum** | **Calmar Ratio** | **Maximum Drawdown** | **Yes** |

## Usage Example

```python
from portfolio_backtester.strategies.calmar_momentum_strategy import CalmarMomentumStrategy

# Configure strategy
config = {
    'rolling_window': 6,           # 6-month rolling window
    'top_decile_fraction': 0.1,    # Top 10% of assets
    'smoothing_lambda': 0.5,       # 50% smoothing
    'leverage': 1.0,               # 100% leverage
    'long_only': True,             # Long-only
    'sma_filter_window': 10        # 10-month SMA filter
}

# Create strategy
strategy = CalmarMomentumStrategy(config)

# Generate signals
weights = strategy.generate_signals(price_data, benchmark_data)
```

## Backtesting

To run the Calmar Momentum strategy in the backtester:

```bash
python src/portfolio_backtester/backtester.py --portfolios "Calmar_Momentum"
```

The strategy is configured in `config.py` with walk-forward optimization on the `rolling_window` parameter.

## Performance Considerations

- **Best For**: Markets with varying volatility regimes where drawdown control is important
- **Avoid When**: Very short time series or highly trending markets without significant drawdowns
- **Optimization**: The `rolling_window` parameter is typically the most important to optimize

## Implementation Notes

- Uses pandas rolling window operations for efficiency
- Handles edge cases like zero drawdowns and insufficient data
- Maintains consistency with other strategy implementations in the framework
- Supports both explicit number of holdings and fractional universe selection