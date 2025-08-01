# Rebalance Frequency Configuration Guide

This guide documents the comprehensive rebalance frequency support in the Portfolio Backtester framework.

## Overview

The framework now supports all major pandas frequency strings for portfolio rebalancing, providing flexibility for various trading strategies from high-frequency to long-term investment approaches.

## Supported Frequencies

### Daily and Weekly Frequencies
- **`D`** - Daily (business days)
- **`B`** - Business days (excludes weekends)
- **`W`** - Weekly (ending on Sunday)
- **`W-MON`** - Weekly ending on Monday
- **`W-TUE`** - Weekly ending on Tuesday
- **`W-WED`** - Weekly ending on Wednesday
- **`W-THU`** - Weekly ending on Thursday
- **`W-FRI`** - Weekly ending on Friday
- **`W-SAT`** - Weekly ending on Saturday
- **`W-SUN`** - Weekly ending on Sunday

### Monthly Frequencies
- **`M`** - Month end (legacy, same as ME)
- **`ME`** - Month end (recommended)
- **`MS`** - Month start
- **`BM`** - Business month end
- **`BMS`** - Business month start

### Quarterly Frequencies
- **`Q`** - Quarter end (legacy, same as QE)
- **`QE`** - Quarter end (recommended)
- **`QS`** - Quarter start
- **`BQ`** - Business quarter end
- **`BQS`** - Business quarter start
- **`2Q`** - Every 2 quarters (semi-annual)

### Semi-Annual Frequencies
- **`6M`** - Every 6 months
- **`6ME`** - Every 6 months ending on month end
- **`6MS`** - Every 6 months starting on month start

### Annual Frequencies
- **`A`** - Year end (legacy, same as YE)
- **`AS`** - Year start
- **`Y`** - Year end (legacy, same as YE)
- **`YE`** - Year end (recommended)
- **`YS`** - Year start
- **`BA`** - Business year end
- **`BAS`** - Business year start
- **`BY`** - Business year end
- **`BYS`** - Business year start
- **`2A`** - Every 2 years

### High-Frequency Support
For algorithmic and high-frequency strategies:
- **`H`** - Hourly
- **`2H`** - Every 2 hours
- **`3H`** - Every 3 hours
- **`4H`** - Every 4 hours
- **`6H`** - Every 6 hours
- **`8H`** - Every 8 hours
- **`12H`** - Every 12 hours

## Usage Examples

### In Strategy Configuration
```yaml
strategy_params:
  rebalance_frequency: "ME"  # Month end rebalancing
```

### In Optimization Configuration
```yaml
optimize:
  - parameter: rebalance_frequency
    type: categorical
    values: ["ME", "QE", "6M", "YE"]  # Test different frequencies
```

### Common Use Cases

#### Conservative Long-Term Strategy
```yaml
strategy_params:
  rebalance_frequency: "YE"  # Annual rebalancing
```

#### Tactical Asset Allocation
```yaml
strategy_params:
  rebalance_frequency: "QE"  # Quarterly rebalancing
```

#### Active Portfolio Management
```yaml
strategy_params:
  rebalance_frequency: "ME"  # Monthly rebalancing
```

#### High-Frequency Trading
```yaml
strategy_params:
  rebalance_frequency: "H"   # Hourly rebalancing
```

## Migration from Legacy Frequencies

The framework maintains backward compatibility with legacy frequency codes:

| Legacy | Modern Equivalent | Description |
|--------|------------------|-------------|
| `M`    | `ME`            | Month end   |
| `Q`    | `QE`            | Quarter end |
| `A`    | `YE`            | Year end    |
| `Y`    | `YE`            | Year end    |

## Best Practices

### 1. Choose Appropriate Frequency
- **Daily/Hourly**: High-frequency strategies, algorithmic trading
- **Weekly**: Short-term tactical strategies
- **Monthly**: Most active strategies, momentum-based approaches
- **Quarterly**: Tactical asset allocation, factor-based strategies
- **Semi-Annual**: Moderate rebalancing, cost-conscious strategies
- **Annual**: Long-term buy-and-hold, tax-efficient strategies

### 2. Consider Transaction Costs
Higher frequency rebalancing increases transaction costs. Balance strategy performance with cost efficiency.

### 3. Use Modern Frequency Codes
Prefer explicit codes like `ME`, `QE`, `YE` over legacy codes for clarity.

### 4. Optimization Testing
When optimizing rebalance frequency, test a range of frequencies:
```yaml
optimize:
  - parameter: rebalance_frequency
    type: categorical
    values: ["ME", "QE", "6M", "YE"]
```

## Technical Implementation

The frequency validation is implemented in:
- `src/portfolio_backtester/timing/config_validator.py`
- `src/portfolio_backtester/timing/backward_compatibility.py`

All frequencies are validated against pandas' frequency standards to ensure compatibility with the underlying data processing pipeline.

## Error Handling

Invalid frequencies will produce clear error messages:
```
Invalid rebalance_frequency 'INVALID'. Must be one of [...]. 
Common values: 'M' (monthly), 'ME' (month-end), 'Q' (quarterly), 
'QE' (quarter-end), '6M' (semi-annual), 'A' (annual), 'YE' (year-end), 
'D' (daily), 'W' (weekly)
```

## Performance Considerations

- **Higher frequencies** = More rebalancing = Higher transaction costs
- **Lower frequencies** = Less responsive to market changes
- **Business day frequencies** (BM, BQ, etc.) automatically skip weekends/holidays
- **End-of-period frequencies** (ME, QE, YE) align with financial reporting cycles