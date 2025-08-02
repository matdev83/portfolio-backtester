# Capital Allocation Modes in Portfolio Backtester

## Overview

The portfolio backtester now supports **two capital allocation modes** that control how position sizes are calculated throughout the backtest:

### 1. Reinvestment Mode (Default)
- **Industry names**: "reinvestment", "compound"
- **Behavior**: Position sizes based on current account balance
- **Effect**: Enables compounding - profits increase future position sizes, losses decrease them
- **Use case**: Realistic simulation of account growth/decline

### 2. Fixed Fractional Mode
- **Industry names**: "fixed_fractional", "fixed_capital"  
- **Behavior**: Position sizes always based on initial capital
- **Effect**: Disables compounding - position sizes stay constant relative to starting capital
- **Use case**: Strategy comparison without compounding effects

## Key Changes Made

### 1. Enhanced TradeTracker Class

**File**: `src/portfolio_backtester/trading/trade_tracker.py`

#### New Properties:
- `initial_portfolio_value`: Starting capital (e.g., $1M)
- `current_portfolio_value`: Dynamic capital that changes with each trade
- `allocation_mode`: Controls position sizing behavior ("reinvestment" or "fixed_fractional")
- `daily_portfolio_value`: Timeline of portfolio value changes
- `daily_cash_balance`: Timeline of cash balance

#### New Methods:
- `get_current_portfolio_value()`: Returns current available capital
- `get_total_return()`: Returns total return percentage
- `get_capital_timeline()`: Returns DataFrame with capital tracking over time

#### Enhanced Position Tracking:
- Position sizes calculated based on allocation mode:
  - **Reinvestment**: Uses `current_portfolio_value` (enables compounding)
  - **Fixed fractional**: Uses `initial_portfolio_value` (disables compounding)
- Trade P&L (including commissions) automatically updates `current_portfolio_value`
- Capital tracking metrics added to trade statistics

### 2. Dynamic Capital in Portfolio Logic

**File**: `src/portfolio_backtester/backtester_logic/portfolio_logic.py`

#### New Function:
- `_track_trades_with_dynamic_capital()`: Enhanced trade tracking that uses current portfolio value for commission calculations and position sizing

#### Key Improvements:
- Transaction costs calculated based on allocation mode
- Position updates use dynamic capital for proper compounding effect
- Meta strategy support with allocation mode integration

### 3. Meta Strategy Support

**Files**: 
- `src/portfolio_backtester/strategies/base/meta_strategy.py`
- `src/portfolio_backtester/strategies/base/trade_aggregator.py`

#### Enhanced Meta Strategy Features:
- **TradeAggregator**: Updated to accept and respect allocation mode
- **Capital Allocation**: Sub-strategy capital calculated based on allocation mode
- **Signal Aggregation**: Respects allocation mode for consistent behavior
- **Configuration**: `allocation_mode` added to tunable parameters

### 3. Configuration Update

**File**: `config/parameters.yaml`

#### Added:
```yaml
GLOBAL_CONFIG:
  # Portfolio capital configuration
  portfolio_value: 1000000.0  # Starting capital: $1M for compounding
```

## How Allocation Modes Work

### Regular Strategies

#### Reinvestment Mode (Compounding Enabled):
```
Initial Capital: $1,000,000
Trade 1: 10% allocation = $100,000 position
Trade 1 Profit: $10,000 (capital becomes $1,010,000)
Trade 2: 10% allocation = $101,000 position (larger due to compounding!)
```

#### Fixed Fractional Mode (Compounding Disabled):
```
Initial Capital: $1,000,000
Trade 1: 10% allocation = $100,000 position
Trade 1 Profit: $10,000 (capital becomes $1,010,000)
Trade 2: 10% allocation = $100,000 position (same as initial, no compounding)
```

### Meta Strategies

#### Reinvestment Mode (Sub-Strategy Compounding):
```
Initial Capital: $1,000,000
Sub-Strategy A (60%): $600,000 allocated
Sub-Strategy B (40%): $400,000 allocated
After 10% Portfolio Profit: $1,100,000 total
Sub-Strategy A (60%): $660,000 allocated (compounding!)
Sub-Strategy B (40%): $440,000 allocated (compounding!)
```

#### Fixed Fractional Mode (No Sub-Strategy Compounding):
```
Initial Capital: $1,000,000
Sub-Strategy A (60%): $600,000 allocated
Sub-Strategy B (40%): $400,000 allocated
After 10% Portfolio Profit: $1,100,000 total
Sub-Strategy A (60%): $600,000 allocated (no compounding)
Sub-Strategy B (40%): $400,000 allocated (no compounding)
```

### Strategy Configuration

Add to your strategy configuration:

```yaml
# Regular Strategy
strategy_config:
  name: "MyStrategy"
  allocation_mode: "reinvestment"  # or "fixed_fractional"
  # ... other strategy parameters

# Meta Strategy
meta_strategy_config:
  name: "MyMetaStrategy"
  strategy: "SimpleMetaStrategy"
  allocation_mode: "reinvestment"  # Controls sub-strategy capital allocation
  strategy_params:
    allocation_mode: "reinvestment"  # Also pass to strategy params
    allocations:
      - strategy_id: "momentum"
        strategy_class: "MomentumStrategy"
        weight: 0.6
      - strategy_id: "mean_reversion"
        strategy_class: "MeanReversionStrategy"
        weight: 0.4
```

**Available modes:**
- `"reinvestment"` or `"compound"` - Enable compounding (default)
- `"fixed_fractional"` or `"fixed_capital"` - Disable compounding

## Example Usage

```python
from src.portfolio_backtester.trading.trade_tracker import TradeTracker

# Initialize with reinvestment mode (compounding enabled)
tracker_compound = TradeTracker(
    initial_portfolio_value=1000000.0, 
    allocation_mode="reinvestment"
)

# Initialize with fixed fractional mode (compounding disabled)
tracker_fixed = TradeTracker(
    initial_portfolio_value=1000000.0, 
    allocation_mode="fixed_fractional"
)

# After profitable trades, get current capital
current_capital = tracker_compound.get_current_portfolio_value()  # e.g., $1,050,000

# Get total return
total_return = tracker_compound.get_total_return()  # e.g., 0.05 (5%)

# Get capital timeline
timeline = tracker_compound.get_capital_timeline()
print(timeline.head())

# Check allocation mode
print(f"Allocation mode: {tracker_compound.allocation_mode}")
```

## Trade Statistics Enhancement

The trade statistics now include capital tracking metrics:

- `allocation_mode`: The allocation mode used ("reinvestment" or "fixed_fractional")
- `initial_capital`: Starting portfolio value
- `final_capital`: Ending portfolio value  
- `total_return_pct`: Total return percentage
- `capital_growth_factor`: Final capital / Initial capital

## Testing

Run the allocation mode tests to verify functionality:

### Regular Strategy Test:
```bash
python test_compounding.py
```

This test demonstrates:
1. **Reinvestment mode**: Position sizes change with account balance (compounding)
2. **Fixed fractional mode**: Position sizes stay constant relative to initial capital
3. **Both modes**: Proper P&L and commission tracking
4. **Side-by-side comparison**: Clear difference in allocation behavior

### Meta Strategy Tests:
```bash
python test_meta_strategy_allocation_modes.py
python test_meta_strategy_cash_balance.py
python test_meta_strategy_available_capital.py
```

These tests demonstrate:
1. **TradeAggregator**: Supports both allocation modes with proper cash balance tracking
2. **Meta strategy integration**: Proper capital allocation to sub-strategies
3. **Available capital updates**: Meta strategy capital updates correctly after each trade P&L
4. **Cash balance maintenance**: Accurate tracking of cash through all trade operations
5. **Configuration**: Allocation mode properly passed through system
6. **Compounding verification**: Clear difference in sub-strategy capital allocation based on mode

## Benefits of Allocation Mode Implementation

1. **Flexible Analysis**: Choose between compounding and non-compounding analysis
2. **Realistic Performance**: Reinvestment mode reflects actual account behavior
3. **Strategy Comparison**: Fixed fractional mode enables fair strategy comparison
4. **Risk Management**: Reinvestment mode adjusts position sizes based on performance
5. **Industry Standard**: Uses common financial industry terminology
6. **User Control**: Strategy-level setting allows per-strategy customization

## Migration Notes

- **Default behavior**: Existing backtests will use "reinvestment" mode (compounding enabled)
- **No breaking changes**: Existing strategy implementations work without modification
- **New configuration**: Add `allocation_mode` to strategy configs to explicitly control behavior
- **Historical results**: May differ slightly due to more accurate capital tracking
- **Optimization**: `allocation_mode` can be included in parameter optimization

## Performance Impact

The compounding implementation has minimal performance impact:
- Capital tracking adds negligible computational overhead
- Memory usage slightly increased for capital timeline storage
- All existing optimizations (Numba, caching) remain functional

## Verification

To verify allocation modes are working in your backtests:

1. **Check allocation mode**: Look for `allocation_mode` in trade statistics
2. **Compare capital metrics**: `final_capital` vs `initial_capital` shows P&L tracking
3. **Monitor position sizes**: 
   - **Reinvestment mode**: Should vary with performance
   - **Fixed fractional mode**: Should stay constant relative to initial capital
4. **Review capital timeline**: Use `get_capital_timeline()` for detailed analysis

## When to Use Each Mode

### Use Reinvestment Mode When:
- Simulating realistic account growth/decline
- Evaluating long-term compounding potential
- Modeling actual trading account behavior
- Assessing risk-adjusted returns with compounding
- **Meta strategies**: Want sub-strategies to benefit from overall portfolio performance

### Use Fixed Fractional Mode When:
- Comparing strategies on equal footing
- Isolating strategy performance from compounding effects
- Academic research requiring constant position sizing
- Analyzing strategy signals without capital bias
- **Meta strategies**: Want consistent sub-strategy capital allocation regardless of performance

The backtester now provides flexible, industry-standard capital allocation modes for comprehensive strategy analysis.

## ✅ **Complete Implementation Summary**

### **1. Regular Strategies**
- ✅ **TradeTracker**: Supports both `"reinvestment"` and `"fixed_fractional"` modes
- ✅ **Position Sizing**: Based on current capital (reinvestment) or initial capital (fixed fractional)
- ✅ **Commission Calculation**: Uses appropriate capital base for each mode
- ✅ **Configuration**: Strategy-level `allocation_mode` setting

### **2. Meta Strategies** 
- ✅ **TradeAggregator**: Updated to accept and respect allocation mode with proper cash balance tracking
- ✅ **Available Capital Updates**: Meta strategy capital updates correctly after each sub-strategy trade P&L
- ✅ **Cash Balance Maintenance**: Accurate tracking of cash balance through all trade operations
- ✅ **Capital Allocation**: Sub-strategy capital calculated based on allocation mode
  - **Reinvestment**: Uses current available capital (enables compounding)
  - **Fixed fractional**: Uses initial capital (disables compounding)
- ✅ **Signal Aggregation**: Respects allocation mode for consistent behavior
- ✅ **Configuration**: `allocation_mode` added to tunable parameters

### **3. System Integration**
- ✅ **Portfolio Logic**: Both regular and meta strategies use consistent allocation modes
- ✅ **Commission Calculations**: Respect allocation mode for all strategy types
- ✅ **Configuration**: Seamless integration through scenario config
- ✅ **Parameter Optimization**: `allocation_mode` available for optimization

### **4. Cash Balance & Capital Tracking**
- ✅ **Real-time Updates**: Available capital updates immediately after each trade P&L
- ✅ **Accurate Calculations**: Portfolio value = cash balance + position values
- ✅ **Compounding Verification**: Clear difference in capital allocation based on mode
- ✅ **Multi-strategy Support**: Complex portfolios with multiple sub-strategies handled correctly

### **5. Industry Standards & Testing**
- ✅ **Terminology**: Uses standard financial industry names
- ✅ **Default Behavior**: Reinvestment mode (compounding) is default
- ✅ **Backward Compatibility**: Existing strategies work without changes
- ✅ **Comprehensive Testing**: Multiple test suites verify all functionality

The implementation provides users with complete control over capital allocation behavior, supporting both realistic compounding scenarios and academic research requirements, with consistent behavior across all strategy types including complex meta strategies with proper cash balance maintenance.