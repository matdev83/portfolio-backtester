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
- `daily_portfolio_value`: Timeline of portfolio value changes
- `daily_cash_balance`: Timeline of cash balance

#### New Methods:
- `get_current_portfolio_value()`: Returns current available capital
- `get_total_return()`: Returns total return percentage
- `get_capital_timeline()`: Returns DataFrame with capital tracking over time

#### Enhanced Position Tracking:
- Position sizes now calculated using `current_portfolio_value` instead of fixed value
- Trade P&L (including commissions) automatically updates `current_portfolio_value`
- Capital tracking metrics added to trade statistics

### 2. Dynamic Capital in Portfolio Logic

**File**: `src/portfolio_backtester/backtester_logic/portfolio_logic.py`

#### New Function:
- `_track_trades_with_dynamic_capital()`: Enhanced trade tracking that uses current portfolio value for commission calculations and position sizing

#### Key Improvements:
- Transaction costs calculated based on current portfolio value
- Position updates use dynamic capital for proper compounding effect

### 3. Configuration Update

**File**: `config/parameters.yaml`

#### Added:
```yaml
GLOBAL_CONFIG:
  # Portfolio capital configuration
  portfolio_value: 1000000.0  # Starting capital: $1M for compounding
```

## How Compounding Works

### Before (Fixed Capital):
```
Initial Capital: $1,000,000
Trade 1: 10% allocation = $100,000 position
Trade 1 Profit: $10,000
Trade 2: 10% allocation = $100,000 position (same as before!)
```

### After (Dynamic Capital with Compounding):
```
Initial Capital: $1,000,000
Trade 1: 10% allocation = $100,000 position
Trade 1 Profit: $10,000 (capital becomes $1,010,000)
Trade 2: 10% allocation = $101,000 position (larger due to compounding!)
```

## Example Usage

```python
from src.portfolio_backtester.trading.trade_tracker import TradeTracker

# Initialize with $1M
tracker = TradeTracker(initial_portfolio_value=1000000.0)

# After profitable trades, get current capital
current_capital = tracker.get_current_portfolio_value()  # e.g., $1,050,000

# Get total return
total_return = tracker.get_total_return()  # e.g., 0.05 (5%)

# Get capital timeline
timeline = tracker.get_capital_timeline()
print(timeline.head())
```

## Trade Statistics Enhancement

The trade statistics now include capital tracking metrics:

- `initial_capital`: Starting portfolio value
- `final_capital`: Ending portfolio value  
- `total_return_pct`: Total return percentage
- `capital_growth_factor`: Final capital / Initial capital

## Testing

Run the compounding test to verify functionality:

```bash
python test_compounding.py
```

This test demonstrates:
1. Profitable trades increase available capital
2. Losing trades decrease available capital
3. Position sizes adjust based on current capital
4. Proper P&L and commission tracking

## Benefits of Compounding Implementation

1. **Realistic Performance**: Results now reflect actual compounding returns
2. **Accurate Position Sizing**: Each trade uses the correct amount of available capital
3. **Proper Risk Management**: Position sizes automatically adjust based on account performance
4. **Better Strategy Comparison**: Strategies can be compared on their ability to compound returns
5. **Real-world Accuracy**: Matches how actual trading accounts work

## Migration Notes

- Existing backtests will automatically use the new compounding logic
- No changes needed to strategy implementations
- Historical results may differ slightly due to more accurate capital tracking
- The `portfolio_value` configuration now represents initial capital, not fixed capital

## Performance Impact

The compounding implementation has minimal performance impact:
- Capital tracking adds negligible computational overhead
- Memory usage slightly increased for capital timeline storage
- All existing optimizations (Numba, caching) remain functional

## Verification

To verify compounding is working in your backtests:

1. Check that `final_capital` ≠ `initial_capital` in trade statistics
2. Monitor `capital_growth_factor` for compound growth
3. Review `get_capital_timeline()` for capital evolution over time
4. Compare position sizes over time - they should vary with performance

The backtester now provides a much more realistic and accurate representation of how trading strategies perform with compounding returns.