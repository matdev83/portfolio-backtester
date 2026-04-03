# Design Document

## Overview

This design addresses six critical roadblocks in the portfolio backtester test suite that are causing cascading failures across multiple test modules. The fixes are prioritized by impact - resolving these issues will restore functionality to timing systems, momentum strategies, trade aggregation, position sizing, optimization testing, and universe configuration.

## Architecture

The fixes follow a targeted approach, addressing root causes in core components that have wide-reaching effects:

1. **Signal-Based Timing System**: Fix variable initialization order
2. **Momentum Strategy Logic**: Resolve pandas Index boolean evaluation
3. **Trade Aggregator Validation**: Correct quantity/direction handling
4. **Position Sizer Interface**: Standardize method signatures
5. **Optimization Test Isolation**: Implement unique study naming
6. **Universe Configuration Testing**: Add proper test fixtures and mocking

## Components and Interfaces

### 1. Signal-Based Timing Fix

**File**: `src/portfolio_backtester/timing/signal_based_timing.py`

**Issue**: `UnboundLocalError: local variable 'base_scan_dates' referenced before assignment`

**Root Cause**: The `base_scan_dates` variable is conditionally assigned but unconditionally used.

**Solution**: Ensure `base_scan_dates` is always initialized before the loop that references it.

```python
# Current problematic pattern:
if condition:
    base_scan_dates = some_value
# base_scan_dates used here without guarantee of initialization

# Fixed pattern:
base_scan_dates = default_value
if condition:
    base_scan_dates = some_value
# base_scan_dates guaranteed to be initialized
```

### 2. Momentum Strategy Pandas Fix

**File**: `src/portfolio_backtester/strategies/portfolio/momentum_strategy.py`

**Issue**: `ValueError: The truth value of a Index is ambiguous`

**Root Cause**: Direct boolean evaluation of pandas Index objects in conditional statements.

**Solution**: Replace ambiguous boolean checks with explicit pandas methods.

```python
# Current problematic pattern:
if current_date in roro_output.index:  # This can cause ambiguity

# Fixed pattern:
if not roro_output.index.empty and current_date in roro_output.index:
# Or use .any(), .all(), .empty as appropriate
```

### 3. Trade Aggregator Validation Fix

**File**: `src/portfolio_backtester/strategies/base/trade_aggregator.py`

**Issue**: `ValueError: quantity must be non-negative; use side flag to indicate direction`

**Root Cause**: The validation logic incorrectly rejects negative quantities for sell trades, but the trade creation logic may legitimately create negative quantities for sells.

**Solution**: Either:
- Option A: Modify trade creation to always use positive quantities with side flags
- Option B: Modify validation to accept negative quantities when they align with trade direction

**Recommended**: Option B - Allow negative quantities for sell trades while maintaining validation for inconsistent quantity/side combinations.

### 4. Position Sizer Interface Fix

**File**: `src/portfolio_backtester/portfolio/position_sizer.py`

**Issue**: `TypeError: EqualWeightSizer.calculate_weights() missing 1 required positional argument: 'prices'`

**Root Cause**: Inconsistent method signatures across position sizer implementations.

**Solution**: Standardize all position sizer `calculate_weights` methods to accept `(signals, prices, **kwargs)`.

### 5. Optuna Study Isolation Fix

**Files**: `tests/unit/optimization/test_parallel_optimization_enhanced.py`

**Issue**: `optuna.exceptions.DuplicatedStudyError: Another study with name 'X' already exists`

**Root Cause**: Tests use hardcoded study names that conflict when run in sequence.

**Solution**: Generate unique study names using timestamps, UUIDs, or test-specific prefixes.

```python
# Current problematic pattern:
study_name = "test_study"

# Fixed pattern:
import uuid
study_name = f"test_study_{uuid.uuid4().hex[:8]}"
# Or use test method name + timestamp
```

### 6. Universe Configuration Test Fixtures

**Files**: `tests/unit/universe/test_universe_loader.py`, `tests/unit/strategies/test_universe_configuration.py`

**Issue**: `UniverseLoaderError: Universe file not found`

**Root Cause**: Tests depend on external universe files that don't exist in the test environment.

**Solution**: Create proper test fixtures that either:
- Create temporary universe files for testing
- Mock the file system interactions
- Use existing universe files and adapt test expectations

## Data Models

### Trade Validation Model

```python
@dataclass
class TradeValidationResult:
    is_valid: bool
    error_message: Optional[str] = None
    normalized_quantity: Optional[float] = None
    normalized_side: Optional[str] = None
```

### Study Name Generator

```python
class StudyNameGenerator:
    @staticmethod
    def generate_unique_name(base_name: str) -> str:
        timestamp = int(time.time() * 1000)  # milliseconds
        return f"{base_name}_{timestamp}"
```

## Error Handling

### Signal-Based Timing
- Add defensive initialization of variables
- Provide meaningful error messages for invalid configurations
- Handle edge cases like empty date ranges gracefully

### Momentum Strategy
- Wrap pandas Index operations in try-catch blocks
- Provide fallback behavior for ambiguous conditions
- Log warnings when falling back to default behavior

### Trade Aggregator
- Implement comprehensive trade validation with clear error messages
- Allow configuration of validation strictness
- Provide trade normalization utilities

### Position Sizer
- Ensure all implementations follow the same interface contract
- Add parameter validation with helpful error messages
- Provide default implementations for optional parameters

### Optimization Tests
- Implement automatic cleanup of test studies
- Add retry logic for transient database conflicts
- Use test-specific database configurations when possible

### Universe Configuration
- Implement robust test fixtures with cleanup
- Add mocking for file system operations
- Provide fallback behavior for missing universe files

## Testing Strategy

### Unit Tests
- Each fix will have dedicated unit tests to prevent regression
- Tests will cover both the happy path and edge cases
- Mock external dependencies to ensure test isolation

### Integration Tests
- Verify that fixes don't break existing functionality
- Test interactions between fixed components
- Ensure performance characteristics are maintained

### Test Isolation
- Each test will clean up after itself
- Use unique identifiers to prevent test interference
- Mock external resources (files, databases) where appropriate

### Regression Prevention
- Add specific tests for each fixed bug
- Use property-based testing where applicable
- Implement continuous integration checks for the fixed issues