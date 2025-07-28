# Test Suite Maintenance Guide

## Overview

This guide documents the refactored test suite structure, patterns, and maintenance procedures for the Portfolio Backtester project.

## Test Suite Organization

### Directory Structure

```
tests/
├── unit/                          # Fast, isolated unit tests
│   ├── strategies/               # Strategy unit tests
│   ├── timing/                   # Timing framework unit tests
│   ├── data_sources/            # Data source unit tests
│   ├── core/                    # Core functionality unit tests
│   ├── monte_carlo/             # Monte Carlo unit tests
│   ├── optimization/            # Optimization unit tests
│   ├── portfolio/               # Portfolio unit tests
│   ├── reporting/               # Reporting unit tests
│   ├── data_integrity/          # Data integrity unit tests
│   └── universe/                # Universe unit tests
├── integration/                  # Integration tests
│   ├── backtester/              # Backtester integration tests
│   └── strategy_timing/         # Strategy-timing integration tests
├── system/                      # System/end-to-end tests
├── fixtures/                    # Shared test data fixtures
│   ├── market_data.py           # Market data generation
│   ├── strategy_data.py         # Strategy-specific test data
│   ├── timing_data.py           # Timing framework test data
│   ├── optimized_data_generator.py  # Optimized data generation
│   └── performance_fixtures.py  # Performance-optimized fixtures
└── base/                        # Base test classes
    ├── strategy_test_base.py    # Base strategy testing patterns
    ├── timing_test_base.py      # Base timing testing patterns
    └── integration_test_base.py # Base integration testing patterns
```

## Test Categories and Markers

### Pytest Markers

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Integration tests (moderate speed)
- `@pytest.mark.system` - System tests (slow, end-to-end)
- `@pytest.mark.fast` - Fast-running tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.network` - Tests requiring network access
- `@pytest.mark.strategy` - Strategy-related tests
- `@pytest.mark.timing` - Timing framework tests
- `@pytest.mark.data_sources` - Data source tests
- `@pytest.mark.optimization` - Optimization tests
- `@pytest.mark.monte_carlo` - Monte Carlo tests
- `@pytest.mark.reporting` - Reporting tests

### Running Specific Test Categories

```bash
# Run only fast unit tests
pytest -m "unit and fast"

# Run integration tests
pytest -m "integration"

# Run all tests except slow ones
pytest -m "not slow"

# Run strategy-related tests
pytest -m "strategy"
```

## Base Test Classes

### BaseStrategyTest

Use for all strategy tests. Provides:
- Common setup/teardown patterns
- Standard data generation methods
- Assertion helpers for strategy testing

```python
from tests.base.strategy_test_base import BaseStrategyTest

class TestMyStrategy(BaseStrategyTest):
    def test_signal_generation(self):
        # Use inherited methods and fixtures
        data = self.generate_test_data()
        strategy = self.create_strategy()
        signals = strategy.generate_signals(data)
        self.assert_valid_signals(signals)
```

### BaseMomentumStrategyTest

Specialized base class for momentum strategies:
- Momentum-specific test patterns
- Common momentum calculations
- Momentum signal validation

### BaseTimingTest

Use for timing framework tests:
- Common timing setup patterns
- Standard assertions for timing behavior
- Migration test patterns

### BaseIntegrationTest

Use for integration tests:
- Integration test utilities
- Common validation methods
- Cross-component testing patterns

## Performance-Optimized Fixtures

### Using Cached Fixtures

```python
# Use session-scoped cached fixtures for expensive data generation
def test_strategy_performance(large_ohlcv_data):
    # large_ohlcv_data is cached and reused across tests
    strategy = MomentumStrategy(config)
    results = strategy.generate_signals(large_ohlcv_data)
    assert len(results) > 0

# Use smaller fixtures for fast unit tests
def test_strategy_initialization(fast_test_data):
    # fast_test_data is minimal for speed
    strategy = MomentumStrategy(config)
    assert strategy is not None
```

### Available Performance Fixtures

- `small_ohlcv_data` - Small dataset for fast unit tests
- `medium_ohlcv_data` - Medium dataset for integration tests
- `large_ohlcv_data` - Large dataset for performance tests (cached)
- `benchmark_data` - Cached benchmark data
- `momentum_strategy_config` - Cached momentum strategy configuration
- `fast_test_data` - Very small dataset for fastest tests

## Test Data Generation Best Practices

### 1. Use Cached Fixtures for Expensive Data

```python
# ❌ Bad - generates data in every test
def test_strategy(self):
    data = generate_large_dataset()  # Expensive!
    # test logic

# ✅ Good - uses cached fixture
def test_strategy(self, large_ohlcv_data):
    # large_ohlcv_data is cached and reused
    # test logic
```

### 2. Use Class-Level Setup for Shared Data

```python
# ✅ Good - data generated once per test class
class TestStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = generate_test_data()
    
    def test_method_1(self):
        # Use self.test_data
        pass
```

### 3. Use Appropriate Data Sizes

- Unit tests: 5-50 data points
- Integration tests: 100-1000 data points
- Performance tests: 1000+ data points (cached)

## Writing New Tests

### 1. Strategy Tests

```python
from tests.base.strategy_test_base import BaseStrategyTest

class TestNewStrategy(BaseStrategyTest):
    def setUp(self):
        self.strategy_config = {
            'param1': value1,
            'param2': value2
        }
        self.strategy = NewStrategy(self.strategy_config)
    
    def test_initialization(self):
        # Test strategy initialization
        self.assertIsInstance(self.strategy, NewStrategy)
    
    def test_signal_generation(self):
        # Use base class methods
        data = self.generate_test_data()
        signals = self.strategy.generate_signals(data)
        self.assert_valid_signals(signals)
```

### 2. Timing Tests

```python
from tests.base.timing_test_base import BaseTimingTest

class TestNewTimingFeature(BaseTimingTest):
    def test_timing_behavior(self):
        # Use base class timing utilities
        controller = self.create_timing_controller()
        result = controller.should_generate_signal(date, strategy)
        self.assert_valid_timing_result(result)
```

### 3. Integration Tests

```python
from tests.base.integration_test_base import BaseIntegrationTest

@pytest.mark.integration
class TestComponentIntegration(BaseIntegrationTest):
    def test_component_interaction(self):
        # Test interaction between components
        result = self.run_integration_scenario()
        self.assert_integration_success(result)
```

## Test Performance Guidelines

### Performance Targets

- Unit tests: < 0.1 seconds each
- Integration tests: < 1 second each
- System tests: < 10 seconds each
- Total test suite: < 5 minutes

### Performance Monitoring

Use the `performance_monitor` fixture to track slow tests:

```python
def test_potentially_slow_operation(performance_monitor):
    # Test logic here
    # Will warn if test takes > 1 second
    pass
```

### Optimizing Slow Tests

1. **Use cached fixtures** instead of generating data
2. **Reduce data size** for unit tests
3. **Mock expensive operations** in unit tests
4. **Use class-level setup** for shared expensive operations
5. **Consider moving to integration tests** if testing component interaction

## Migration from Old Patterns

### Old Pattern → New Pattern

```python
# ❌ Old: Duplicate test setup
class TestStrategy1(unittest.TestCase):
    def setUp(self):
        # Duplicate setup code
        pass

# ✅ New: Use base class
class TestStrategy1(BaseStrategyTest):
    # Inherits common setup

# ❌ Old: Expensive data generation in setUp
def setUp(self):
    self.data = generate_expensive_data()

# ✅ New: Use cached fixtures
def test_method(self, cached_data):
    # Use cached_data

# ❌ Old: Large monolithic test files
# test_everything.py (1000+ lines)

# ✅ New: Focused modules
# test_specific_feature.py (< 200 lines)
```

## Debugging Test Failures

### Common Issues and Solutions

1. **Import Errors**
   - Check that modules are in the correct directory structure
   - Verify `__init__.py` files exist in test directories

2. **Fixture Not Found**
   - Ensure fixture is imported in `conftest.py`
   - Check fixture scope (session vs function)

3. **Slow Tests**
   - Use performance fixtures instead of generating data
   - Check if test should be marked as `@pytest.mark.slow`

4. **Flaky Tests**
   - Set random seeds for reproducible results
   - Use mocks for external dependencies

## Continuous Integration

### Test Execution Strategy

```bash
# Fast feedback loop
pytest -m "unit and fast" --maxfail=5

# Full test suite
pytest -m "not slow" --cov=src

# Performance tests (run less frequently)
pytest -m "slow" --tb=short
```

### Coverage Requirements

- Minimum 80% overall coverage
- New code must have 90%+ coverage
- Critical paths (strategies, backtester core) require 95%+ coverage

## Adding New Test Categories

When adding new functionality:

1. **Create appropriate directory structure**
2. **Add pytest markers** in `conftest.py`
3. **Create base test class** if needed
4. **Add performance fixtures** for expensive operations
5. **Update this documentation**

## Best Practices Summary

✅ **DO:**
- Use base test classes for common patterns
- Use cached fixtures for expensive data generation
- Write focused, single-responsibility test modules
- Use appropriate pytest markers
- Keep unit tests fast (< 0.1s)
- Mock external dependencies in unit tests

❌ **DON'T:**
- Generate expensive test data in `setUp` methods
- Create monolithic test files (> 500 lines)
- Duplicate test setup code across classes
- Mix unit and integration test concerns
- Skip performance considerations
- Forget to update documentation when adding new patterns