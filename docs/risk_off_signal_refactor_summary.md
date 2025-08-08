# Risk-off Signal Generator System Implementation

## Overview

This document summarizes the implementation of the new `RiskOffSignalGenerator` system, which replaces the deprecated `RoRo` (Risk-on/Risk-off) signal system with a more robust, SOLID-principles-based architecture.

## Architecture Changes

### From RoRo to RiskOffSignalGenerator

The refactor involved:

1. **Renamed and Clarified Semantics**:
   - `BaseRoRoSignal` → `IRiskOffSignalGenerator` 
   - `DummyRoRoSignal` → `DummyRiskOffSignalGenerator`
   - Signal semantics clarified: `True` = Risk-off conditions, `False` = Risk-on conditions

2. **Applied SOLID Principles**:
   - **Single Responsibility**: Each generator focuses only on risk regime detection
   - **Open/Closed**: Open for extension (new generators), closed for modification
   - **Liskov Substitution**: All implementations are fully substitutable
   - **Interface Segregation**: Focused interface with minimal dependencies
   - **Dependency Inversion**: Strategies depend on abstractions, not concretions

3. **Provider Pattern Integration**:
   - Follows framework's provider pattern used by other systems
   - Configuration-based instantiation instead of class attributes
   - Factory pattern for provider creation

## New Architecture Components

### 1. Interface Layer (`IRiskOffSignalGenerator`)

```python
# Abstract interface for risk-off signal generation
class IRiskOffSignalGenerator(ABC):
    @abstractmethod
    def generate_risk_off_signal(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> bool:
        """True = Risk-off conditions, False = Risk-on conditions"""
        pass
```

### 2. Implementation Layer

- **`NoRiskOffSignalGenerator`**: Default implementation that never signals risk-off (Null Object Pattern)
- **`DummyRiskOffSignalGenerator`**: Configurable test implementation with hardcoded windows (Test Double Pattern)

### 3. Provider Layer

- **`IRiskOffSignalProvider`**: Abstract provider interface
- **`ConfigBasedRiskOffSignalProvider`**: Configuration-driven provider
- **`FixedRiskOffSignalProvider`**: Fixed generator type provider
- **`RiskOffSignalProviderFactory`**: Factory for creating providers

### 4. Integration Layer

- Updated `BaseStrategy` to use provider pattern
- Maintained backward compatibility through deprecation warnings
- Updated strategy interface to reflect new naming

## Usage Examples

### Basic Configuration (Default)

```yaml
# No risk-off signals (recommended default)
strategy_config:
  name: "ExampleStrategy"
  risk_off_signal_config:
    type: "NoRiskOffSignalGenerator"
```

### Testing Configuration

```yaml
# For testing with specific risk-off periods
strategy_config:
  name: "TestStrategy"
  risk_off_signal_config:
    type: "DummyRiskOffSignalGenerator"
    default_risk_state: "on"  # risk-on by default
    risk_off_windows:
      - ["2008-09-01", "2009-03-31"]  # Financial crisis
      - ["2020-02-15", "2020-04-30"]  # COVID crash
```

### Strategy Implementation

```python
class MyStrategy(BaseStrategy):
    def generate_signals(self, all_historical_data, benchmark_historical_data, 
                        non_universe_historical_data, current_date, **kwargs):
        # ... calculate initial weights ...
        
        # Apply risk-off filter
        risk_off_generator = self.get_risk_off_signal_generator()
        if risk_off_generator:
            risk_off_signal = risk_off_generator.generate_risk_off_signal(
                all_historical_data, benchmark_historical_data, 
                non_universe_historical_data, current_date
            )
            if risk_off_signal:  # True means risk-off conditions
                final_weights[:] = 0.0  # Zero out all positions
        
        return final_weights_df
```

## Migration Guide

### For Strategy Developers

**Old Pattern (Deprecated)**:
```python
class MyStrategy(BaseStrategy):
    roro_signal_class = DummyRoRoSignal  # Class attribute
    
    def generate_signals(...):
        roro_signal = self.get_roro_signal()
        if roro_signal:
            signal = roro_signal.generate_signal(data, benchmark, date)
            if not signal:  # False meant risk-off in old system
                # Zero out weights
```

**New Pattern**:
```python
class MyStrategy(BaseStrategy):
    # No class attributes needed - configuration-driven
    
    def generate_signals(...):
        risk_off_generator = self.get_risk_off_signal_generator()
        if risk_off_generator:
            signal = risk_off_generator.generate_risk_off_signal(
                all_data, benchmark, non_universe_data, date
            )
            if signal:  # True means risk-off in new system
                # Zero out weights
```

### Key Migration Changes

1. **Method Rename**: `get_roro_signal()` → `get_risk_off_signal_generator()`
2. **Signal Semantics**: Inverted logic - `True` now means risk-off
3. **Configuration**: Use `risk_off_signal_config` in strategy configuration
4. **Method Signature**: Enhanced with `non_universe_historical_data` parameter

## Testing

Comprehensive test suite created covering:
- Interface compliance
- Default behavior (no risk-off signals)
- Dummy implementation with configurable windows
- Provider factory functionality
- Configuration validation
- End-to-end strategy integration

## Backward Compatibility

- Old `roro_signals` module maintained with deprecation warnings
- Existing strategies continue to work but show warnings
- Migration timeline allows gradual transition

## Benefits

1. **Clearer Semantics**: Explicit "risk-off" naming eliminates confusion
2. **Better Architecture**: Follows SOLID principles and framework patterns
3. **Enhanced Testability**: Mockable interfaces and test doubles
4. **Configuration Flexibility**: Runtime configuration instead of class attributes
5. **Type Safety**: Full mypy compliance throughout
6. **Extensibility**: Easy to add new signal generators (VIX-based, technical indicators, etc.)

## Future Extensions

The architecture supports easy addition of real-world risk-off signal generators:

- **VIX-based**: Signals based on volatility index levels
- **Technical**: Moving averages, RSI, momentum indicators
- **Fundamental**: Economic indicators, yield curve analysis
- **Machine Learning**: ML-based regime detection models

## Files Modified/Created

### New Files
- `src/portfolio_backtester/risk_off_signals/interface.py`
- `src/portfolio_backtester/risk_off_signals/implementations.py`
- `src/portfolio_backtester/risk_off_signals/provider.py`
- `src/portfolio_backtester/risk_off_signals/__init__.py`
- `tests/unit/risk_off_signals/test_risk_off_signals.py`
- `config/examples/risk_off_signal_examples.yaml`

### Modified Files
- `src/portfolio_backtester/strategies/base/base_strategy.py`
- `src/portfolio_backtester/interfaces/strategy_base_interface.py`
- `src/portfolio_backtester/strategies/portfolio/momentum/sharpe_momentum_portfolio_strategy.py`
- `src/portfolio_backtester/roro_signals/__init__.py` (added deprecation warning)

## Quality Assurance

All modified files passed:
- Black code formatting
- Ruff linting with auto-fixes
- MyPy type checking
- Comprehensive test coverage
- End-to-end integration testing

The implementation successfully maintains backward compatibility while providing a modern, extensible foundation for risk-off signal generation in the backtesting framework.
