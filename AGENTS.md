# AGENTS.md

> **AI Agent Operating Manual for Portfolio Backtester**

This document provides comprehensive instructions for AI coding agents working on the Portfolio Backtester codebase. Follow all guidelines precisely.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Critical Environment Setup](#critical-environment-setup)
3. [Project Architecture](#project-architecture)
4. [Key Files Reference](#key-files-reference)
5. [Essential Commands](#essential-commands)
6. [Coding Standards](#coding-standards)
7. [Testing Strategy](#testing-strategy)
8. [Post-Edit Quality Assurance](#post-edit-quality-assurance)
9. [Configuration & YAML Guidelines](#configuration--yaml-guidelines)
10. [Strategy Development](#strategy-development)
11. [Provider Interface System](#provider-interface-system)
12. [Data Sources & MDMP Integration](#data-sources--mdmp-integration)
13. [Common Pitfalls & Troubleshooting](#common-pitfalls--troubleshooting)
14. [API Stability & Protected Signatures](#api-stability--protected-signatures)
15. [Agent Development Principles](#agent-development-principles)
16. [Pull Request Workflow](#pull-request-workflow)
17. [Kiro-style SDD Workflows](#kiro-style-sdd-workflows)
18. [Project Steering Management](#project-steering-management)

---

## Project Overview

**Portfolio Backtester** is a sophisticated Python-based tool for backtesting portfolio strategies with advanced features including:

- **Walk-Forward Optimization (WFO)** with robustness testing
- **Two-Stage Monte Carlo Simulation** for stress testing
- **11+ Verified Strategy Types** (Momentum, VAMS, Calmar, Sortino, EMA Crossover, etc.)
- **Advanced Risk Management** (ATR-based stop-loss and take-profit)
- **Provider Interface Architecture** following SOLID principles
- **Dual Optimization Engines** (Optuna Bayesian + Genetic Algorithm)

### Primary Use Cases

- Strategy development and backtesting
- Parameter optimization
- Monte Carlo stress testing
- Risk management analysis

---

## Critical Environment Setup

### ⚠️ MANDATORY: Python Virtual Environment

This project uses a **Windows Python virtual environment** located in `.venv/`. You **MUST** prefix all Python commands with the virtual environment interpreter:

```bash
# ✅ CORRECT - Always use this format
.venv\Scripts\python.exe <command>
.venv\Scripts\python.exe -m <module>

# ❌ INCORRECT - NEVER use these
python <command>
python3 <command>
source .venv/bin/activate
```

### Examples of Correct Usage

```bash
# Running the backtester
.venv\Scripts\python.exe -m src.portfolio_backtester.backtester --mode backtest

# Running pytest
.venv\Scripts\python.exe -m pytest tests/ -v

# Running linters
.venv\Scripts\python.exe -m ruff check src/
.venv\Scripts\python.exe -m mypy src/
.venv\Scripts\python.exe -m black src/

# Installing packages (editable mode only)
.venv\Scripts\python.exe -m pip install -e .[dev]
```

---

## Project Architecture

### Directory Structure

```text
portfolio-backtester/
├── src/
│   └── portfolio_backtester/          # Main package
│       ├── backtester.py              # 🚀 ENTRY POINT
│       ├── backtester_logic/          # Core backtesting logic
│       ├── strategies/                # Strategy implementations
│       │   ├── base_strategy.py       # Base classes
│       │   ├── portfolio/             # Portfolio strategies
│       │   └── signal/                # Signal strategies
│       ├── interfaces/                # Provider interfaces (SOLID)
│       │   ├── universe/              # IUniverseProvider
│       │   ├── position_sizing/       # IPositionSizerProvider
│       │   ├── stop_loss/             # IStopLossProvider
│       │   └── take_profit/           # ITakeProfitProvider
│       ├── optimization/              # Optuna & Genetic optimization
│       ├── monte_carlo/               # Monte Carlo simulation
│       ├── reporting/                 # Report generation
│       ├── timing/                    # Rebalancing timing logic
│       ├── trading/                   # Transaction costs & execution
│       ├── risk_management/           # Stop loss & take profit
│       ├── data_sources/              # Data provider integrations
│       └── config_loader.py           # YAML configuration loading
├── tests/
│   ├── unit/                          # Fast, isolated unit tests
│   ├── integration/                   # Integration tests
│   ├── system/                        # End-to-end tests
│   ├── fixtures/                      # Shared test data
│   ├── base/                          # Base test classes
│   └── conftest.py                    # Pytest configuration
├── config/
│   ├── parameters.yaml                # Global parameters
│   ├── scenarios/                     # Strategy scenario configs
│   │   ├── builtins/                  # Built-in strategies
│   │   └── examples/                  # Example configurations
│   └── universes/                     # Universe definitions
├── scripts/                           # Utility scripts
├── docs/                              # Documentation
└── data/                              # Data storage (gitignored)
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `backtester.py` | Entry point, CLI argument parsing, mode dispatch |
| `backtester_logic/` | Core backtesting workflow (generate signals → size positions → execute) |
| `strategies/` | Strategy implementations (generate_signals, get_params_space) |
| `interfaces/` | Provider interfaces following SOLID (dependency injection) |
| `optimization/` | WFO, Optuna sampler, Genetic Algorithm |
| `monte_carlo/` | GARCH-based synthetic data generation |
| `timing/` | Rebalancing frequencies (40+ supported) |
| `config_loader.py` | YAML parsing and validation |

---

## Key Files Reference

### Critical Files (High Impact)

| File | Purpose | Caution Level |
|------|---------|---------------|
| `src/portfolio_backtester/backtester.py` | Main entry point | 🔴 Critical |
| `src/portfolio_backtester/strategies/base_strategy.py` | Strategy base class | 🔴 Critical |
| `src/portfolio_backtester/config_loader.py` | Configuration loading | 🔴 Critical |
| `src/portfolio_backtester/scenario_validator.py` | Config validation | 🔴 Critical |
| `config/parameters.yaml` | Global parameters | 🟠 High |
| `tests/conftest.py` | Test fixtures | 🟠 High |

### Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project dependencies, tool configs |
| `pytest.ini` | Pytest markers and settings |
| `bandit.yaml` | Security analysis config |
| `.vscode/settings.json` | IDE settings |

### Protected API Files

| File | Notes |
|------|-------|
| `api_stable_signatures.json` | Protected method signatures |
| Methods decorated with `@api_stable` | Must update signatures after changes |

---

## Essential Commands

### Setup & Installation

```bash
# Install in editable mode with dev dependencies
.venv\Scripts\python.exe -m pip install -e .[dev]

# Verify installation
.venv\Scripts\python.exe -c "import portfolio_backtester; print('OK')"
```

### Backtesting Modes

```bash
# Simple backtest
.venv\Scripts\python.exe -m src.portfolio_backtester.backtester \
  --mode backtest \
  --scenario-filename "config/scenarios/builtins/portfolio/simple_momentum_strategy/default.yaml"

# Optimization
.venv\Scripts\python.exe -m src.portfolio_backtester.backtester \
  --mode optimize \
  --scenario-filename "config/scenarios/builtins/signal/ema_crossover_signal_strategy/default.yaml" \
  --optuna-trials 100 \
  --n-jobs 4

# Monte Carlo stress testing
.venv\Scripts\python.exe -m src.portfolio_backtester.backtester \
  --mode monte_carlo \
  --scenario-filename "path/to/scenario.yaml" \
  --mc-simulations 1000
```

### Quality Assurance

```bash
# Linting
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\python.exe -m ruff check --fix <file>

# Formatting
.venv\Scripts\python.exe -m black <file>

# Type checking
.venv\Scripts\python.exe -m mypy src

# Security analysis
.venv\Scripts\python.exe -m bandit -r src -c bandit.yaml

# Dead code detection
.venv\Scripts\python.exe -m vulture src tools/vulture_whitelist.py
```

### Testing

```bash
# Run all tests
.venv\Scripts\python.exe -m pytest tests/ -v

# Run specific test file
.venv\Scripts\python.exe -m pytest tests/unit/strategies/test_momentum.py -v

# Run tests by marker
.venv\Scripts\python.exe -m pytest -m "unit and fast"
.venv\Scripts\python.exe -m pytest -m "integration"
.venv\Scripts\python.exe -m pytest -m "not slow"

# Run with coverage
.venv\Scripts\python.exe -m pytest --cov=src/portfolio_backtester
```

---

## Coding Standards

### Python Requirements

| Aspect | Requirement |
|--------|-------------|
| **Version** | Python 3.10+ |
| **Formatting** | Black (line-length 100) |
| **Linting** | Ruff with Google docstring convention |
| **Type Hints** | **REQUIRED** for all function signatures |
| **Imports** | Absolute imports within `src/portfolio_backtester/` |
| **Naming** | `snake_case` functions/variables, `PascalCase` classes |
| **Docstrings** | Google style for all public functions/classes |
| **Logging** | Use `logging` module (not print statements) |

### Type Hints Example

```python
# ✅ CORRECT - Full type hints
def calculate_returns(
    prices: pd.DataFrame,
    period: int = 1,
    include_first: bool = False,
) -> pd.Series:
    """Calculate returns from price data.
    
    Args:
        prices: DataFrame with OHLCV columns.
        period: Lookback period for returns.
        include_first: Whether to include first row.
        
    Returns:
        Series of returns indexed by date.
    """
    ...

# ❌ INCORRECT - Missing type hints
def calculate_returns(prices, period=1, include_first=False):
    ...
```

### Architectural Principles

- **TDD** (Test-Driven Development) - Write tests first
- **SOLID** - Single responsibility, Open/closed, Liskov, Interface segregation, Dependency inversion
- **DRY** - Don't Repeat Yourself
- **KISS** - Keep It Simple
- **Convention over Configuration** - Use sensible defaults

---

## Testing Strategy

### Test Categories & Markers

| Marker | Purpose | Speed Target |
|--------|---------|--------------|
| `@pytest.mark.unit` | Isolated unit tests | < 0.1s each |
| `@pytest.mark.integration` | Component integration | < 1s each |
| `@pytest.mark.system` | End-to-end tests | < 10s each |
| `@pytest.mark.fast` | Fast-running tests | < 0.1s |
| `@pytest.mark.slow` | Slow tests | Variable |
| `@pytest.mark.network` | Network-dependent | Skip offline |
| `@pytest.mark.strategy` | Strategy tests | Variable |

### Coverage Requirements

| Scope | Minimum |
|-------|---------|
| Overall | 65% (configured in pyproject.toml) |
| New code | 90%+ |
| Critical paths (strategies, core) | 95%+ |

### Base Test Classes

Use the appropriate base class for test consistency:

```python
# For strategy tests
from tests.base.strategy_test_base import BaseStrategyTest

class TestMyStrategy(BaseStrategyTest):
    def test_signals(self):
        data = self.generate_test_data()
        # ...

# For timing tests
from tests.base.timing_test_base import BaseTimingTest

# For integration tests
from tests.base.integration_test_base import BaseIntegrationTest
```

### Property-Based Testing (Hypothesis)

The project uses Hypothesis for property-based testing. See:

- `tests/strategies/common_strategies.py` - Reusable strategies
- `CONTRIBUTING.md` - Detailed Hypothesis guidelines

---

## Post-Edit Quality Assurance

### ⚠️ MANDATORY After Every Python File Edit

After editing any `.py` file, you **MUST** run the following QA pipeline:

```bash
.venv\Scripts\python.exe -m ruff check --fix <modified_file> && \
.venv\Scripts\python.exe -m black <modified_file> && \
.venv\Scripts\python.exe -m mypy <modified_file>
```

### Using MCP Tool (Preferred)

If the `patch_file` MCP tool is available, use it instead of other editing tools. It automatically:

- Validates patch uniqueness
- Runs QA checks after edit
- Creates git snapshots for rollback

### Verification Checklist

Before marking any task complete:

- [ ] All modified files pass `ruff check`
- [ ] All modified files pass `black` formatting
- [ ] All modified files pass `mypy` type checking
- [ ] Related tests pass: `.venv\Scripts\python.exe -m pytest tests/path/to/tests -v`
- [ ] No regressions: `.venv\Scripts\python.exe -m pytest tests/ -x`

---

## Configuration & YAML Guidelines

### Scenario Configuration Structure

```yaml
# config/scenarios/builtins/portfolio/<strategy>/default.yaml

# Universe configuration
universe_config:
  type: fixed  # or: single_symbol, named, method
  tickers: [SPY, QQQ, IWM]

# Strategy configuration
strategy_config:
  name: "MyStrategy"
  strategy: "SimpleMomentumPortfolioStrategy"
  allocation_mode: "reinvestment"  # or "fixed_fractional"
  strategy_params:
    lookback_period: 63
    hold_period: 21

# Risk management (optional)
strategy_params:
  stop_loss_config:
    type: "AtrBasedStopLoss"
    atr_length: 14
    atr_multiple: 2.0
  take_profit_config:
    type: "AtrBasedTakeProfit"
    atr_length: 21
    atr_multiple: 3.0

# Optimization parameters (for optimize mode)
optimize:
  - parameter: lookback_period
    min_value: 20
    max_value: 126
    step: 7
```

### Universe Types

| Type | Usage | Example |
|------|-------|---------|
| `single_symbol` | Single ticker | `ticker: SPY` |
| `fixed` | Fixed list | `tickers: [SPY, QQQ]` |
| `named` | Named universe | `universe_name: sp500_top50` |
| `method` | Dynamic | `method_name: get_top_weight_sp500_components` |

### Configuration Rules

- ❌ Never hardcode configuration in Python files
- ✅ Always validate YAML syntax before committing
- ✅ Use `config/examples/` for reference configurations
- ✅ Keep scenario files focused on a single strategy

---

## Strategy Development

### Strategy Types

| Type | Base Class | Purpose |
|------|------------|---------|
| Portfolio | `PortfolioStrategy` | Multi-asset allocation |
| Signal | `SignalStrategy` | Single-asset signals |
| Meta | `BaseMetaStrategy` | Strategy composition |

### Creating a New Strategy

1. **Create strategy file** in `src/portfolio_backtester/strategies/portfolio/` or `/signal/`
2. **Inherit from base class**
3. **Implement required methods**:
   - `generate_signals(data, **kwargs) -> pd.DataFrame`
   - `get_params_space() -> dict` (for optimization)
4. **Register in factory** (if needed)
5. **Create scenario YAML** in `config/scenarios/`
6. **Add tests** in `tests/unit/strategies/`

### Strategy Template

```python
from typing import Any
import pandas as pd
from src.portfolio_backtester.strategies.base_strategy import PortfolioStrategy

class MyNewStrategy(PortfolioStrategy):
    """My new portfolio strategy.
    
    Args:
        config: Strategy configuration dictionary.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.lookback = config.get("lookback_period", 63)
    
    def generate_signals(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate trading signals.
        
        Args:
            data: OHLCV price data.
            **kwargs: Additional arguments.
            
        Returns:
            DataFrame with signal weights per asset.
        """
        # Implementation here
        ...
    
    @staticmethod
    def get_params_space() -> dict[str, Any]:
        """Get parameter optimization space."""
        return {
            "lookback_period": {"type": "int", "low": 20, "high": 126},
        }
```

---

## Provider Interface System

The project uses a **Provider Interface Architecture** following SOLID principles. All strategies receive provider instances automatically.

### Core Interfaces

| Interface | Purpose | Implementations |
|-----------|---------|-----------------|
| `IUniverseProvider` | Asset universe selection | `ConfigBasedUniverseProvider`, `FixedListUniverseProvider` |
| `IPositionSizerProvider` | Position sizing | `ConfigBasedPositionSizerProvider`, `FixedPositionSizerProvider` |
| `IStopLossProvider` | Stop loss logic | `ConfigBasedStopLossProvider`, `FixedStopLossProvider` |
| `ITakeProfitProvider` | Take profit logic | `ConfigBasedTakeProfitProvider`, `FixedTakeProfitProvider` |

### Usage in Strategies

```python
# Providers are automatically injected
class MyStrategy(PortfolioStrategy):
    def generate_signals(self, data, **kwargs):
        # Access providers
        universe = self.get_universe_provider().get_universe_symbols()
        sizer = self.get_position_sizer_provider()
        stop_loss = self.get_stop_loss_provider()
        take_profit = self.get_take_profit_provider()
```

### Enforcement

- **Runtime Enforcement** - Legacy patterns will fail
- **Static Analysis** - `scripts/enforce_provider_usage.py`
- **Pre-commit Hooks** - Automated checking

---

## Data Sources & MDMP Integration

### ⚠️ CRITICAL: Data Access Rules

> **Agents are PROHIBITED from creating code that downloads time series data in this project.**

All market data access **MUST** go through the **Market Data Multi-Provider (MDMP)** package.

### What is MDMP?

**MDMP** (`market-data-multi-provider`) is a **separate local package** that handles all data fetching, caching, and normalization. It is:

- **Location**: `C:\Users\Mateusz\source\repos\market-data-multi-provider`
- **Shortcut/Alias**: "MDMP"
- **Dependency**: Already configured in `pyproject.toml` and installed locally
- **Workspace**: Part of the same VS Code workspace (see `portfolio-backtester.code-workspace`)

### Data Access Architecture

```text
┌─────────────────────────────────────┐
│     Portfolio Backtester            │
│  (this project - CONSUMES data)     │
│                                     │
│   ┌─────────────────────────────┐   │
│   │   MDMPDataSource adapter    │   │
│   │  src/.../data_sources/      │   │
│   └──────────────┬──────────────┘   │
└──────────────────┼──────────────────┘
                   │ uses
                   ▼
┌─────────────────────────────────────┐
│   Market Data Multi-Provider        │
│  (MDMP - PROVIDES data)             │
│                                     │
│  • Symbol definitions               │
│  • Data fetching logic              │
│  • Provider failover                │
│  • Caching & normalization          │
└─────────────────────────────────────┘
```

### Using MDMP in This Project

The proper way to access data in Portfolio Backtester:

```python
# Via the data source interface (PREFERRED)
from portfolio_backtester.interfaces.data_source_interface import create_data_source

ds = create_data_source({"data_source": "mdmp"})
data = ds.get_data(["SPY", "QQQ"], start_date, end_date)

# Or directly (less common)
from market_data_multi_provider import Client

client = Client()
data = client.fetch("SPY", period="daily", start="2020-01-01")
```

### ❌ What Agents MUST NOT Do

| Prohibited Action | Why |
|-------------------|-----|
| Creating new data downloaders | MDMP handles all data fetching |
| Adding `yfinance.download()` calls | Use MDMP instead |
| Adding `requests.get()` for market data | Use MDMP instead |
| Creating new data provider classes | Configure MDMP symbols file |
| Fetching data from APIs directly | All data goes through MDMP |

### ✅ What To Do If Data/Symbols Are Missing

If a backtest requires symbols or data series that aren't available:

1. **DO NOT** create download code in Portfolio Backtester
2. **DO** inform the user that the symbol needs to be added to MDMP
3. **The fix must happen in MDMP**, not here:
   - Symbol definitions: MDMP's symbol configuration files
   - New data providers: MDMP's provider implementations
   - Data processing: MDMP's normalization logic

**Example response when symbols are missing:**

> "The symbol `XYZ` is not available in the MDMP data source. To add this symbol:
>
> 1. Open the MDMP project at `C:\Users\Mateusz\source\repos\market-data-multi-provider`
> 2. Add the symbol to the appropriate symbols definition file
> 3. Re-run the data download in MDMP
> 4. The symbol will then be available in Portfolio Backtester"

### MDMP Data Source Priority

MDMP automatically handles provider failover:

1. **Stooq** (primary - free historical data)
2. **yfinance** (fallback - Yahoo Finance)
3. **CBOE** (options data)
4. **FRED** (economic data)
5. **TradingView** (specialized, requires auth)

### Local Data Storage

- **Data cache**: `data/` directory (gitignored)
- **Internal cache**: `src/cache/` directory
- **Configuration**: `config/parameters.yaml`

### Symbol Mapping

Portfolio Backtester uses a symbol mapper for MDMP compatibility:

- **Mapper location**: `src/portfolio_backtester/data_sources/symbol_mapper.py`
- **Purpose**: Converts local ticker names to MDMP canonical IDs
- **Automatic**: Usually transparent; MDMP handles resolution

---

## Common Pitfalls & Troubleshooting

### ❌ Common Mistakes

| Mistake | Solution |
|---------|----------|
| Using `python` instead of `.venv\Scripts\python.exe` | Always prefix with venv path |
| Using `pip install` directly | Edit `pyproject.toml` and reinstall with `-e .[dev]` |
| Removing functions without explicit request | Only add/improve, never remove unless asked |
| Skipping type hints | All functions must have type hints |
| Skipping QA checks after edit | Always run ruff/black/mypy |

### 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Check absolute imports, verify `__init__.py` files |
| Type check failures | Run `mypy` with `--show-error-codes` for details |
| Test fixture not found | Check `tests/conftest.py` for fixture definitions |
| API stability test failure | Run `scripts/update_protected_signatures.py --quiet` |
| Slow tests | Use cached fixtures, reduce data size |

### Debug Commands

```bash
# Verbose mypy output
.venv\Scripts\python.exe -m mypy src --show-error-codes --pretty

# Single-file ruff with all messages
.venv\Scripts\python.exe -m ruff check src/path/to/file.py --show-fixes

# Pytest with verbose output and stop on first failure
.venv\Scripts\python.exe -m pytest tests/ -v -x --tb=short
```

---

## API Stability & Protected Signatures

### When to Update Signatures

The `@api_stable` decorator protects critical method signatures. Update only when:

1. Adding a new `@api_stable` decorated method
2. Changing signature of an existing `@api_stable` method
3. Removing an `@api_stable` decorated method

### Update Command

```bash
.venv\Scripts\python.exe scripts/update_protected_signatures.py --quiet
```

Then commit the updated `api_stable_signatures.json` file.

### Test-Driven Updates

If tests fail with signature mismatch errors, run the update command.

---

## Agent Development Principles

### ✅ ALLOWED Actions

- Adding new functions, methods, classes
- Improving existing implementations
- Adding tests for new/modified code
- Fixing bugs and issues
- Performing maintenance (refactoring, cleanup)
- Adding documentation

### ❌ PROHIBITED Actions

- **Creating data download code** (all data fetching goes through MDMP - see [Data Sources & MDMP Integration](#data-sources--mdmp-integration))
- Installing packages via `pip` directly (edit `pyproject.toml` instead)
- Removing functions/features without explicit user request
- Degrading code quality or test coverage
- Making destructive changes without confirmation
- Skipping verification before marking tasks complete

### Dependency Management

```bash
# ❌ NEVER do this
pip install some-package

# ✅ CORRECT approach
# 1. Edit pyproject.toml to add dependency
# 2. Reinstall:
.venv\Scripts\python.exe -m pip install -e .[dev]
```

### Verification Before Completion

Before marking any task complete, you **MUST**:

1. Run specific tests related to changes
2. Run full test suite to check for regressions
3. Ensure all QA checks pass
4. Update documentation if user-facing changes

---

## Pull Request Workflow

### Commit Guidelines

1. Make **atomic commits** with clear, descriptive messages
2. Prefix commits with type: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
3. Reference issues when applicable: `fix: resolve issue #42`

### Pre-Commit Checklist

- [ ] All tests pass
- [ ] Linting passes (`ruff check`)
- [ ] Type checking passes (`mypy`)
- [ ] Formatting is correct (`black`)
- [ ] Documentation updated if needed
- [ ] No new abstractions without explicit request

### Code Review Standards

- [ ] Type hints on all functions
- [ ] Docstrings on public functions
- [ ] Tests for new functionality
- [ ] No hardcoded configuration
- [ ] Provider interfaces used (no legacy patterns)

---

## Kiro-style SDD Workflows

The project implements **Spec-Driven Development (SDD)** using the Kiro framework. This ensures that features are well-defined before implementation begins.

### Usage Context & Detection

**Agents must not force the use of Kiro-based SDD.** 

- **Check User Intent**: Before starting a task, determine if it is part of a Kiro workflow.
- **SDD Mode**: Only use Kiro workflows if the user explicitly uses `/kiro` commands, mentions "Kiro", "specification", or asks for the SDD process.
- **Direct Mode**: If the user provides direct instructions without mentioning Kiro, follow the standard workflow (Understand → Plan → Implement → Verify) without creating Kiro spec files.

### Lifecycle of a Specification

Specifications follow a strict phase-separated lifecycle in `.kiro/specs/[feature-name]/`:

1.  **Init**: Initialize the spec directory and metadata.
2.  **Requirements**: Define "what" needs to be built (using EARS format).
3.  **Design**: Define "how" it will be built (architecture, components).
4.  **Tasks**: Break down the design into actionable, parallelizable tasks.
5.  **Implementation**: Execute the tasks and verify against requirements.

### Using Kiro Slash Commands

Agents should use the following slash commands (via `Task` or direct invocation if supported) to manage the SDD lifecycle:

- `/kiro:spec-init <description>`: Start a new feature spec.
- `/kiro:spec-requirements <feature>`: Generate/update requirements.
- `/kiro:spec-design <feature>`: Create/update technical design.
- `/kiro:spec-tasks <feature>`: Generate implementation task list.
- `/kiro:spec-impl <feature>`: Execute implementation based on tasks.
- `/kiro:spec-status <feature>`: Check progress of a specification.

---

## Project Steering Management

**Steering files** in `.kiro/steering/` serve as persistent project memory. They capture patterns and principles that guide development.

### Core Steering Files

| File | Purpose |
|------|---------|
| `product.md` | Core purpose, value proposition, and high-level capabilities. |
| `tech.md` | Frameworks, technology decisions, and coding standards. |
| `structure.md` | Project organization, naming conventions, and import patterns. |

### Steering Principles for Agents

- **Patterns over Lists**: Document architectural patterns, not exhaustive file listings.
- **Golden Rule**: If new code follows existing patterns, steering shouldn't need updating.
- **Preservation**: User customizations in steering files are sacred; updates should be additive.
- **Syncing**: Use `/kiro:steering` to sync codebase changes back to steering files to prevent "knowledge drift".

---

## Quick Reference Card

```bash
# === SETUP ===
.venv\Scripts\python.exe -m pip install -e .[dev]

# === RUN ===
.venv\Scripts\python.exe -m src.portfolio_backtester.backtester --mode backtest \
  --scenario-filename "config/scenarios/builtins/portfolio/simple_momentum_strategy/default.yaml"

# === TEST ===
.venv\Scripts\python.exe -m pytest tests/ -v

# === QA (after every edit) ===
.venv\Scripts\python.exe -m ruff check --fix <file> && \
.venv\Scripts\python.exe -m black <file> && \
.venv\Scripts\python.exe -m mypy <file>

# === API STABILITY (when needed) ===
.venv\Scripts\python.exe scripts/update_protected_signatures.py --quiet
```

---

### You Are Not Alone

Never assume you are the only one agent working on this codebase. You may encounter files changed by other agents from time to time while you are working on your tasks.

**NEVER** ever try to **remove** or **revert back** changes to the unrelated code made by other agents or the human.

---
