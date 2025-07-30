# Repository Tour

## 🎯 What This Repository Does

**Portfolio Backtester** is a sophisticated Python-based tool for backtesting portfolio strategies with advanced features including two-stage Monte Carlo simulation, walk-forward optimization robustness testing, and comprehensive performance analysis.

**Key responsibilities:**
- Execute and optimize 11 verified trading strategies (Momentum, VAMS, Calmar, Sortino, EMA Crossover, Low Volatility Factor, etc.)
- Perform robust walk-forward optimization with randomized windows and Monte Carlo stress testing
- Generate comprehensive performance analysis and visualization reports
- Provide API stability protection system for critical methods
- Support flexible universe configuration (fixed, named, method-based approaches)
- Deliver high-performance optimization with Numba acceleration and genetic algorithm enhancements

---

## 🏗️ Architecture Overview

### System Context
```
[Market Data Sources] → [Portfolio Backtester] → [Performance Reports]
    (Stooq/yfinance)         ↓                    (Metrics/Visualizations)
                      [Optimization Engine]
                      (Optuna/Genetic Algorithms)
```

### Key Components
- **Strategy Engine** - 11 verified trading strategies with base strategy interface and parameter optimization
- **Optimization System** - Dual engines (Optuna Bayesian + Enhanced Genetic Algorithm v3) with walk-forward validation and robustness testing
- **Data Management** - Hybrid data source with Stooq primary/yfinance fallback, validation, and caching
- **Monte Carlo System** - Two-stage GARCH-based synthetic data generation for parameter robustness and stress testing
- **Reporting Engine** - Comprehensive performance metrics, advanced visualizations, and configurable parameter analysis
- **API Stability System** - Signature protection and validation for critical methods
- **Universe Management** - Flexible configuration system with S&P 500 historical data integration
- **Performance Acceleration** - Numba JIT compilation for high-speed backtesting and optimization

### Data Flow
1. **Data Acquisition**: Hybrid data source fetches market data with automatic failover and validation
2. **Strategy Execution**: Price data flows through strategy signals, position sizing, to portfolio returns
3. **Optimization**: Parameter generators create candidates, walk-forward evaluator tests performance, Monte Carlo adds robustness
4. **Analysis**: Performance metrics calculated, visualizations generated, optimization reports created

---

## 📁 Project Structure [Partial Directory Tree]

```
portfolio-backtester/
├── src/portfolio_backtester/           # Main application code
│   ├── strategies/                     # 11 verified trading strategies
│   │   ├── base_strategy.py           # Base strategy interface with API stability
│   │   ├── momentum_strategy.py       # Momentum-based strategies with Numba optimization
│   │   └── low_volatility_factor_strategy.py
│   ├── optimization/                   # Advanced optimization engines and components
│   │   ├── factory.py                 # Parameter generator factory
│   │   ├── orchestrator.py            # Optimization orchestrator
│   │   ├── evaluator.py               # Backtest evaluator
│   │   ├── adaptive_parameters.py     # Genetic algorithm adaptive parameter control
│   │   ├── advanced_crossover.py      # Advanced crossover operators (SBX, multi-point, etc.)
│   │   ├── elite_archive.py           # Elite preservation system for genetic algorithms
│   │   └── generators/                # Optuna and genetic algorithm generators
│   ├── data_sources/                   # Data acquisition and management
│   │   ├── hybrid_data_source.py      # Primary data source with failover
│   │   ├── yfinance_data_source.py    # yfinance integration
│   │   └── stooq_data_source.py       # Stooq integration
│   ├── monte_carlo/                    # Two-stage Monte Carlo system
│   │   ├── synthetic_data_generator.py # GARCH-based data generation with Numba acceleration
│   │   └── asset_replacement.py       # Asset replacement for robustness
│   ├── backtesting/                    # Pure backtesting components
│   │   ├── strategy_backtester.py     # Strategy execution engine
│   │   └── results.py                 # Results management
│   ├── reporting/                      # Performance analysis and visualization
│   │   ├── performance_metrics.py     # Comprehensive metrics calculation with Numba
│   │   └── plotting.py                # Advanced visualizations
│   ├── api_stability/                  # API stability protection system
│   │   ├── protection.py              # Signature validation and protection
│   │   ├── registry.py                # Signature registry management
│   │   └── api_stable_signatures.json # Reference signatures database
│   ├── universe_data/                  # S&P 500 universe data management
│   │   ├── spy_holdings.py            # SPY holdings data collection
│   │   └── load_kaggle_sp500_data.py  # Kaggle S&P 500 data processing
│   ├── numba_optimized.py             # Numba-accelerated functions
│   ├── numba_kernels.py               # High-performance Numba kernels
│   ├── universe_resolver.py           # Flexible universe configuration resolver
│   ├── core.py                        # Main Backtester class with API stability
│   └── backtester.py                  # CLI entry point
├── config/                            # YAML configuration system
│   ├── parameters.yaml                # Global settings and optimization defaults
│   ├── scenarios/                     # Strategy-specific configurations
│   │   ├── momentum/                  # Momentum strategy scenarios
│   │   └── low_volatility_factor/     # Low volatility scenarios
│   └── universes/                     # Predefined trading universes
├── tests/                             # Comprehensive test suite
│   ├── unit/                          # Unit tests for all components
│   ├── integration/                   # End-to-end integration tests
│   └── fixtures/                      # Test data and utilities
├── docs/                              # Documentation and guides
├── examples/                          # Example configurations and usage
└── pyproject.toml                     # Project dependencies and configuration
```

### Key Files to Know

| File | Purpose | When You'd Touch It |
|------|---------|---------------------|
| `src/portfolio_backtester/backtester.py` | CLI entry point and argument parsing | Adding new CLI options or modes |
| `src/portfolio_backtester/core.py` | Main Backtester class with orchestration logic | Modifying core backtesting workflow |
| `config/parameters.yaml` | Global configuration and optimization defaults | Changing system-wide settings or optimization parameters |
| `config/scenarios/momentum/momentum_simple.yaml` | Example strategy configuration | Creating new strategy scenarios |
| `src/portfolio_backtester/strategies/base_strategy.py` | Base strategy interface with API stability | Implementing new trading strategies |
| `src/portfolio_backtester/optimization/factory.py` | Parameter generator factory | Adding new optimization algorithms |
| `src/portfolio_backtester/optimization/genetic_optimizer.py` | Enhanced genetic algorithm with v3 features | Configuring adaptive parameters, elite preservation, advanced crossover |
| `src/portfolio_backtester/data_sources/hybrid_data_source.py` | Primary data source with failover | Modifying data acquisition logic |
| `src/portfolio_backtester/api_stability/protection.py` | API stability protection system | Managing signature validation for critical methods |
| `src/portfolio_backtester/universe_resolver.py` | Universe configuration resolver | Adding new universe configuration types |
| `src/portfolio_backtester/numba_optimized.py` | Numba-accelerated functions | Adding high-performance optimizations |
| `scripts/update_protected_signatures.py` | API signature update utility | Updating protected method signatures |
| `docs/sp500_universe_management.md` | S&P 500 data management guide | Managing historical constituent data |
| `pyproject.toml` | Project dependencies and build config | Adding new libraries or changing Python version |
| `pytest.ini` | Test configuration | Modifying test execution settings |

---

## 🔧 Technology Stack

### Core Technologies
- **Language:** Python (3.10+) - Modern Python with type hints and advanced features
- **Data Processing:** pandas - Primary data manipulation and time series analysis
- **Numerical Computing:** numpy - High-performance numerical operations and array processing
- **Scientific Computing:** scipy - Statistical functions and optimization algorithms

### Key Libraries
- **Optimization Engines:** Optuna (Bayesian optimization), PyGAD (genetic algorithms)
- **Data Sources:** yfinance (Yahoo Finance), pandas-datareader (Stooq integration)
- **Performance Acceleration:** Numba (JIT compilation for walk-forward evaluation)
- **Visualization:** matplotlib, seaborn, plotly (comprehensive plotting and interactive charts)
- **Statistical Analysis:** statsmodels (GARCH modeling), arch (volatility modeling)
- **Configuration:** PyYAML (YAML configuration management)

### Development Tools
- **Testing Framework:** pytest with comprehensive unit and integration tests
- **Code Quality:** ruff (linting), mypy (type checking), coverage (test coverage)
- **Performance Monitoring:** Rich (enhanced console output), tqdm (progress tracking)
- **Documentation:** Markdown-based documentation with examples

---

## 🌐 External Dependencies

### Required Services
- **Market Data Sources** - Stooq (primary) and Yahoo Finance (fallback) for historical price data
- **File System** - Local storage for data caching, configuration files, and generated reports

### Optional Integrations
- **Database Storage** - SQLite for Optuna study persistence (configurable storage URL)
- **S&P 500 Data** - Kaggle dataset integration for historical constituent data

---

## 🔄 Common Workflows

### Strategy Backtesting (`--mode backtest`)
1. Load scenario configuration from YAML files
2. Fetch market data using hybrid data source with automatic failover
3. Execute strategy logic to generate trading signals
4. Apply position sizing and calculate portfolio returns
5. Generate comprehensive performance metrics and visualizations

**Code path:** `backtester.py` → `core.py` → `strategy_backtester.py` → `base_strategy.py`

### Parameter Optimization (`--mode optimize`)
1. Initialize parameter generator (Optuna or Genetic Algorithm)
2. Generate parameter combinations within defined search space
3. Execute walk-forward optimization with randomized windows
4. Apply Monte Carlo robustness testing with synthetic data
5. Select optimal parameters and generate optimization reports

**Code path:** `backtester.py` → `core.py` → `orchestrator.py` → `evaluator.py` → `parameter_generator.py`

### Monte Carlo Stress Testing
1. Generate GARCH-based synthetic market data preserving statistical properties
2. Replace percentage of assets with synthetic equivalents (5%, 7.5%, 10%)
3. Execute strategy with modified data across multiple simulations
4. Analyze performance degradation and robustness metrics
5. Generate Monte Carlo robustness visualization charts

**Code path:** `monte_carlo_analyzer.py` → `synthetic_data_generator.py` → `asset_replacement.py`

---

## 🚀 Advanced Features

### Genetic Algorithm Enhancements (v3)

The genetic optimizer includes three major optional subsystems for improved convergence:

**Evidence:** `src/portfolio_backtester/optimization/adaptive_parameters.py`, `src/portfolio_backtester/optimization/elite_archive.py`, `src/portfolio_backtester/optimization/advanced_crossover.py`

1. **Adaptive Parameter Control** - Mutation and crossover probabilities adjust during optimization based on population diversity and fitness variance
2. **Elite Preservation System** - Best chromosomes are preserved in a fixed-size archive and periodically reinjected to prevent genetic drift
3. **Advanced Crossover Operators** - Four sophisticated recombination strategies:
   - Simulated Binary Crossover (SBX) for continuous optimization
   - Multi-point Crossover for linked gene problems
   - Uniform Crossover Variant with bias control
   - Arithmetic Crossover for weighted averaging

### API Stability Protection System

**Evidence:** `src/portfolio_backtester/api_stability/` directory with `protection.py`, `registry.py`, `api_stable_signatures.json`

Critical methods are protected with `@api_stable` decorators to prevent breaking changes:
- Signature validation against reference database
- Automatic detection of parameter and return type changes
- Protected methods in core.py, base_strategy.py, and optimization modules
- Update utility: `scripts/update_protected_signatures.py`

### Flexible Universe Configuration

**Evidence:** `src/portfolio_backtester/universe_resolver.py`, extensive test coverage in `tests/unit/strategies/test_universe_configuration.py`

Four different approaches for defining trading universes:
1. **Fixed Universe** - Static list of tickers in YAML configuration
2. **Named Universe** - Reference predefined universe files in `config/universes/`
3. **Multiple Named Universes** - Combine multiple universe files (union of tickers)
4. **Method-Based Universe** - Dynamic generation using methods like `get_top_weight_sp500_components`

### S&P 500 Universe Data Management

**Evidence:** `src/portfolio_backtester/universe_data/` directory, `docs/sp500_universe_management.md`

Comprehensive historical S&P 500 constituent data management:
- Kaggle dataset integration (2009-2024) as primary historical source
- SSGA daily basket XLSX data with 1-day lag
- SEC N-PORT-P XML (monthly) and N-Q HTML (quarterly) data
- Automated data collection and integrity preservation
- Scripts: `spy_holdings.py`, `load_kaggle_sp500_data.py`

### Numba Performance Acceleration

**Evidence:** `src/portfolio_backtester/numba_optimized.py`, `src/portfolio_backtester/numba_kernels.py`, extensive NUMBA integration across strategies

High-performance JIT compilation throughout the codebase:
- Walk-forward evaluation acceleration (10-20× speed improvement)
- Strategy-specific optimizations (momentum scoring, VAMS, Sharpe ratios)
- Monte Carlo synthetic data generation
- Performance metrics calculation (Sortino, drawdown, ATR)
- Environment variable control: `ENABLE_NUMBA_WALKFORWARD=1`

### Advanced Reporting Configuration

**Evidence:** `config/parameters.yaml` advanced_reporting_config section, `src/portfolio_backtester/reporting/optimizer_report_generator.py`

Configurable reporting system balancing performance vs. detail:
- `enable_advanced_parameter_analysis` - Complex correlation/sensitivity analysis (disabled by default)
- `defer_expensive_plots` - Generate expensive visualizations only after optimization
- `defer_parameter_analysis` - Postpone statistical analysis for speed
- Performance vs. robustness trade-offs with optimization modes

---

## 📈 Performance & Scale

### Performance Optimizations
- **Numba Acceleration:** JIT-compiled walk-forward evaluation for 10-20× speed improvement
- **Parallel Processing:** Multi-core optimization with configurable worker processes
- **Smart Caching:** Data preprocessing cache and synthetic data generation cache
- **Deferred Reporting:** Expensive visualizations generated only after optimization completes

### Monitoring
- **Optimization Progress:** Real-time trial progress with pruning for unpromising parameters
- **Performance Metrics:** Comprehensive tracking of Sharpe, Sortino, Calmar ratios, and drawdown analysis
- **Robustness Metrics:** Parameter stability and consistency across walk-forward windows

---

## 🛠️ Development Environment

### WSL Environment Specifics

**Evidence:** `AGENTS.md` development guidelines

This project uses a Windows Python virtual environment (.venv) inside WSL:
- **Correct execution:** `./.venv/Scripts/python.exe script.py`
- **Package management:** Only through `pyproject.toml` editing, then `./.venv/Scripts/python.exe -m pip install -e .[dev]`
- **Testing:** `./.venv/Scripts/python.exe -m pytest tests/ -v`
- **Linting:** `./.venv/Scripts/python.exe -m ruff check src tests`
- **Type checking:** `./.venv/Scripts/python.exe -m mypy src`

### CLI Parameters and Usage

**Evidence:** `src/portfolio_backtester/backtester.py` argument parser, `README.md` examples

Comprehensive command-line interface with advanced options:

**Core Parameters:**
- `--mode` - Required: `backtest`, `optimize` 
- `--scenario-name` - Strategy scenario from config/scenarios/
- `--log-level` - DEBUG, INFO, WARNING, ERROR, CRITICAL

**Optimization Parameters:**
- `--optimizer` - `optuna` (default) or `genetic`
- `--optuna-trials` - Maximum trials (default: 200)
- `--n-jobs` - Parallel workers (default: 8, -1 = all cores)
- `--pruning-enabled` - Enable trial pruning for faster optimization
- `--random-seed` - Set seed for reproducibility

**Genetic Algorithm Parameters:**
- `--ga-advanced-crossover-type` - SBX, multi-point, uniform-variant, arithmetic
- Adaptive mutation and elite preservation configured via YAML

**Performance Metrics Tables:**

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Sharpe Ratio** | Risk-adjusted return (excess return / volatility) | Higher is better |
| **Sortino Ratio** | Downside risk-adjusted return | Higher is better |
| **Calmar Ratio** | Return / Maximum Drawdown | Higher is better |
| **Max Drawdown** | Largest peak-to-trough decline | Lower is better |
| **VaR (95%)** | Value at Risk at 95% confidence | Lower absolute value is better |
| **CVaR (95%)** | Conditional Value at Risk | Lower absolute value is better |

**Stability Metrics (WFO Robustness):**

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Metric_Std** | Standard deviation across WFO windows | Lower indicates more stable performance |
| **Metric_CV** | Coefficient of variation (Std/Mean) | Lower indicates more consistent performance |
| **Metric_Worst_Xpct** | Worst percentile performance | Higher worst-case is better |
| **Metric_Consistency_Ratio** | Fraction of windows above threshold | Higher indicates more reliable performance |

---

## 🚨 Things to Be Careful About

### 🔒 Security Considerations
- **Data Sources:** API rate limits for yfinance, ensure proper error handling for data source failures
- **File System:** Configuration files contain sensitive optimization parameters, ensure proper access controls
- **Caching:** Data cache may contain large datasets, monitor disk space usage

### ⚡ Performance Considerations
- **Memory Usage:** Large universes and long time periods can consume significant memory during optimization
- **Optimization Time:** Genetic algorithms and comprehensive Monte Carlo testing can be time-intensive
- **Data Quality:** Hybrid data source provides robustness but may introduce latency during failover

### 🧪 Testing and Validation
- **Strategy Verification:** All 11 strategies have been post-refactoring verified with comprehensive testing
- **API Stability:** Core methods protected with `@api_stable` decorator to prevent breaking changes
- **Configuration Validation:** YAML configuration files validated on startup to prevent runtime errors

*Updated at: 2025-01-30 15:45:00 UTC*

**Major improvements in this update:**
- Added comprehensive coverage of Genetic Algorithm v3 enhancements (adaptive parameters, elite preservation, advanced crossover)
- Documented API stability protection system with signature validation
- Detailed flexible universe configuration system (fixed, named, method-based approaches)
- Comprehensive Numba performance acceleration documentation
- S&P 500 universe data management system coverage
- Advanced reporting configuration options
- Development environment specifics (WSL)
- CLI parameters and performance metrics tables
- All claims verified with specific file references