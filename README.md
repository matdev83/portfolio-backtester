# Portfolio Backtester

This project is a Python-based tool for backtesting portfolio strategies.

## Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

2.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -e .
    ```

## Usage

The main backtesting script can be run directly:

```bash
python src/portfolio_backtester/backtester.py
```

### CLI Parameters for `backtester.py`

*   `--log-level`: Set the logging level.
    *   **Choices:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
    *   **Default:** `INFO`
    *   **Description:** Controls the verbosity of the backtester's output.
*   `--mode`: Mode to run the backtester in.
    *   **Choices:** `backtest`, `optimize`
    *   **Required:** Yes
    *   **Description:** `backtest` for single scenario backtesting, `optimize` for walk-forward optimization.
*   `--scenario-name`: Name of the scenario to run/optimize from `BACKTEST_SCENARIOS` in `src/portfolio_backtester/config.py`.
    *   **Required:** Yes
    *   **Description:** Specifies which predefined scenario configuration to use.
*   `--study-name`: Name of the Optuna study to use.
    *   **Description:** In `optimize` mode, this names the study where optimization results are saved. In `backtest` mode, if provided, it loads the best parameters from this study; otherwise, default parameters from the scenario are used.
*   `--random-seed`: Set a random seed for reproducibility.
    *   **Default:** `None`
*   `--optimize-min-positions`: Minimum number of positions to consider during optimization of `num_holdings`.
    *   **Default:** `10`
*   `--optimize-max-positions`: Maximum number of positions to consider during optimization of `num_holdings`.
    *   **Default:** `30`
*   `--top-n-params`: Number of top performing parameter values to keep per grid.
    *   **Default:** `3`
*   `--n-jobs`: Parallel worker processes to use.
    *   **Default:** `8` (`-1` means all cores).
*   `--early-stop-patience`: Stop optimization after N successive ~zero-return evaluations.
    *   **Default:** `10`
*   `--optuna-trials`: Maximum trials per WFA slice.
    *   **Default:** `200`
*   `--optuna-timeout-sec`: Time budget per WFA slice (seconds).
    *   **Default:** `None` (no timeout)

### Examples:

**1. Run an optimization for a scenario:**
```bash
python src/portfolio_backtester/backtester.py --mode optimize --scenario-name "Sharpe_Momentum" --study-name "sharpe_momentum_opt_run_1" --optuna-trials 100 --optuna-timeout-sec 3600
```

**2. Run a backtest using optimized parameters from a study:**
```bash
python src/portfolio_backtester/backtester.py --mode backtest --scenario-name "Sharpe_Momentum" --study-name "sharpe_momentum_opt_run_1"
```

**3. Run a backtest using default parameters for a scenario:**
```bash
python src/portfolio_backtester/backtester.py --mode backtest --scenario-name "Momentum_Unfiltered"
```

The tool for downloading SPY holdings can be run with:

```bash
python src/portfolio_backtester/spy_holdings.py --out spy_holdings.csv
```

### CLI Parameters for `spy_holdings.py`

*   `--start`: Start date for data download.
    *   **Format:** `YYYY-MM-DD`
    *   **Default:** `2004-01-01` (earliest SEC N-Q filing)
    *   **Description:** Specifies the beginning of the date range for which to download holdings data.
*   `--end`: End date for data download.
    *   **Format:** `YYYY-MM-DD`
    *   **Default:** Today's date
    *   **Description:** Specifies the end of the date range for which to download holdings data.
*   `--out`: Output filename for the downloaded data.
    *   **Format:** `.parquet` or `.csv` extension
    *   **Required:** Yes
    *   **Description:** The name and format of the file where the downloaded SPY holdings will be saved.
*   `--log-level`: Set the logging level.
    *   **Choices:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
    *   **Default:** `INFO`
    *   **Description:** Controls the verbosity of the data downloader's output.

Example:
```bash
python src/portfolio_backtester/spy_holdings.py --start 2020-01-01 --end 2023-12-31 --out spy_holdings_2020_2023.parquet --log-level INFO
```

```bash
# First full build (creates data/spy_holdings_full.parquet)
python -m portfolio_backtester.spy_holdings \
       --start 2004-01-01 --end 2025-06-30 \
       --out spy_holdings_full.parquet

# Nightly / incremental refresh (only fetches new dates)
python -m portfolio_backtester.spy_holdings \
       --start 2004-01-01 --end 2025-06-30 \
       --out spy_holdings_full.parquet --update
```

The script saves the file inside the top-level `data/` directory (never in `src/data`).  Internally it automatically forward-fills missing business days so every ticker has a continuous daily weight series.

### Optimizer Configuration

The default search space for optimizable parameters is now defined in `src/portfolio_backtester/config.py` within the `OPTIMIZER_PARAMETER_DEFAULTS` dictionary. This centralizes the configuration and makes it easier to manage.

Individual scenarios in `BACKTEST_SCENARIOS` can still override these defaults by specifying `min_value`, `max_value`, and `step` within their `optimize` section.

## Development

### Directory Structure Map

```
.
├───.gitignore: Specifies intentionally untracked files to ignore.
├───pyproject.toml: Project configuration, dependencies, and build system settings.
├───README.md: Project overview, setup, usage, and development guidelines.
├───.git/: Git version control system directory.
├───.pytest_cache/: Cache directory for pytest.
├───.venv/: Python virtual environment.
├───config/: Configuration files for the backtester (e.g., GLOBAL_CONFIG, BACKTEST_SCENARIOS).
├───data/: Stores downloaded historical data (e.g., stock prices, SPY holdings).
│   ├───^GSPC.csv: S&P 500 index data.
│   ├───^VIX.csv: VIX index data.
│   └───... (various stock/ETF CSVs)
├───docs/: Documentation files and examples.
│   ├───dp_vams_example.md: Documentation for VAMS strategy example.
│   └───dp_vams_example.py: Example code for VAMS strategy.
├───logs/: Directory for application logs.
├───src/: Source code for the portfolio backtester.
│   ├───data/: Contains CSV files for various assets, likely used for testing or specific data loading.
│   │   ├───^GSPC.csv: S&P 500 index data.
│   │   └───... (various stock/ETF CSVs)
│   ├───portfolio_backtester/: Main Python package for the backtester.
│   │   ├───__init__.py: Initializes the Python package.
│   │   ├───backtester.py: Core backtesting logic, runs scenarios, calculates and displays results.
│   │   ├───config.py: Defines global configurations and backtest scenarios.
│   │   ├───spy_holdings.py: Script to download historical SPY holdings data.
│   │   ├───__pycache__/: Python bytecode cache.
│   │   ├───data_sources/: Handles data retrieval from various sources.
│   │   │   ├───__init__.py: Initializes the data_sources package.
│   │   │   ├───base_data_source.py: Abstract base class for data sources.
│   │   │   └───yfinance_data_source.py: Implements data retrieval using yfinance.
│   │   ├───portfolio/: Contains modules related to portfolio management.
│   │   │   ├───__init__.py: Initializes the portfolio package.
│   │   │   ├───position_sizer.py: Logic for determining position sizes.
│   │   │   └───rebalancing.py: Logic for rebalancing portfolio weights.
│   │   ├───reporting/: Modules for generating performance reports and metrics.
│   │   │   ├───__init__.py: Initializes the reporting package.
│   │   │   └───performance_metrics.py: Functions to calculate various performance metrics.
│   │   └───strategies/: Implements different backtesting strategies.
│   │       ├───__init__.py: Initializes the strategies package.
│   │       ├───base_strategy.py: Abstract base class for trading strategies.
│   │       ├───momentum_strategy.py: Implements a basic momentum strategy.
│   │       ├───sharpe_momentum_strategy.py: Implements a Sharpe ratio-based momentum strategy.
│   │       └───vams_momentum_strategy.py: Implements a VAMS (Volatility-Adjusted Momentum Strategy).
│   └───portfolio_backtester.egg-info/: Metadata for the Python package.
└───tests/: Unit and integration tests for the project.
    ├───__init__.py: Initializes the tests package.
    ├───test_backtester.py: Tests for the core backtester logic.
    ├───__pycache__/: Python bytecode cache.
    ├───data_sources/: Tests for data source modules.
    │   ├───__init__.py: Initializes the data_sources tests package.
    │   └───test_yfinance_data_source.py: Tests for the yfinance data source.
    ├───portfolio/: Tests for portfolio management modules.
    │   ├───__init__.py: Initializes the portfolio tests package.
    │   └───test_position_sizer.py: Tests for position sizing logic.
    ├───reporting/: Tests for reporting modules.
    │   ├───__init__.py: Initializes the reporting tests package.
    │   └───test_performance_metrics.py: Tests for performance metrics calculation.
    └───strategies/: Tests for trading strategies.
        ├───__init__.py: Initializes the strategies tests package.
        └───test_momentum_strategy.py: Tests for the momentum strategy.
```

### Development Practices and Standards

To ensure the long-term quality, maintainability, and scalability of this project, all contributors are expected to adhere to the following development practices and principles:

### Modular, Layered Architecture
The project follows a modular and layered architecture. This approach promotes separation of concerns and allows for proper code re-use. Each component should have a single, well-defined responsibility and interact with other components through clear interfaces.

### Test-Driven Development (TDD)
We practice Test-Driven Development. This means that for any new feature or bug fix, a test should be written *before* the implementation code. The development cycle is as follows:
1.  **Red:** Write a failing test that captures the requirements of the new feature.
2.  **Green:** Write the simplest possible code to make the test pass.
3.  **Refactor:** Clean up and optimize the code while ensuring all tests still pass.

### SOLID Principles
We adhere to the SOLID principles of object-oriented design:

*   **S - Single-responsibility Principle:** A class should have only one reason to change, meaning it should have only one job or responsibility.
*   **O - Open-closed Principle:** Software entities (classes, modules, functions, etc.) should be open for extension but closed for modification. This means you should be able to add new functionality without changing existing code.
*   **L - Liskov Substitution Principle:** Subtypes must be substitutable for their base types. In other words, objects of a superclass should be replaceable with objects of a subclass without affecting the correctness of the program.
*   **I - Interface Segregation Principle:** No client should be forced to depend on methods it does not use. This principle suggests that larger interfaces should be split into smaller, more specific ones.
*   **D - Dependency Inversion Principle:** High-level modules should not depend on low-level modules. Both should depend on abstractions. Abstractions should not depend on details; details should depend on abstractions.

### KISS (Keep It Simple, Stupid)
We favor simplicity in our designs and implementations. Avoid unnecessary complexity and over-engineering. A simple, clear solution is always preferable to a complex one, as it is easier to understand, maintain, and debug.

### Convention over Configuration
The project prefers convention over configuration. This means we rely on established conventions to reduce the number of decisions a developer needs to make. Defaults should be sane, logical, and work out-of-the-box for the most common use cases, while still allowing for configuration when necessary.