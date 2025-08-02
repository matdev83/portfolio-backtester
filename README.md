# Portfolio Backtester

This project is a sophisticated Python-based tool for backtesting portfolio strategies with advanced features including two-stage Monte Carlo simulation, walk-forward optimization robustness testing, and comprehensive performance analysis.

## Quick Start

Run a simple momentum strategy backtest with a single command:

```bash
# Make sure you have installed the dependencies (see Setup section)
python -m src.portfolio_backtester.backtester --mode backtest --scenario-name "Momentum_Unfiltered"
```

This will run the backtest and generate a performance report in the `data/reports` directory.

### Optimization Shortcut

You can run an optimization with a single command using the provided shortcut script:

```bash
./optimize.py <strategy_config.yaml> [<optional_optimizer_args>...]
```

For example:
```bash
./optimize.py config/scenarios/portfolio/calmar_momentum_strategy/default.yaml --optuna-trials 100 --n-jobs 4
```

This will invoke the optimizer with the specified scenario YAML and any additional arguments you provide. The script automatically uses the correct Python environment and CLI flags.

## Key Features

### Core Capabilities
- **Multiple Strategy Types**: 11+ fully verified strategies including Momentum, VAMS, Calmar, Sortino, EMA Crossover, Low Volatility Factor, and Static Allocation
- **Advanced Position Sizing**: Equal weight, volatility-based, risk-adjusted sizing
- **Comprehensive Rebalancing**: 40+ supported frequencies from hourly to bi-annual (see [Rebalance Frequencies](#rebalance-frequencies))
- **Static Allocation Strategies**: Fixed weight and volatility-targeted allocation strategies for long-term investing
- **Transaction Cost Modeling**: Realistic cost simulation with basis points
- **Production-Ready**: All strategies verified post-refactoring with comprehensive testing

### Advanced Optimization
- **Walk-Forward Optimization (WFO)**: Robust parameter optimization with time-series validation
- **WFO Robustness Features**: Randomized window sizes and start dates for enhanced robustness
- **Multi-Objective Optimization**: Simultaneous optimization of multiple metrics (Sharpe, Sortino, Max Drawdown)
- **Dual Optimization Engines**: Optuna (Bayesian) and Genetic Algorithm support
- **Advanced Genetic Algorithm Features**: Adaptive parameter control, elite preservation, and sophisticated crossover operators
- **Trial Pruning**: Early stopping of unpromising parameter combinations
- **Static Strategy Optimization**: Comprehensive optimization of asset weights and rebalancing frequencies for allocation strategies

### Two-Stage Monte Carlo System
- **Stage 1 (During Optimization)**: Lightweight synthetic data injection for parameter robustness testing
- **Stage 2 (Post-Optimization)**: Comprehensive stress testing with multiple replacement levels
- **GARCH-Based Synthetic Data**: Realistic market condition simulation preserving statistical properties
- **Asset Replacement Strategy**: Configurable percentage of assets replaced with synthetic equivalents

### Advanced Analytics & Reporting
- **Comprehensive Performance Metrics**: Sharpe, Sortino, Calmar ratios, drawdown analysis
- **Stability Metrics**: Parameter consistency across walk-forward windows
- **Trial P&L Visualization**: Monte Carlo-style plots showing optimization trial performance
- **Parameter Impact Analysis**: Sensitivity, correlation, and importance ranking (configurable)
- **Robustness Stress Testing**: Visual analysis of strategy performance under synthetic market conditions
- **Configurable Reporting**: Advanced hyperparameter analysis can be disabled for faster optimization

### Enhanced Configuration System
- **YAML-Based Configuration**: Flexible parameter and scenario management
- **Configurable Universes**: Three flexible ways to define trading universes (fixed, named, method-based)
- **Robustness Configuration**: Fine-tuned control over WFO randomization
- **Monte Carlo Configuration**: Detailed control over synthetic data generation
- **Advanced Reporting Configuration**: Control over statistical analysis generation for faster optimization
- **Strategy Parameter Defaults**: Centralized optimization parameter management

### Fail-Tolerance Data Gathering
- **Hybrid Data Source**: Automatic failover between Stooq and yfinance data sources
- **Data Validation**: Comprehensive validation of downloaded data quality
- **Format Normalization**: Consistent MultiIndex output regardless of source
- **Failure Tracking**: Detailed reporting of data source failures and successes
- **Configurable Preferences**: Choose primary data source (Stooq or yfinance)

## Setup

1. **Create a virtual environment:**

    ```bash
    python -m venv .venv
    ```

2. **Activate the virtual environment:**
    * **Windows:**

        ```bash
        .venv\Scripts\activate
        ```

    * **macOS/Linux:**

        ```bash
        source .venv/bin/activate
        ```

3. **Install dependencies:**

    ```bash
    pip install -e .
    ```

## Usage

The main backtesting script can be run directly as a Python module:

```bash
python -m src.portfolio_backtester.backtester
```

### CLI Parameters

#### Core Parameters
* `--mode`: Mode to run the backtester in.
  * **Choices:** `backtest`, `optimize`, `monte_carlo`
  * **Required:** Yes
  * **Description:** 
    - `backtest`: Single scenario backtesting
    - `optimize`: Walk-forward optimization with robustness features
    - `monte_carlo`: Full Monte Carlo stress testing analysis
* `--scenario-name`: Name of the scenario from `config/scenarios/` subdirectories
  * **Required:** Yes for optimize/monte_carlo modes
* `--log-level`: Set the logging level
  * **Choices:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
  * **Default:** `INFO`

#### S&P 500 Universe Data Management
For instructions on how to manage and update the S&P 500 historical constituent data, please refer to the [S&P 500 Universe Data Management Guide](docs/sp500_universe_management.md).
All related scripts are located in `src/portfolio_backtester/universe_data/`.

#### Optimization Parameters
* `--optimizer`: Choose the optimization algorithm
  * **Choices:** `optuna`, `genetic`
  * **Default:** `optuna`
* `--optuna-trials`: Maximum trials per optimization
  * **Default:** `200`
* `--timeout`: Time budget per optimization (seconds)
  * **Default:** `None` (no timeout)
* `--n-jobs`: Parallel worker processes
  * **Default:** `8` (`-1` means all cores)
* `--random-seed`: Set random seed for reproducibility
  * **Default:** `None`

#### Advanced Optimization Features
* `--pruning-enabled`: Enable trial pruning for faster optimization
* `--pruning-n-startup-trials`: Trials to complete before pruning begins
  * **Default:** `5`
* `--pruning-interval-steps`: Report interval for pruning checks
  * **Default:** `1`

##### Early Stopping Parameters
* `--early-stop-patience`: Stop after N consecutive trials with poor relative performance
  * **Default:** `10`
  * **Purpose:** Optimization efficiency - stops when no improvement is being made
  * **Trigger:** Trials perform worse than recent best (e.g., 0.15 → 0.14 → 0.13...)
* `--early-stop-zero-trials`: Stop after N consecutive trials with exactly zero values
  * **Default:** `20`
  * **Purpose:** Problem detection - identifies fundamental configuration/data issues
  * **Trigger:** Trials return exactly 0.0 (indicating data unavailability or setup errors)
  * **When to use:** Set lower (5-10) when testing new scenarios to quickly catch configuration problems

#### Monte Carlo Parameters
* `--mc-simulations`: Number of Monte Carlo simulations
  * **Default:** `1000`
* `--mc-years`: Years to project in Monte Carlo analysis
  * **Default:** `10`
* `--interactive`: Show plots interactively

### Examples

**1. Basic Strategy Backtest:**
```bash
python -m src.portfolio_backtester.backtester --mode backtest --scenario-name "Momentum_Unfiltered"
```

**2. Advanced Optimization with Robustness (using scenario name):**
```bash
python -m src.portfolio_backtester.backtester \
  --mode optimize \
  --scenario-name "Momentum_Unfiltered" \
  --study-name "robust_momentum_v1" \
  --optimizer optuna \
  --optuna-trials 500 \
  --pruning-enabled \
  --n-jobs -1 \
  --random-seed 42
```

**3. Advanced Optimization with Robustness (using scenario filename):**
```bash
python -m src.portfolio_backtester.backtester \
  --mode optimize \
  --scenario-filename "config/scenarios/momentum/Momentum_Unfiltered.yaml" \
  --study-name "robust_momentum_v1" \
  --optimizer optuna \
  --optuna-trials 500 \
  --pruning-enabled \
  --n-jobs -1 \
  --random-seed 42
```


**3. Monte Carlo Stress Testing:**
```bash
python -m src.portfolio_backtester.backtester \
  --mode monte_carlo \
  --scenario-name "Momentum_Unfiltered" \
  --mc-simulations 2000 \
  --mc-years 15 \
  --interactive
```

**4. Genetic Algorithm Optimization:**
```bash
python -m src.portfolio_backtester.backtester \
  --mode optimize \
  --scenario-name vams_momentum_strategy/default \
  --optimizer genetic \
  --optuna-trials 200 \
  --study-name "genetic_vams_opt"
```

**5. Optimization with Early Stopping for Zero Values:**
```bash
python -m src.portfolio_backtester.backtester \
  --mode optimize \
  --scenario-filename "config/scenarios/signal/intramonth_seasonal_strategy/TLT_long_month_1.yaml" \
  --early-stop-zero-trials 5 \
  --optuna-trials 100
```
*This example shows how to stop optimization early if 5 consecutive trials return zero values, which is useful for detecting data availability issues or configuration problems.*

## Configuration

The backtester uses a set of YAML files in the `config/` directory to manage global settings, define specific backtest experiments, and provide examples.

### Universe Configuration

The `universe_config` section of a scenario YAML defines which assets are included in the backtest or optimization. Supported types are:

- `single_symbol`: A single ticker (e.g., SPY)
- `fixed`: A fixed list of tickers
- `named`: A named universe (e.g., S&P 500)
- `method`: A programmatically generated universe

**Example: Single Symbol Universe**
```yaml
universe_config:
  type: single_symbol
  ticker: SPY
```

**Example: Fixed Universe**
```yaml
universe_config:
  type: fixed
  tickers: [SPY, GLD, QQQ]
```

**Example: Named Universe**
```yaml
universe_config:
  type: named
  universe_name: sp500_top50
```

**Example: Method Universe**
```yaml
universe_config:
  type: method
  method_name: get_top_weight_sp500_components
  n_holdings: 20
```

### Commission and Slippage Configuration

Commission and slippage are configured globally in the `config/parameters.yaml` file. The system uses a detailed, IBKR-style model by default.

**To configure commissions**, modify the following parameters in `config/parameters.yaml`:

```yaml
# Default commission per share
commission_per_share: 0.005

# Minimum commission per order
commission_min_per_order: 1.0

# Maximum commission as a percentage of trade value
commission_max_percent_of_trade: 0.005

# Slippage in basis points (1 bps = 0.01%)
slippage_bps: 2.5
```

### Creating a Custom Commission Model

To implement a custom commission model, follow these steps:

1.  **Create a new class** that inherits from `TransactionCostModel` in `src/portfolio_backtester/trading/transaction_costs.py`.
2.  **Implement the `calculate` method**. This method should take `turnover`, `weights_daily`, `price_data`, and `portfolio_value` as input and return a tuple containing the total costs as a pandas Series and a dictionary with a breakdown of the costs.
3.  **Register your new model** in the `get_transaction_cost_model` factory function in the same file.

    ```python
    def get_transaction_cost_model(config: dict) -> TransactionCostModel:
        model_name = config.get("transaction_cost_model", "realistic").lower()
        if model_name == "realistic":
            return RealisticTransactionCostModel(config)
        elif model_name == "your_custom_model":
            return YourCustomModel(config)
        else:
            raise ValueError(f"Unsupported transaction cost model: {model_name}")
    ```
4.  **Select your model** in `config/parameters.yaml`:

    ```yaml
    transaction_cost_model: your_custom_model
    ```

### Detailed Configuration Guides

*   **[WFO Robustness Configuration](docs/configuration_wfo.md)**
*   **[Monte Carlo Configuration](docs/configuration_carlo.md)**
*   **[Genetic Algorithm Configuration](docs/configuration_genetic_algorithm.md)**

## Contributing

Contributions are welcome! If you'd like to help improve the Portfolio Backtester, please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See the [LICENSE.md](LICENSE.md) file for details.