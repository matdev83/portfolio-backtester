# Portfolio Backtester

This project is a sophisticated Python-based tool for backtesting portfolio strategies with advanced features including two-stage Monte Carlo simulation, walk-forward optimization robustness testing, and comprehensive performance analysis.

## Key Features

### Core Capabilities
- **Multiple Strategy Types**: Momentum, VAMS, Calmar, Sortino-based strategies
- **Advanced Position Sizing**: Equal weight, volatility-based, risk-adjusted sizing
- **Flexible Rebalancing**: Monthly, quarterly, custom frequencies
- **Transaction Cost Modeling**: Realistic cost simulation with basis points

### Advanced Optimization
- **Walk-Forward Optimization (WFO)**: Robust parameter optimization with time-series validation
- **WFO Robustness Features**: Randomized window sizes and start dates for enhanced robustness
- **Multi-Objective Optimization**: Simultaneous optimization of multiple metrics (Sharpe, Sortino, Max Drawdown)
- **Dual Optimization Engines**: Optuna (Bayesian) and Genetic Algorithm support
- **Trial Pruning**: Early stopping of unpromising parameter combinations

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
- **Robustness Configuration**: Fine-tuned control over WFO randomization
- **Monte Carlo Configuration**: Detailed control over synthetic data generation
- **Advanced Reporting Configuration**: Control over statistical analysis generation for faster optimization
- **Strategy Parameter Defaults**: Centralized optimization parameter management

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
* `--scenario-name`: Name of the scenario from `config/scenarios.yaml`
  * **Required:** Yes for optimize/monte_carlo modes
* `--log-level`: Set the logging level
  * **Choices:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
  * **Default:** `INFO`

#### Optimization Parameters
* `--optimizer`: Choose the optimization algorithm
  * **Choices:** `optuna`, `genetic`
  * **Default:** `optuna`
* `--optuna-trials`: Maximum trials per optimization
  * **Default:** `200`
* `--optuna-timeout-sec`: Time budget per optimization (seconds)
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
* `--early-stop-patience`: Stop after N consecutive poor trials
  * **Default:** `10`

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

**2. Advanced Optimization with Robustness:**
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
  --scenario-name "VAMS_Momentum" \
  --optimizer genetic \
  --optuna-trials 200 \
  --study-name "genetic_vams_opt"
```

## Configuration

### WFO Robustness Configuration (`config/parameters.yaml`)

```yaml
wfo_robustness_config:
  enable_window_randomization: true
  enable_start_date_randomization: true
  train_window_randomization:
    min_offset: 3    # Minimum months to add to base train window
    max_offset: 14   # Maximum months to add to base train window
  test_window_randomization:
    min_offset: 3    # Minimum months to add to base test window
    max_offset: 14   # Maximum months to add to base test window
  start_date_randomization:
    min_offset: 0    # Minimum months to offset start date
    max_offset: 12   # Maximum months to offset start date
  stability_metrics:
    enable: true
    worst_percentile: 10
    consistency_threshold: 0.0
  random_seed: null  # Set for reproducible randomization
```

### Monte Carlo Configuration

```yaml
monte_carlo_config:
  enable_synthetic_data: true
  enable_during_optimization: true    # Stage 1: Lightweight MC during optimization
  enable_stage2_stress_testing: true  # Stage 2: Full stress testing after optimization
  replacement_percentage: 0.05        # 5% of assets replaced with synthetic data
  min_historical_observations: 200    # Minimum data for parameter estimation
  
  garch_config:
    model_type: "GARCH"
    p: 1
    q: 1
    distribution: "studentt"
    bounds:
      omega: [1e-6, 1.0]
      alpha: [0.01, 0.3]
      beta: [0.5, 0.99]
      nu: [2.1, 30.0]
  
  generation_config:
    buffer_multiplier: 1.2
    max_attempts: 2
    validation_tolerance: 0.3
  
  validation_config:
    enable_validation: false  # Disabled during optimization for speed
    tolerance: 0.8
```

### Advanced Reporting Configuration

```yaml
advanced_reporting_config:
  # Enable/disable advanced hyperparameter statistical analysis
  enable_advanced_parameter_analysis: false  # Disabled by default for faster optimization
  
  # Enable/disable regular performance metrics calculation
  enable_performance_metrics: true  # Always enabled for basic reporting
  
  # Enable/disable comprehensive optimization reports
  enable_optimization_reports: true  # Enabled by default for organized report directories
```

**Advanced Parameter Analysis includes:**
- Parameter correlation matrices
- Parameter sensitivity analysis  
- Parameter importance ranking
- Parameter stability analysis
- Parameter robustness assessment
- Parameter performance heatmaps

**When disabled (default):**
- ✅ **Faster optimization** - No complex statistical plots generated
- ✅ **Cleaner reports** - Focus on essential performance metrics
- ✅ **Regular performance metrics** - Sharpe, Calmar, drawdown analysis still included
- ✅ **Basic visualizations** - Performance plots and trial analysis still generated

## Performance Metrics

The backtester calculates comprehensive performance metrics:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Total Return** | Cumulative return over period | Higher is better |
| **Annualized Return** | Geometric mean annual return | Higher is better |
| **Volatility** | Annualized standard deviation | Lower is generally better |
| **Sharpe Ratio** | Risk-adjusted return (excess return / volatility) | Higher is better |
| **Sortino Ratio** | Downside risk-adjusted return | Higher is better |
| **Calmar Ratio** | Return / Maximum Drawdown | Higher is better |
| **Max Drawdown** | Largest peak-to-trough decline | Lower is better |
| **VaR (95%)** | Value at Risk at 95% confidence | Lower absolute value is better |
| **CVaR (95%)** | Conditional Value at Risk | Lower absolute value is better |

### Stability Metrics (WFO Robustness)

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Metric_Std** | Standard deviation across WFO windows | Lower indicates more stable performance |
| **Metric_CV** | Coefficient of variation (Std/Mean) | Lower indicates more consistent performance |
| **Metric_Worst_Xpct** | Worst percentile performance | Higher worst-case is better |
| **Metric_Consistency_Ratio** | Fraction of windows above threshold | Higher indicates more reliable performance |

## Architecture

### Two-Stage Monte Carlo System

```
Stage 1 (During Optimization):
Parameter Trial -> Lightweight MC (5% replacement) -> Robustness Test

Stage 2 (Post-Optimization):
Optimal Params -> Comprehensive MC (5%-15% levels) -> Stress Analysis & Visualization
```

### WFO Robustness Process

```
Standard WFO:
[Train 36m][Test 12m] -> [Train 36m][Test 12m] -> [Train 36m][Test 12m]

Robust WFO:
[Train 39-50m][Test 15-26m] -> [Train 42-53m][Test 9-20m] -> [Train 45-56m][Test 12-23m]
        Randomized windows        Different start dates    Stability metrics
```

## Testing

The project includes comprehensive test coverage:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/monte_carlo/ -v          # Monte Carlo tests
python -m pytest tests/optimization/ -v         # WFO robustness tests
python -m pytest tests/reporting/ -v            # Visualization tests
python -m pytest tests/integration/ -v          # End-to-end tests

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Advanced Features

### Walk-Forward Optimization Robustness

Traditional WFO uses fixed window sizes, which can lead to overfitting to specific time periods. The enhanced WFO system introduces:

1. **Window Randomization**: Train and test windows are randomly extended within configured bounds
2. **Start Date Randomization**: Starting points are randomly offset to test different market regimes
3. **Stability Metrics**: Performance consistency is measured across randomized windows
4. **Parameter Impact Analysis**: Comprehensive analysis of parameter sensitivity and importance

### Two-Stage Monte Carlo Process

The Monte Carlo system operates in two distinct stages:

**Stage 1 (Optimization Robustness)**:
- Lightweight synthetic data injection during parameter optimization
- 5-10% asset replacement with GARCH-generated synthetic data
- Fast generation with minimal validation for optimization speed
- Tests parameter stability against slightly modified market conditions

**Stage 2 (Strategy Stress Testing)**:
- Comprehensive stress testing after optimization completes
- Multiple replacement levels (5%, 7.5%, 10%, 12.5%, 15%)
- Full validation and quality checks for realistic synthetic data
- Generates detailed robustness analysis and visualization

### GARCH-Based Synthetic Data

The synthetic data generation uses sophisticated GARCH models to preserve:
- **Volatility Clustering**: Periods of high/low volatility
- **Return Distribution Properties**: Skewness, kurtosis, tail behavior
- **Autocorrelation Structure**: Serial correlation in returns and squared returns
- **Jump-Diffusion Processes**: Sudden market movements and regime changes

## Dependencies

### Core Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing and statistics
- **matplotlib**: Plotting and visualization
- **yfinance**: Financial data acquisition
- **optuna**: Bayesian optimization framework
- **pygad**: Genetic algorithm optimization
- **arch**: GARCH modeling for synthetic data

### Advanced Features
- **rich**: Enhanced console output and progress bars
- **tqdm**: Progress tracking for long operations
- **statsmodels**: Statistical modeling and analysis
- **seaborn**: Statistical data visualization
- **PyYAML**: YAML configuration file handling