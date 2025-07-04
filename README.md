# Portfolio Backtester

This project is a Python-based tool for backtesting portfolio strategies.

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

### CLI Parameters for `backtester.py`

* `--log-level`: Set the logging level.
  * **Choices:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
  * **Default:** `INFO`
  * **Description:** Controls the verbosity of the backtester's output.
* `--mode`: Mode to run the backtester in.
  * **Choices:** `backtest`, `optimize`
  * **Required:** Yes
  * **Description:** `backtest` for single scenario backtesting, `optimize` for walk-forward optimization.
* `--scenario-name`: Name of the scenario to run/optimize from `BACKTEST_SCENARIOS` in `src/portfolio_backtester/config.py`.
  * **Required:** Yes
  * **Description:** Specifies which predefined scenario configuration to use.
* `--study-name`: Name of the Optuna study to use.
  * **Description:** In `optimize` mode, this names the study where optimization results are saved. In `backtest` mode, if provided, it loads the best parameters from this study; otherwise, default parameters from the scenario are used.
* `--random-seed`: Set a random seed for reproducibility.
  * **Default:** `None`
* `--optimize-min-positions`: Minimum number of positions to consider during optimization of `num_holdings`.
  * **Default:** `10`
* `--optimize-max-positions`: Maximum number of positions to consider during optimization of `num_holdings`.
  * **Default:** `30`
* `--top-n-params`: Number of top performing parameter values to keep per grid.
  * **Default:** `3`
* `--n-jobs`: Parallel worker processes to use.
  * **Default:** `8` (`-1` means all cores).
* `--early-stop-patience`: Stop optimization after N successive ~zero-return evaluations.
  * **Default:** `10`
* `--optuna-trials`: Maximum trials per WFA slice.
  * **Default:** `200`
* `--optuna-timeout-sec`: Time budget per WFA slice (seconds).
  * **Default:** `None` (no timeout)
* `--optimizer`: Choose the optimization algorithm.
  * **Choices:** `optuna`, `genetic`
  * **Default:** `optuna`
  * **Description:** Selects whether to use Optuna (hyperparameter optimization framework) or a Genetic Algorithm for finding optimal strategy parameters.

#### Optuna Pruning Configuration

Trial pruning can significantly speed up optimization by stopping unpromising trials early. This is based on evaluating intermediate results during the walk-forward analysis. The `MedianPruner` is used.

* `--pruning-enabled`: Enable trial pruning.
  * **Action:** `store_true` (flag, disabled by default)
  * **Description:** If set, enables the `MedianPruner` to stop trials early if their intermediate performance (e.g., average Sharpe ratio over initial walk-forward windows) is poor compared to other trials.
* `--pruning-n-startup-trials`: Number of initial trials to complete before pruning begins.
  * **Default:** `5`
  * **Description:** The pruner will not prune any of the first `N` trials, allowing it to gather initial data.
* `--pruning-n-warmup-steps`: Number of intermediate steps (walk-forward windows) to complete within a trial before it can be pruned.
  * **Default:** `0`
  * **Description:** A trial will report intermediate values after each walk-forward window (or every `pruning-interval-steps`). This parameter specifies how many such reports must occur before the pruner considers pruning that trial.
* `--pruning-interval-steps`: Report intermediate value and check for pruning every N walk-forward windows.
  * **Default:** `1`
  * **Description:** For example, if set to `2`, a trial reports its performance and is eligible for pruning after its 2nd, 4th, 6th, etc., walk-forward window evaluation (subject to `pruning-n-warmup-steps`).

**Note on Early Stopping Mechanisms:**

* **Trial Pruning** (configured above): Stops *individual unpromising trials* early during the optimization process based on intermediate performance across walk-forward windows. This helps focus computational resources on more promising parameter sets.
* `--early-stop-patience`: Stops the *entire optimization study* if a specified number of *consecutive trials* result in near-zero returns in any of their test windows. This acts as a global failsafe for the overall optimization process.

These two mechanisms are complementary.

### Examples

**1. Run an optimization for a scenario:**

```bash
python -m src.portfolio_backtester.backtester --mode optimize --scenario-name "Sharpe_Momentum" --study-name "sharpe_momentum_opt_run_1" --optuna-trials 100 --optuna-timeout-sec 3600
```

**2. Run a backtest using optimized parameters from a study:**

```bash
python -m src.portfolio_backtester.backtester --mode backtest --scenario-name "Sharpe_Momentum" --study-name "sharpe_momentum_opt_run_1"
```

**3. Run a backtest using default parameters for a scenario:**

```bash
python -m src.portfolio_backtester.backtester --mode backtest --scenario-name "Momentum_Unfiltered"
```

**4. Run backtests on all defined scenarios:**

```bash
python -m src.portfolio_backtester.backtester --mode backtest
```
*   This command will iterate through all scenarios defined in `src/portfolio_backtester/config.py` (or `config/scenarios.yaml`) and run a backtest for each.

The tool for downloading SPY holdings can be run with:

```bash
python src/portfolio_backtester/spy_holdings.py --out spy_holdings.csv
```

### CLI Parameters for `spy_holdings.py`

* `--start`: Start date for data download.
  * **Format:** `YYYY-MM-DD`
  * **Default:** `2004-01-01` (earliest SEC N-Q filing)
  * **Description:** Specifies the beginning of the date range for which to download holdings data.
* `--end`: End date for data download.
  * **Format:** `YYYY-MM-DD`
  * **Default:** Today's date
  * **Description:** Specifies the end of the date range for which to download holdings data.
* `--out`: Output filename for the downloaded data.
  * **Format:** `.parquet` or `.csv` extension
  * **Required:** Yes
  * **Description:** The name and format of the file where the downloaded SPY holdings will be saved.
* `--log-level`: Set the logging level.
  * **Choices:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
  * **Default:** `INFO`
  * **Description:** Controls the verbosity of the data downloader's output.

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

### Position Sizing Methods

The project supports pluggable position sizing methods, defined in `src/portfolio_backtester/portfolio/position_sizer.py`. These methods determine how capital is allocated to different assets based on signals and other market data.

* **`equal_weight`**:
  * **Description**: Distributes capital equally among all assets that have a signal. If there are N assets with signals, each receives 1/N of the capital. This is a simple and common method for diversified portfolios.

* **`rolling_sharpe`**:
  * **Description**: Weights positions based on the rolling Sharpe ratio of asset returns. Assets with higher historical risk-adjusted returns (Sharpe ratio) receive a larger allocation. This method aims to maximize the portfolio's overall Sharpe ratio.
  * **Parameters**:
    * `window` (int): The look-back window for calculating rolling returns and standard deviation.

* **`rolling_sortino`**:
  * **Description**: Weights positions based on the rolling Sortino ratio of asset returns. Similar to Sharpe, but it penalizes only downside volatility (returns below a target return). Assets with higher Sortino ratios (better risk-adjusted returns with respect to downside risk) receive larger allocations.
  * **Parameters**:
    * `window` (int): The look-back window for calculating rolling returns and downside deviation.
    * `target_return` (float, default: 0.0): The minimum acceptable return used in the Sortino ratio calculation.

* **`rolling_beta`**:
  * **Description**: Weights positions inversely proportional to their rolling beta against a specified benchmark. Assets with lower beta (less sensitivity to market movements) receive larger allocations, aiming to reduce overall portfolio volatility relative to the market.
  * **Parameters**:
    * `window` (int): The look-back window for calculating rolling covariance and variance.
    * `benchmark` (pd.Series): The benchmark returns used for beta calculation.

* **`rolling_benchmark_corr`**:
  * **Description**: Weights positions inversely proportional to their rolling correlation with a specified benchmark. Assets with lower correlation to the benchmark receive larger allocations, aiming to improve diversification and reduce systematic risk.
  * **Parameters**:
    * `window` (int): The look-back window for calculating rolling correlation.
    * `benchmark` (pd.Series): The benchmark returns used for correlation calculation.

### Optimizer Configuration

## Strategy Optimization

This section details how to configure and run strategy parameter optimization to find the best-performing parameters based on your objectives.

### Supported Performance Metrics

The optimizer can target any of the following performance metrics, which are calculated by the backtester.

| Metric | Description | Goal |
| --- | --- | --- |
| **Total Return** | The total return of the strategy over the entire backtest period. | `maximize` |
| **Ann. Return** | The annualized geometric mean return. | `maximize` |
| **Ann. Vol** | The annualized volatility (standard deviation of returns). | `minimize` |
| **Sharpe** | The annualized Sharpe ratio (risk-adjusted return). | `maximize` |
| **Sortino** | The annualized Sortino ratio (risk-adjusted return focusing on downside volatility). | `maximize` |
| **Calmar** | The Calmar ratio (annualized return divided by the maximum drawdown). | `maximize` |
| **Max DD** | The maximum drawdown of the strategy. | `minimize` |
| **Alpha (ann)** | The annualized alpha of the strategy against the benchmark. | `maximize` |
| **Beta** | The beta of the strategy against the benchmark. | `minimize` or `maximize` |
| **R^2** | The R-squared value of the strategy against the benchmark. | `maximize` |
| **K-Ratio** | The K-Ratio, a measure of return consistency. | `maximize` |
| **ADF Statistic** | The Augmented Dickey-Fuller test statistic for stationarity of the equity curve. | `minimize` |
| **ADF p-value** | The p-value from the ADF test. | `minimize` |
| **Deflated Sharpe** | The Deflated Sharpe Ratio (DSR), which accounts for the probability of "backtest overfitting". | `maximize` |

### Optimization Goals

For each metric, you can specify an optimization goal. The supported goals are:

*   `maximize`: Find parameters that result in the highest possible value for the metric.
*   `minimize`: Find parameters that result in the lowest possible value for the metric.
*   `less_than`: Find parameters where the metric is less than a specified value.
*   `greater_than`: Find parameters where the metric is greater than a specified value.

### YAML Configuration for Optimization

Optimization parameters are configured in the `config/scenarios.yaml` file. Here are some examples of how to set up your optimization scenarios.

#### Single-Objective Optimization

This example shows how to optimize for a single metric (Calmar ratio).

```yaml
# In config/scenarios.yaml
- name: "Calmar_Momentum_Optimization"
  strategy: "MomentumStrategy"
  strategy_params:
    lookback: 12
    num_assets: 20
  optimization_metric: "Calmar"
  optimize:
    - parameter: "lookback"
      type: "int"
      min_value: 3
      max_value: 24
    - parameter: "num_assets"
      type: "int"
      min_value: 5
      max_value: 50
```

#### Multi-Objective Optimization

This example shows how to optimize for multiple metrics simultaneously (Sharpe ratio and Max DD).

```yaml
# In config/scenarios.yaml
- name: "Multi_Objective_Sharpe_MaxDD"
  strategy: "MomentumStrategy"
  strategy_params:
    lookback: 12
    num_assets: 20
  optimization_targets:
    - name: "Sharpe"
      direction: "maximize"
    - name: "Max DD"
      direction: "minimize"
  optimize:
    - parameter: "lookback"
      type: "int"
      min_value: 3
      max_value: 24
    - parameter: "num_assets"
      type: "int"
      min_value: 5
      max_value: 50
```

#### Optimization with Constraints

This example shows how to optimize for the Sharpe ratio while ensuring the Beta remains below a certain value.

```yaml
# In config/scenarios.yaml
- name: "Sharpe_With_Beta_Constraint"
  strategy: "MomentumStrategy"
  strategy_params:
    lookback: 12
    num_assets: 20
  optimization_targets:
    - name: "Sharpe"
      direction: "maximize"
  optimization_constraints:
    - name: "Beta"
      max_value: 0.8
  optimize:
    - parameter: "lookback"
      type: "int"
      min_value: 3
      max_value: 24
    - parameter: "num_assets"
      type: "int"
      min_value: 5
      max_value: 50
```

### Optuna vs. Genetic Algorithm

The backtester supports two optimization algorithms: Optuna and a genetic algorithm. You can choose between them using the `--optimizer` command-line argument.

| Feature | Optuna | Genetic Algorithm |
| --- | --- | --- |
| **Algorithm** | Bayesian optimization (TPE sampler) | Evolutionary algorithm (NSGA-II for multi-objective) |
| **Search Strategy** | Builds a probabilistic model of the objective function and uses it to select the most promising parameters to try next. | Evolves a population of solutions over several generations, using selection, crossover, and mutation to find optimal solutions. |
| **Configuration** | Fewer hyperparameters to tune. The main parameters are the number of trials and the timeout. | More hyperparameters to tune (population size, mutation rate, crossover rate, etc.). |
| **Use Cases** | Good for a wide range of problems, especially when the objective function is expensive to evaluate. | Can be effective for problems with a large number of parameters or complex, non-smooth objective functions. |
| **Multi-Objective** | Supported directly. | Supported via NSGA-II, which finds a set of non-dominated solutions (the Pareto front). |

#### When to Choose Optuna

*   You have a relatively small number of parameters to optimize.
*   The objective function is expensive to evaluate (e.g., long backtests).
*   You want a good solution quickly.

#### When to Choose the Genetic Algorithm

*   You have a large number of parameters or a very complex search space.
*   You are looking for a globally optimal solution and are willing to spend more time searching.
*   You want to explore a diverse set of solutions (the Pareto front in multi-objective optimization).

### Multi-Objective Optimization in Detail

When you run a multi-objective optimization, the optimizer tries to find a set of solutions that represent the best possible trade-offs between the different objectives.

*   **With Optuna**, the result is a set of "best" trials, each representing a different trade-off. The study will report the best trial based on a weighted combination of the objectives.
*   **With the Genetic Algorithm**, the result is the Pareto front, which is a set of solutions where you cannot improve one objective without making another objective worse. The optimizer will then select one solution from the Pareto front as the "best" (typically the one that is most balanced or best for the first objective specified).

To run a multi-objective optimization, simply define multiple `optimization_targets` in your `scenarios.yaml` file, as shown in the example above.



The default search space for optimizable parameters is now defined in `src/portfolio_backtester/config.py` within the `OPTIMIZER_PARAMETER_DEFAULTS` dictionary. This centralizes the configuration and makes it easier to manage.

Individual scenarios in `BACKTEST_SCENARIOS` can still override these defaults by specifying `min_value`, `max_value`, and `step` within their `optimize` section.

### Genetic Algorithm (GA) Settings

When using the Genetic Algorithm (`--optimizer genetic`), its behavior can be tuned using parameters specified in `src/portfolio_backtester/config_loader.py` under `OPTIMIZER_PARAMETER_DEFAULTS` (with keys typically starting `ga_`) or overridden per scenario within the `genetic_algorithm_params` dictionary in your scenario configuration.

Key GA parameters include:

* **`ga_num_generations`**:
  * **Description**: The number of generations the GA will run.
  * **Default**: `100`
* **`ga_sol_per_pop`**:
  * **Description**: The number of solutions (individuals) in each population.
  * **Default**: `50`
* **`ga_num_parents_mating`**:
  * **Description**: The number of solutions to be selected as parents for the next generation.
  * **Default**: `10`
* **`ga_parent_selection_type`**:
  * **Description**: Method for selecting parents (e.g., `sss` for steady-state selection, `rws` for roulette wheel, `tournament`).
  * **Default**: `sss`
* **`ga_crossover_type`**:
  * **Description**: Method for crossover (e.g., `single_point`, `two_points`, `uniform`).
  * **Default**: `single_point`
* **`ga_mutation_type`**:
  * **Description**: Method for mutation (e.g., `random`, `swap`, `adaptive`).
  * **Default**: `random`
* **`ga_mutation_percent_genes`**:
  * **Description**: The percentage of genes to mutate in each chromosome. Can be "default" for PyGAD's internal default, or a specific percentage (e.g., 10 for 10%).
  * **Default**: `"default"`

These parameters are defined with their defaults in `src/portfolio_backtester/optimization/genetic_optimizer.py` via `get_ga_optimizer_parameter_defaults()` and are loaded into the global `OPTIMIZER_PARAMETER_DEFAULTS`. You can override them in `config/parameters.yaml` under the `OPTIMIZER_PARAMETER_DEFAULTS` section, or for a specific scenario by adding a `genetic_algorithm_params` dictionary to its configuration in `config/scenarios.yaml`.

**Example Scenario Override for GA:**

```yaml
# In config/scenarios.yaml
- name: "My_GA_Optimized_Strategy"
  # ... other strategy settings ...
  optimizer: "genetic" # Not a direct config, but implies these params are relevant
  genetic_algorithm_params:
    ga_num_generations: 150
    ga_sol_per_pop: 75
    ga_mutation_type: "adaptive"
  optimize:
    # ... parameter optimization specs ...
```

## Development

### Development Practices and Standards

To ensure the long-term quality, maintainability, and scalability of this project, all contributors are expected to adhere to the following development practices and principles:

### Modular, Layered Architecture

The project follows a modular and layered architecture. This approach promotes separation of concerns and allows for proper code re-use. Each component should have a single, well-defined responsibility and interact with other components through clear interfaces.

### Test-Driven Development (TDD)

We practice Test-Driven Development. This means that for any new feature or bug fix, a test should be written *before* the implementation code. The development cycle is as follows:

1. **Red:** Write a failing test that captures the requirements of the new feature.
2. **Green:** Write the simplest possible code to make the test pass.
3. **Refactor:** Clean up and optimize the code while ensuring all tests still pass.

### SOLID Principles

We adhere to the SOLID principles of object-oriented design:

* **S - Single-responsibility Principle:** A class should have only one reason to change, meaning it should have only one job or responsibility.
* **O - Open-closed Principle:** Software entities (classes, modules, functions, etc.) should be open for extension but closed for modification. This means you should be able to add new functionality without changing existing code.
* **L - Liskov Substitution Principle:** Subtypes must be substitutable for their base types. In other words, objects of a superclass should be replaceable with objects of a subclass without affecting the correctness of the program.
* **I - Interface Segregation Principle:** No client should be forced to depend on methods it does not use. This principle suggests that larger interfaces should be split into smaller, more specific ones.
* **D - Dependency Inversion Principle:** High-level modules should not depend on low-level modules. Both should depend on abstractions. Abstractions should not depend on details; details should depend on abstractions.

### KISS (Keep It Simple, Stupid)

We favor simplicity in our designs and implementations. Avoid unnecessary complexity and over-engineering. A simple, clear solution is always preferable to a complex one, as it is easier to understand, maintain, and debug.

### Convention over Configuration

The project prefers convention over configuration. This means we rely on established conventions to reduce the number of decisions a developer needs to make. Defaults should be sane, logical, and work out-of-the-box for the most common use cases, while still allowing for configuration when necessary.
