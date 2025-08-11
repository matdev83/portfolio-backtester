# Development & Profiling Scripts

This directory contains scripts used for development, debugging, and performance profiling of the portfolio backtester. These are intended for developer use and are not part of the core application logic.

## Running the Scripts

All scripts are designed to be run as modules from the project's root directory. Ensure you have activated the virtual environment first.

**Windows:**
```bash
.venv\\Scripts\\python.exe -m dev.scripts.<script_name>
```

**Linux/macOS:**
```bash
.venv/bin/python -m dev.scripts.<script_name>
```

Replace `<script_name>` with the name of the script you wish to run (without the `.py` extension).

---

## Available Scripts

### `line_profile_optimizer.py`

- **Purpose**: A general-purpose profiler for the entire optimization process. It uses `line_profiler` to provide a line-by-line breakdown of the `StrategyBacktester.backtest_strategy` method.
- **When to Use**: This is a good starting point for any performance investigation. Use it to get a high-level overview of where time is being spent during a typical optimization run.
- **Output**: Saves `cProfile` and `line_profiler` results to uniquely timestamped files in the system's temporary directory.

### `profile_wfo.py`

- **Purpose**: A specialized profiler designed to diagnose performance issues within the Walk-Forward Optimization (WFO) logic. It specifically targets the `BacktestEvaluator` and its methods related to evaluating individual WFO windows.
- **When to Use**: Use this script when you suspect that the WFO process itself, rather than a specific strategy's logic, is the source of a slowdown. It was instrumental in identifying the redundant backtesting logic that was a major bottleneck.
- **Output**: Saves `cProfile` and `line_profiler` results to `wfo_cprofile_results.pstats` and `wfo_lprofile_results.txt` in the system's temporary directory.

### `profile_hotspots.py`

- **Purpose**: A highly-focused profiler that targets the most performance-critical functions identified during previous optimization efforts: `generate_signals`, `size_positions`, and `calculate_portfolio_returns`.
- **When to Use**: Use this script to verify the impact of changes made to the core signal generation and portfolio calculation logic. It provides the most granular view of the backtester's "hotspots".
- **Output**: Saves `line_profiler` results to `hotspot_profile_results_final.txt` in the system's temporary directory.
