# Vulture Dead Code Analysis Report

**Generated:** 2025-08-10  
**Tool:** Vulture 2.14  
**Confidence Threshold:** 70%+  
**Project:** Portfolio Backtester

## Executive Summary

- **Total Issues Found:** 16
- **In Source (`src/`)**: 3
- **In Tests (`tests/`)**: 13
- **Primary Category:** Unused variables

## Current Findings (70%+)

```text
src\portfolio_backtester\optimization\performance\genetic_optimizer.py:223: unused variable 'fitness_function' (100% confidence)
src\portfolio_backtester\optimization\standalone_evaluator.py:14: unused variable 'get_strategy_method' (100% confidence)
src\portfolio_backtester\yaml_validator.py:213: unused variable 'col_idx' (100% confidence)
tests\base\integration_test_base.py:204: unused variable 'expected_transformations' (100% confidence)
tests\base\strategy_test_base.py:282: unused variable 'top_fraction' (100% confidence)
tests\conftest.py:563: unused variable 'nextitem' (100% confidence)
tests\conftest.py:569: unused variable 'session' (100% confidence)
tests\conftest.py:581: unused variable 'session' (100% confidence)
tests\unit\core\test_config_loader.py:99: unused variable 'encoding' (100% confidence)
tests\unit\optimization\test_parameter_generator_interface.py:46: unused variable 'optimization_spec' (100% confidence)
tests\unit\portfolio\test_position_sizer.py:84: unused variable 'daily_points_per_month' (100% confidence)
tests\unit\test_attribute_accessor_dip.py:301: unused variable 'backtest_data' (100% confidence)
tests\unit\test_attribute_accessor_dip.py:301: unused variable 'strategy_state' (100% confidence)
tests\unit\test_attribute_accessor_dip.py:359: unused variable 'backtest_data' (100% confidence)
tests\unit\test_attribute_accessor_dip.py:359: unused variable 'strategy_state' (100% confidence)
tests\unit\test_detailed_commissions_fastpath_equivalence.py:32: unused variable 'use_numba' (100% confidence)
```

The full raw output is saved at `dev/vulture_70.txt`.

## Comparison to Previous Report

- Previous findings (60%+ threshold) reported hundreds of potential issues and flagged items in `optuna_objective.py` and `parallel_execution.py`.
- Those specific issues no longer appear at 70%+:
  - No hit for `optuna_objective.py` (e.g., previously reported `trial_params`).
  - No hits for `parallel_execution.py` exception variables.
- Current high-confidence issues are limited and mostly within tests; in `src/`, only three unused variables remain.

## Recommendations (Actionable)

- **src findings**
  - `optimization/performance/genetic_optimizer.py:223`: Remove or use `fitness_function`.
  - `optimization/standalone_evaluator.py:14`: Remove or use `get_strategy_method`.
  - `yaml_validator.py:213`: Remove or use `col_idx`.
- **tests findings**
  - Clean up unused local variables; where variables are placeholders, prefix with `_` or use `del` to document intent.
- Optionally generate a whitelist for intentional cases: run with `--make-whitelist` and keep it under `dev/vulture_whitelist.py`.

---

This document reflects the latest Vulture scan at 70% confidence and supersedes prior, broader inventories.