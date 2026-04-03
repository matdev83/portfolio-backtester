# Product Overview

Updated: 2026-01-31

Portfolio Backtester is a Python tool for researching, backtesting, and stress-testing portfolio strategies using configuration-driven scenarios.

It is designed for repeatable research: encode experiments as YAML, run backtests/optimizations, and compare results across versions and parameter sets.

## Core Workflows

### Backtest

- Input: a scenario YAML (strategy + universe + timing + sizing + risk controls) and a global parameters YAML.
- Output: reports and artifacts under `data/` (gitignored) suitable for comparing strategy variants.

### Optimization (WFO)

- Run walk-forward optimization (train/test windows) with Optuna or GA and stitch out-of-sample results.
- Preserve reproducibility by pinning scenario files and (optionally) random seeds.

### Robustness / Stress Testing

- Monte Carlo style robustness tooling exists to evaluate stability under synthetic or perturbed data.
- Stress testing is intentionally treated as a second step (after a strategy works on baseline historical data).

## Core Capabilities

- Scenario-driven backtests: YAML defines *what to run*; code implements *how it runs*.
- Walk-forward optimization (rolling/expanding windows) and parameter search.
- Portfolio execution simulation with transaction costs and slippage.
- Risk management hooks (stop loss / take profit) and risk-off signal plumbing.
- Extensible strategy framework with automatic strategy discovery.

## Key Constraints (Non-Negotiable)

- Market data downloads must not be implemented in this repository. All time-series fetching goes through the MDMP integration.
- Strategies must be discoverable via the automatic registry (no manual registration, no hardcoded imports).
- Configuration belongs in YAML. Adding a feature usually implies adding config schema/validation and an example scenario.

## Target Users / Use Cases

- Quants/traders iterating on strategy ideas with fast feedback loops.
- Regression-style validation of changes: keep scenario files stable and re-run to verify no unintended drift.
- Comparative research: swap strategy params/universes/timing rules while keeping the backtest engine constant.

## Non-Goals

- Live trading / brokerage integration.
- High-frequency execution simulation.
- A GUI-first workflow (primary interface is CLI + config files).

## Glossary

- Scenario: a YAML file describing a single experiment (strategy + universe + timing + optimization params).
- Global config: `config/parameters.yaml` (`GLOBAL_CONFIG` block) defining defaults and run-wide settings.
- WFO: walk-forward optimization using train/test windows.

---
_Focus on patterns and purpose, not exhaustive feature lists_
