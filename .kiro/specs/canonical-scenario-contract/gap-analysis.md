# Implementation Gap Analysis

Feature: `canonical-scenario-contract`
Updated: 2026-01-31

## Analysis Summary

- The codebase already has strong *semantic validation* for scenarios, but lacks a single *runtime* normalization step that produces a canonical config consumed uniformly by all execution paths.
- The largest source of divergence is strategy initialization: different paths instantiate strategies with different config shapes and attach scenario context inconsistently.
- Universe and timing configuration are interpreted in multiple places (ticker collectors, providers, and ad-hoc overrides), which increases the risk of "key exists but is ignored".
- Best leverage point for this feature is introducing a canonical normalization step at scenario load/start-of-run, and then ensuring all strategy instantiation and pipeline code consumes that normalized config.

## Context Loaded

- Spec: `.kiro/specs/canonical-scenario-contract/spec.json`
- Requirements: `.kiro/specs/canonical-scenario-contract/requirements.md`
- Steering: `.kiro/steering/product.md`, `.kiro/steering/tech.md`, `.kiro/steering/structure.md`
- Gap analysis framework: `.kiro/settings/rules/gap-analysis.md`

## 1. Current State Investigation

### Key Runtime Components

- Scenario load + validation:
  - `src/portfolio_backtester/config_loader.py` loads YAML, runs syntax validation, then semantic validation (`validate_scenario_semantics`).
  - `src/portfolio_backtester/scenario_validator.py` enforces strategy existence + tunable parameters + some meta-strategy constraints.
  - `src/portfolio_backtester/config_initializer.py` populates/normalizes optimization parameter defaults (but does not normalize runtime scenario shape).

- Execution entry and orchestration:
  - `src/portfolio_backtester/backtester.py` parses CLI args and dispatches.
  - `src/portfolio_backtester/backtester_logic/backtester_facade.py` builds components and runs backtest or optimization.

- Backtest execution paths (important divergence point):
  - `src/portfolio_backtester/backtester_logic/backtest_runner.py` runs backtest mode and instantiates strategies via `StrategyManager`.
  - `src/portfolio_backtester/backtesting/strategy_backtester.py` provides a "pure backtesting engine" used heavily in evaluation/optimization.

### Strategy Discovery and Instantiation

- Strategy discovery is automatic and enforced:
  - `src/portfolio_backtester/strategies/_core/registry/registry/strategy_registry.py` and `.../solid_strategy_registry.py` block manual registration.
  - Discovery scans canonical directories under `src/portfolio_backtester/strategies/{builtins,user}/{portfolio,signal,meta}`.

- Strategy instantiation currently differs by runtime path:
  - `StrategyManager.get_strategy()` uses `StrategyFactory.create_strategy(strategy_name, params)` passing only `scenario_config["strategy_params"]`.
    - File: `src/portfolio_backtester/backtester_logic/strategy_manager.py`
    - Factory: `src/portfolio_backtester/strategies/_core/strategy_factory_impl.py`
  - `StrategyBacktester._get_strategy()` resolves a strategy class via registry, instantiates it with only params, then attaches `instance.config = strategy_config` *after* construction.
    - File: `src/portfolio_backtester/backtesting/strategy_backtester.py`

### Configuration Interpretation Hotspots

- `BaseStrategy` reads configuration during `__init__`:
  - Timing: `timing_config` is read in `_initialize_timing_controller()`.
  - Providers: universe/sizer/stop-loss/take-profit/risk-off providers are created from the dict passed to the constructor.
  - File: `src/portfolio_backtester/strategies/_core/base/base/base_strategy.py`

- Universe resolution exists in multiple places:
  - Via `TickerCollectorFactory` in `StrategyBacktester` and `DataFetcher` (scenario top-level `universe` / `universe_config`).
  - Via strategy's universe provider (`BaseStrategy.get_universe()`), which expects `universe_config` to be present in the strategy's init config.
  - `BacktestRunner.run_scenario()` resolves and then mutates `scenario_config["universe"]` after strategy instantiation.

- Timing resolution has multiple competing inputs:
  - `BaseStrategy` will synthesize a default `timing_config` if none is present in the strategy init config.
  - Pipeline code has ad-hoc overrides in signal generation for specific cases.
  - File: `src/portfolio_backtester/backtester_logic/strategy_logic.py`

### Observable Consistency Risks

- Strategies that rely on `self.config` (attached post-init in some paths) can behave differently in backtest vs optimization paths.
  - Example: `src/portfolio_backtester/strategies/builtins/portfolio/autogluon_sortino_ml_portfolio_strategy.py`.
- Some strategies expect either a nested `strategy_params` dict or a flat params dict, and the codebase currently supports both unevenly.
  - This increases the likelihood of defaults being applied in one path but not another.

## 2. Requirements Feasibility Analysis

The requirements are feasible within the current architecture because:

- There is already a clear "pre-run" stage (scenario load/validation and backtester initialization).
- There is already a central registry/factory mechanism for strategies.
- Providers and timing controllers already exist; they mostly need consistent inputs.

However, the change is cross-cutting: many components currently treat `scenario_config` as both immutable input and mutable working state.

## 3. Requirement-to-Asset Map

| Requirement | Existing Assets | Gap Status | Notes |
| --- | --- | --- | --- |
| 1. Canonical Scenario Normalization | `config_loader.py`, `scenario_validator.py`, `config_initializer.py` | Missing | Validation exists, but no single normalization step creates a canonical runtime representation consumed everywhere. |
| 2. Consistent Semantics Across Execution Modes | `BacktestRunner`, `StrategyBacktester`, `EvaluationEngine` | Missing/Constraint | Multiple execution paths interpret universe/timing differently; outputs can diverge for same YAML. |
| 3. Single Strategy Initialization Contract | `BaseStrategy.__init__`, `StrategyManager`, `StrategyBacktester._get_strategy` | Missing | Strategy init inputs differ (params-only vs params + post-init attach). Providers/timing read config too early to rely on post-init attach. |
| 4. Validation and Error Reporting | `scenario_validator.py`, `yaml_validator.py`, config loader failure modes | Partial | Strong semantic validation exists; missing: conflict detection between equivalent keys and normalization-stage errors. |
| 5. Backward Compatibility and Migration Safety | prefix stripping (factory + validator), scripts under `scripts/` | Partial/Unknown | Some legacy support exists, but canonical contract will require explicit mapping rules and migration messaging. |
| 6. Regression Safety | existing unit tests around config loading/validation | Missing | Tests exist for config loader/validator; tests for cross-path semantic equivalence are not established. |

## 4. Implementation Approach Options

### Option A: Extend Existing Components (Normalize at Load + Fix Strategy Init Inputs)

What this would likely extend:

- Add a normalization step in `src/portfolio_backtester/config_loader.py` (and/or in the facade right after `load_config()`).
- Update strategy creation pathways so they receive a consistent init config:
  - `StrategyManager.get_strategy()` / `StrategyFactory.create_strategy()`
  - `StrategyBacktester._get_strategy()`

Benefits:

- Minimizes new abstractions; leverages existing validation and pipeline.
- Makes the change visible at a single choke point (scenario load).

Trade-offs / Risks:

- Higher risk of "bloating" config_loader/backtester_facade if normalization logic grows.
- Requires careful ordering: normalization must happen before any strategy is created.

### Option B: Create New Components (Dedicated Normalizer + Canonical Config Model)

What this would likely add:

- New module for canonical config normalization (e.g. `scenario_normalizer.py`) that produces a stable canonical dict or a typed model.
- New public API to retrieve the canonical config for debugging/testing.

Benefits:

- Clear separation of responsibility; easier to test normalization in isolation.
- Makes it harder for future code to "skip" normalization.

Trade-offs / Risks:

- Requires defining a canonical schema and migration policy up-front.
- More integration points: all execution paths must be updated to consume the canonical representation.

### Option C: Hybrid (Introduce Canonical Normalizer + Incrementally Route Execution Paths)

Phased approach:

- Phase 1: introduce canonical normalization + ensure strategies can be constructed from canonical init config in both paths.
- Phase 2: remove or reduce ad-hoc universe/timing resolution logic where the canonical contract makes it redundant.
- Phase 3: add cross-path equivalence tests and migrate built-in scenarios as needed.

Benefits:

- Reduces risk by allowing incremental convergence.
- Enables early validation of the canonical contract via a small number of representative scenarios.

Trade-offs / Risks:

- Requires discipline to avoid running "half old / half new" semantics for too long.

## 5. Implementation Complexity & Risk

- Effort: L (1-2 weeks)
  - Reason: touches configuration loading, strategy instantiation, and multiple runtime paths; requires new tests and careful backwards compatibility.
- Risk: Medium-High
  - Reason: cross-cutting behavior changes can silently affect results; strong regression tests are required to avoid research result drift.

## 6. Research Needed (Carry Into Design Phase)

- Canonical schema definition:
  - Which keys live at scenario top-level vs inside the strategy init config.
  - How to represent timing/universe/sizer/risk config consistently.

- Conflict resolution policy:
  - e.g. `rebalance_frequency` vs `timing_config.rebalance_frequency`, and what happens when both are present.

- Backward compatibility boundaries:
  - Identify which legacy shapes will still normalize, and which will become validation errors.

- Test strategy:
  - Decide the minimal set of representative scenarios for cross-path equivalence tests (one portfolio strategy, one signal strategy, one meta strategy, and one dynamic-universe scenario).

## Recommendations for Design Phase

- Prefer a dedicated normalization component (Option B or C) to keep responsibilities explicit and testable.
- Treat "strategy init config" as a first-class contract and make it identical across all instantiation paths.
- Plan for a migration layer that is explicit and produces warnings/errors rather than silently changing semantics.
