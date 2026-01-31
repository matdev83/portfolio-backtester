# Requirements Document

## Project Description (Input)
Unify the ScenarioConfig -> StrategyConfig runtime contract so the portfolio-backtester interprets scenario YAML consistently across all execution paths (backtest, optimize/WFO, evaluation, and reporting).

The goal is to introduce a canonical, normalized configuration shape that:

- preserves backward compatibility for existing scenarios where feasible,
- removes ambiguity about where configuration lives (top-level vs strategy_params vs attached strategy.config), and
- ensures universe selection, timing, risk management, and other cross-cutting settings behave the same regardless of which backtesting/optimization path is used.

This change should reduce config drift bugs (a key exists but is ignored), make future feature work predictable (one place to add config + validation), and improve the reliability of optimization results by guaranteeing the same semantics as normal backtests.

## Introduction
This specification defines requirements for a single, canonical scenario configuration contract that is applied consistently throughout Portfolio Backtester.

The requirements prioritize:

- determinism and reproducibility (config input -> canonical config output),
- consistent interpretation across all runtime paths (backtest/optimize/WFO/evaluation/reporting), and
- clear validation and migration behavior for existing scenario files.

The gap analysis identified that the current codebase has strong semantic validation, but multiple runtime paths still interpret scenario keys differently (especially universe selection, timing configuration, and strategy initialization context). These requirements explicitly address those divergence points.

## Requirements

### Requirement 1: Canonical Scenario Normalization
**Objective:** As a strategy developer, I want a single canonical scenario configuration representation, so that configuration meaning is deterministic and reusable across subsystems.

#### Acceptance Criteria
- 1.1 When a scenario configuration is loaded (from a YAML file or provided programmatically), the Portfolio Backtester shall normalize it into a canonical scenario configuration representation before any backtest, evaluation, or optimization begins.
- 1.2 The Portfolio Backtester shall normalize scenario configurations deterministically such that identical inputs produce identical canonical outputs.
- 1.3 If a scenario configuration contains ambiguous or conflicting inputs, the Portfolio Backtester shall fail validation and report which keys or sections conflict.
- 1.4 The Portfolio Backtester shall apply defaults during normalization such that missing optional settings resolve to predictable values derived from global configuration and/or strategy defaults.
- 1.5 The Portfolio Backtester shall make the canonical scenario configuration available for inspection for debugging and testing.
- 1.6 When a scenario provides timing configuration via multiple equivalent fields (for example `rebalance_frequency` and `timing_config.rebalance_frequency`), the Portfolio Backtester shall normalize to a single canonical representation.
- 1.7 If a scenario provides both `rebalance_frequency` and `timing_config.rebalance_frequency` and they are not equivalent, the Portfolio Backtester shall fail validation and report both conflicting values.
- 1.8 If a scenario provides multiple universe definitions (for example `universe` and `universe_config`), the Portfolio Backtester shall fail validation and report the conflicting configuration.
- 1.9 The Portfolio Backtester shall treat the canonical scenario configuration as immutable for the duration of a run.
- 1.10 The Portfolio Backtester shall define a canonical schema that includes all scenario keys that are consumed by any runtime path (backtest, optimize/WFO, evaluation, reporting, and data prefetch).
- 1.11 Where the scenario contains additional keys that are not part of the canonical schema, the Portfolio Backtester shall preserve them as immutable pass-through configuration and shall not silently reinterpret them.

### Requirement 2: Consistent Semantics Across Execution Modes
**Objective:** As a researcher, I want the same scenario to behave the same across backtest and optimize modes, so that optimization results are comparable to standalone backtests.

#### Acceptance Criteria
- 2.1 When the same canonical scenario configuration and the same input market data are used, the Portfolio Backtester shall produce consistent backtest semantics regardless of runtime mode (backtest vs optimize), except for differences explicitly related to optimization parameter selection.
- 2.2 When the scenario defines universe selection, the Portfolio Backtester shall resolve the universe consistently for all execution modes.
- 2.3 When the scenario defines timing behavior (rebalance rules), the Portfolio Backtester shall apply timing behavior consistently for all execution modes.
- 2.4 When the scenario defines risk management settings (stop loss, take profit, or risk-off signals), the Portfolio Backtester shall apply those settings consistently for all execution modes.
- 2.5 When the Portfolio Backtester collects required tickers for data fetching, it shall use the canonical scenario configuration as the source of truth for universe and non-universe data requirements.
- 2.6 When the same scenario is executed through different runtime paths used by the application (including backtest mode and the evaluation path used during optimization), the Portfolio Backtester shall not change the interpretation of scenario keys.
- 2.7 When scenario configuration is provided to runtime components, the Portfolio Backtester shall provide only the canonical scenario configuration (and preserved immutable pass-through configuration, if any) and shall not provide raw, unnormalized scenario dicts.

### Requirement 3: Single Strategy Initialization Contract
**Objective:** As a strategy author, I want strategies to receive all relevant configuration through a single consistent contract, so that strategy behavior does not depend on which pipeline instantiated it.

#### Acceptance Criteria
- 3.1 When a strategy instance is created for a run, the Portfolio Backtester shall provide a single, consistent configuration contract that includes all strategy-owned parameters and cross-cutting settings relevant to strategy behavior.
- 3.2 When a scenario provides timing configuration, the Portfolio Backtester shall ensure that timing configuration is available to the strategy at the time the strategy initializes its timing controller.
- 3.3 When a scenario provides provider-relevant configuration (universe, sizing, stop loss, take profit, risk-off), the Portfolio Backtester shall ensure that those settings are available to the strategy at the time providers are initialized.
- 3.4 If an execution path currently relies on attaching additional configuration onto an already-instantiated strategy, the Portfolio Backtester shall preserve correct behavior by ensuring the effective configuration is consistent with the canonical scenario configuration.
- 3.5 When the same strategy is instantiated for the same scenario and parameters in different runtime paths, the Portfolio Backtester shall provide the same effective configuration contract to the strategy.
- 3.6 The Portfolio Backtester shall not require strategies to depend on configuration that is attached only after strategy initialization.
- 3.7 When a strategy is instantiated, the Portfolio Backtester shall ensure that any configuration used to initialize timing controllers and providers is present in the constructor input.

### Requirement 4: Validation and Error Reporting
**Objective:** As a user, I want configuration errors to be detected early and explained clearly, so that I can fix scenarios quickly and trust the run results.

#### Acceptance Criteria
- 4.1 When a scenario file is invalid (syntax or semantic validation failure), the Portfolio Backtester shall halt before starting the run and report actionable validation errors.
- 4.2 If a scenario references an unknown strategy, the Portfolio Backtester shall report an error that includes the unknown strategy identifier.
- 4.3 If a scenario references an optimization parameter that is not a strategy tunable and not a shared optimizer default, the Portfolio Backtester shall report an error identifying the invalid parameter.
- 4.4 If a scenario uses keys that are accepted only for some strategy types (for example, meta strategy universe constraints), the Portfolio Backtester shall report an error describing the unsupported configuration.
- 4.5 If validation fails due to conflicting or ambiguous configuration, the Portfolio Backtester shall report the specific conflicting keys and their values.
- 4.6 Where a scenario uses deprecated keys or legacy configuration shapes that are still normalizable, the Portfolio Backtester shall emit a warning describing the normalization performed.

### Requirement 5: Backward Compatibility and Migration Safety
**Objective:** As an existing project user, I want current scenarios to keep working or fail with a clear migration path, so that I can adopt the canonical contract safely.

#### Acceptance Criteria
- 5.1 Where an existing scenario uses legacy or alternate configuration shapes that are still supported, the Portfolio Backtester shall normalize them into the canonical scenario configuration without changing the intended meaning.
- 5.2 If an existing scenario uses deprecated keys or patterns that cannot be normalized safely, the Portfolio Backtester shall fail validation and provide a message that indicates the required migration action.
- 5.3 The Portfolio Backtester shall not silently change the meaning of configuration during normalization.
- 5.4 Where a scenario provides strategy parameters using a strategy-qualified prefix (for example `<strategy>.param`), the Portfolio Backtester shall normalize those parameters to the canonical parameter name.
- 5.5 Where a scenario provides strategy parameters in a legacy flat shape (not nested under `strategy_params`), the Portfolio Backtester shall normalize them into the canonical parameter structure if it can do so without ambiguity.

### Requirement 6: Regression Safety
**Objective:** As a maintainer, I want the canonical configuration contract to be protected by tests, so that future changes do not reintroduce divergent behavior.

#### Acceptance Criteria
- 6.1 The Portfolio Backtester shall include automated tests that verify scenario normalization behavior for representative built-in scenario files.
- 6.2 When a scenario is executed through different runtime paths that currently exist in the codebase, the Portfolio Backtester shall include tests that ensure the canonical scenario configuration is interpreted consistently across those paths.
- 6.3 If a future change modifies canonical scenario normalization rules, the Portfolio Backtester shall require updating the tests and documentation to reflect the new behavior.
- 6.4 The Portfolio Backtester shall include at least one regression test that covers a dynamic-universe scenario and verifies consistent normalization and execution semantics.
- 6.5 The Portfolio Backtester shall include at least one regression test that exercises strategy instantiation through both major runtime paths and verifies that the effective strategy initialization configuration is identical.
