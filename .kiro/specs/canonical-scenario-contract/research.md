# Research & Design Decisions

## Summary

- **Feature**: `canonical-scenario-contract`
- **Discovery Scope**: Extension
- **Key Findings**:
  - Scenario semantic validation is strong, but there is no single runtime normalization step that produces a canonical scenario configuration consumed uniformly across execution paths.
  - Strategy initialization differs by path: one path instantiates strategies with params-only, another attaches scenario config after construction, which is too late for timing/provider initialization.
  - Universe and timing are interpreted in multiple places (ticker collectors, providers, and ad-hoc overrides), which creates drift risk when the same scenario is executed via different pipelines.

## Research Log

### Where does scenario config get interpreted today
- **Context**: Requirements demand consistent semantics across backtest and optimize/WFO paths.
- **Sources Consulted**:
  - `src/portfolio_backtester/config_loader.py`
  - `src/portfolio_backtester/scenario_validator.py`
  - `src/portfolio_backtester/backtester_logic/backtester_facade.py`
  - `src/portfolio_backtester/backtester_logic/data_fetcher.py`
  - `src/portfolio_backtester/backtesting/strategy_backtester.py`
  - `src/portfolio_backtester/strategies/_core/base/base/base_strategy.py`
  - `.kiro/specs/canonical-scenario-contract/gap-analysis.md`
- **Findings**:
  - YAML load + semantic validation happens early, but multiple runtime layers still interpret the same keys independently.
  - `BaseStrategy` reads `timing_config` and initializes providers in `__init__`, so configuration must be present at construction time.
  - `StrategyBacktester` currently attaches `instance.config` after construction, which cannot influence initialization-time behavior.
- **Implications**:
  - Canonicalization must occur before any strategy is constructed.
  - The strategy constructor input must be a canonical, complete configuration contract.

### What creates the highest drift risk
- **Context**: Gap analysis highlighted inconsistent handling of universe/timing and mutation of scenario dicts.
- **Sources Consulted**:
  - `src/portfolio_backtester/backtester_logic/strategy_logic.py`
  - `src/portfolio_backtester/backtester_logic/data_fetcher.py`
  - `src/portfolio_backtester/backtesting/strategy_backtester.py`
- **Findings**:
  - Universe can be resolved via ticker collectors, via universe providers, and via ad-hoc fallback logic.
  - Timing configuration can be specified via multiple keys and may be synthesized if missing.
  - Some execution paths mutate `scenario_config` during a run.
- **Implications**:
  - Canonicalization must include conflict detection and a single source of truth for universe/timing.
  - Canonical config should be treated as immutable input for all runtime components.

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Extend existing loaders | Add normalization directly in `config_loader.py` | Minimal new surface area | Risk of bloating loader, harder isolated testing | Good for fast adoption but needs discipline |
| Dedicated normalizer | Add a `ScenarioNormalizer` service used by all flows | Clear boundary, testable, reusable | Requires integration work across paths | Best long-term consistency |
| Hybrid phased | Normalizer + incremental routing of call sites | Lowers risk, enables gradual convergence | Temporary dual semantics if not controlled | Recommended for brownfield refactor |

## Design Decisions

### Decision: Introduce a dedicated scenario normalization service
- **Context**: Multiple runtime paths interpret scenario keys; requirements demand canonical config used everywhere.
- **Alternatives Considered**:
  1. Normalize inline in each runtime path
  2. Normalize in `config_loader.py` only
  3. Dedicated `ScenarioNormalizer` invoked once per scenario
- **Selected Approach**: Create a dedicated normalization component that transforms a scenario input dict into an immutable canonical scenario config, and require all backtest/optimization/evaluation paths to use it.
- **Rationale**: Makes it difficult to bypass normalization and creates a single, testable location for conflict detection and default application.
- **Trade-offs**: Requires updating multiple call sites to consume the canonical contract.
- **Follow-up**: Verify that canonicalization runs before any strategy instantiation in all entry points.

### Decision: Make strategy initialization depend only on the canonical strategy init config
- **Context**: `BaseStrategy` initializes timing and providers in `__init__`, so post-init config attachment is unreliable.
- **Alternatives Considered**:
  1. Keep post-init `strategy.config` assignment and add more synchronization
  2. Pass full scenario dict to strategy constructors
  3. Pass a dedicated canonical strategy init config derived from the canonical scenario config
- **Selected Approach**: Provide strategies a canonical strategy init config at construction time; do not require post-init mutation for correctness.
- **Rationale**: Aligns with existing provider/timing initialization patterns and removes path-specific behavior.
- **Trade-offs**: Requires careful compatibility mapping from legacy scenario shapes.
- **Follow-up**: Identify strategies that currently depend on `self.config` and ensure they continue to work with the new contract.

## Risks & Mitigations

- Result drift risk (behavior changes for existing scenarios) - Mitigation: add cross-path equivalence tests on representative scenarios and treat changes as explicit migrations.
- Migration complexity for legacy scenario shapes - Mitigation: explicit normalization rules with conflict detection and actionable error messages.
- Partial adoption risk - Mitigation: introduce a single canonicalization API and route all execution paths through it as an explicit task sequence.

## References

- `.kiro/specs/canonical-scenario-contract/requirements.md` - canonical contract requirements
- `.kiro/specs/canonical-scenario-contract/gap-analysis.md` - identified integration hotspots
