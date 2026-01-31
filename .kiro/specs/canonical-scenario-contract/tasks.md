# Implementation Plan

- [x] 1. Establish canonical scenario contract and normalization rules
- [x] 1.1 Define the canonical scenario configuration object (typed core + immutable pass-through extras)
  - Identify the full set of scenario inputs consumed by any runtime path (execution, evaluation, optimization, reporting, and data prefetch)
  - Define an explicit canonical schema for those inputs and an explicit container for preserved unknown keys
  - Ensure the canonical representation cannot be mutated during a run (including nested mappings)
  - Ensure canonical output is deterministic and can be compared in tests and debug logs
  - _Requirements: 1.2, 1.9, 1.10, 1.11_

- [x] 1.2 Implement canonicalization for timing and universe configuration
  - Normalize equivalent timing inputs into a single canonical timing configuration
  - Detect conflicts between equivalent timing keys and surface both keys and values in the error
  - Normalize universe definitions into a single canonical universe definition
  - Reject scenarios that provide multiple universe definitions with an actionable conflict error
  - _Requirements: 1.3, 1.6, 1.7, 1.8, 4.5_

- [x] 1.3 Implement strategy parameter normalization and legacy compatibility
  - Normalize strategy-qualified parameter names into canonical parameter names
  - Normalize legacy flat parameter shapes into `strategy_params` only when unambiguous
  - Reject ambiguous legacy parameter shapes with actionable migration guidance
  - Preserve unknown keys as immutable pass-through configuration without reinterpretation
  - Emit warnings when legacy normalization occurs
  - _Requirements: 1.11, 4.6, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 1.4 Normalize optimizer configuration consistently (optimize and optimizers)
  - Normalize optimizer settings into a canonical optimizer configuration with an explicit selection rule
  - If multiple optimizers are provided, select the configured default (prefer optuna if present) and warn about ignored entries
  - Ensure optimizer-derived flattened settings do not silently override explicitly configured top-level settings
  - Validate optimizer configuration structure early and fail fast on invalid shapes
  - _Requirements: 1.3, 1.4, 4.6, 5.1, 5.3_

- [x] 1.5 Apply defaulting rules consistently (global defaults and strategy defaults)
  - Apply run-wide defaults from global configuration when optional scenario inputs are omitted
  - Apply strategy defaults for missing strategy-owned parameters without changing the intended meaning
  - Ensure defaults are applied deterministically and are visible in the canonical scenario representation
  - _Requirements: 1.4, 1.10, 5.3_

- [x] 2. Integrate canonicalization into the runtime and enforce canonical-only configuration propagation
- [x] 2.1 Route all scenario entry points through the normalization boundary
  - Ensure scenarios are canonicalized before any data fetching, strategy instantiation, or evaluation begins
  - Ensure canonicalization is performed for both backtest and optimize/WFO execution modes
  - Ensure canonicalization is performed for both file-based scenarios and programmatic scenario inputs
  - _Requirements: 1.1, 2.6_

- [x] 2.2 Provide canonical scenario inspection for debugging and tests
  - Provide a structured way to access the canonical scenario configuration for a given scenario name
  - Ensure canonical configuration is easy to compare and assert in automated tests
  - Log canonicalization outcomes at debug level to support drift investigations
  - _Requirements: 1.5, 1.9_

- [x] 2.3 Prevent raw, unnormalized scenario dictionaries from leaking into runtime components
  - Ensure runtime components receive canonical configuration (and immutable pass-through extras) rather than raw scenario dicts
  - Add guardrails that detect accidental bypass of canonicalization in critical runtime paths
  - Remove or replace any mid-run mutation of scenario configuration that affects interpretation
  - _Requirements: 2.7, 2.6_

- [x] 2.4 Ensure validation failures and warnings are surfaced before execution begins
  - Halt before execution on blocking normalization conflicts or semantic validation failures
  - Preserve warnings for normalizable legacy shapes while continuing execution
  - _Requirements: 4.1, 4.5, 4.6_

- [x] 3. Unify strategy initialization contract across all strategy instantiation paths
- [x] 3.1 Build a single strategy constructor input contract from the canonical scenario configuration
  - Ensure timing configuration and provider-relevant configuration are present at construction time
  - Ensure strategy-owned parameters are passed consistently regardless of runtime path
  - Ensure WFO window bounds and evaluation context are available to strategies that require them
  - _Requirements: 3.1, 3.2, 3.3, 3.7_

- [x] 3.2 (P) Update backtest-mode strategy instantiation to use the canonical strategy init contract
  - Instantiate strategies using the same effective configuration contract used in optimization/evaluation paths
  - Remove reliance on mutating scenario configuration mid-run to inject universe or timing context
  - Validate that timing, universe, and risk settings match between backtest and optimize paths for the same scenario
  - _Requirements: 2.1, 2.3, 2.4, 2.6, 3.5, 3.6_

- [x] 3.3 (P) Update evaluation/optimization strategy instantiation to use the canonical strategy init contract
  - Remove reliance on post-initialization config attachment for correctness
  - Preserve behavior for strategies that previously depended on attached scenario config by providing equivalent inputs at init time
  - Ensure strategies receive the same effective configuration contract regardless of instantiation path
  - _Requirements: 2.1, 3.4, 3.5, 3.6_

- [x] 3.4 Validate provider and timing initialization behavior under the canonical init contract
  - Confirm timing controllers initialize consistently for representative strategy types
  - Confirm providers (universe, sizing, stop loss, take profit, risk-off) initialize consistently across runtime paths
  - _Requirements: 2.3, 2.4, 3.2, 3.3, 3.7_

- [x] 4. Align universe resolution and data prefetch to the canonical scenario configuration
- [x] 4.1 (P) Derive required ticker collection from the canonical universe definition
  - Ensure the data prefetch step collects tickers from canonical universe configuration, benchmark configuration, and any non-universe data requirements
  - Ensure the same ticker set is derived for the same scenario regardless of runtime mode
  - Avoid relying on raw scenario dictionaries or mid-run scenario mutation to determine tickers
  - _Requirements: 2.2, 2.5, 2.6_

- [x] 4.2 Ensure dynamic-universe scenarios behave consistently across modes and windows
  - Ensure method-based universe definitions are evaluated consistently for the same date/window context
  - Ensure universe resolution remains deterministic when the scenario and data are unchanged
  - _Requirements: 2.2, 2.6, 6.4_

- [x] 4.3 Ensure benchmark and non-universe data requirements respect canonical precedence
  - Ensure scenario benchmark overrides global benchmark when configured
  - Ensure non-universe data requirements are collected without leaking raw scenario configuration
  - _Requirements: 2.5, 2.7_

- [x] 5. Implement standardized normalization errors and consistent validation behavior
- [x] 5.1 Implement standardized normalization errors for conflicts and unsafe migrations
  - Include conflicting keys and values in conflict errors
  - Provide actionable migration guidance when normalization cannot proceed safely
  - Ensure conflict detection fails fast before execution begins
  - _Requirements: 1.3, 1.7, 1.8, 4.5, 5.2_

- [x] 5.2 Ensure semantic validation failures are enforced consistently at the normalization boundary
  - Fail early for unknown strategies
  - Fail early for invalid optimization parameters
  - Fail early for unsupported configuration combinations (e.g., strategy-type specific constraints)
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 5.3 Emit warnings for normalizable legacy shapes and selection decisions
  - Warn when legacy shapes are normalized (flat params, prefixed params)
  - Warn when optimizer selection behavior chooses a subset of provided optimizer configs
  - Ensure warnings describe the normalization performed to avoid silent meaning changes
  - _Requirements: 4.6, 5.1, 5.3_

- [x] 6. Add automated tests to prevent drift and ensure cross-path equivalence
- [x] 6.1 Add unit tests for canonicalization mapping rules and conflict detection
  - Cover timing canonicalization and timing conflicts
  - Cover universe canonicalization and universe conflicts
  - Cover optimizer selection and optimizer configuration validation
  - _Requirements: 1.2, 1.3, 1.6, 1.7, 1.8, 6.1, 6.3_

- [x] 6.2 Add unit tests for legacy compatibility normalization and pass-through extras
  - Cover strategy parameter prefix stripping and flat parameter normalization
  - Verify warnings are emitted for normalizable legacy shapes
  - Verify unknown keys are preserved immutably and are not reinterpreted
  - _Requirements: 1.11, 4.6, 5.1, 5.4, 5.5, 6.1, 6.3_

- [x] 6.3 Add integration test ensuring consistent semantics across backtest and optimization evaluation paths
  - Execute the same scenario through backtest and through the evaluation path used during optimization
  - Compare effective configuration and key outputs to ensure consistent interpretation
  - _Requirements: 2.1, 2.6, 6.2_

- [x] 6.4 (P) Add dynamic-universe regression test
  - Validate that a dynamic-universe scenario resolves the same universe across runtime modes and windows
  - _Requirements: 2.2, 6.4_

- [x] 6.5 (P) Add strategy initialization contract equivalence regression test
  - Instantiate a representative strategy through both major instantiation paths and compare effective init configuration
  - _Requirements: 3.5, 6.5_

- [x] 7. Verification and cleanup
- [x] 7.1 Run targeted QA checks and fix issues uncovered by type checking and linting
  - Ensure new and modified code follows project formatting, linting, and typing standards
  - _Requirements: 6.3_

- [x] 7.2 Run the full test suite and fix any regressions or drift discovered
  - Ensure cross-path equivalence tests are stable and provide actionable failure output
  - _Requirements: 6.2, 6.4, 6.5_
