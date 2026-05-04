# Error handling standards (portfolio-backtester)

This document complements [AGENTS.md](../AGENTS.md) with concrete rules for agents and contributors.

## Principles

1. **Fail fast** — Validate inputs at boundaries before expensive work.
2. **Specific exceptions** — Prefer `ValueError`, `TypeError`, `KeyError`, `ImportError`, and domain types over bare `Exception` in new code.
3. **Preserve context** — Use `raise DomainError(...) from exc` when translating failures.
4. **Broad `except Exception` only when necessary** — Optional dependencies, worker isolation, unknown third-party backends, or top-level facades. Always log with **`exc_info=True`** when not re-raising, and bind the exception: **`except Exception as exc:`**.
5. **No silent swallow** — Do not use `except Exception: pass` without a documented reason; prefer narrow catches or `logger.debug(..., exc_info=True)` (rate-limited or guarded if noisy).
6. **`# noqa: BLE001`** — Only where `except Exception` is intentional. In `src/portfolio_backtester/interfaces/` and `src/portfolio_backtester/research/`, every noqa must have a **one-line rationale** on the same line or the next line.

## Partial failures (optimization)

- Per-window evaluation failures are recorded on **`WindowResult.evaluation_error`** (short `ExcType: message` string).
- **`EvaluationResult.failure_reason`** summarizes window-level errors for the trial (joined when multiple windows fail).

## Optional CI: BLE001 on tightened trees

To flag blind `except Exception` without an inline noqa in boundary packages, run:

[`scripts/check_error_handling_lint.py`](../scripts/check_error_handling_lint.py) (wraps `ruff --select BLE001` on `interfaces/` and `research/` only).

Use after narrowing catches; legacy packages may still need noqa until refactored.

## Inventory (maintenance snapshot)

Regenerate with ripgrep when doing a cleanup pass:

- **Unbound `except Exception:`** (should use `as exc` and log or narrow):

  `donchian_asri_signal_strategy.py`, `optimization_converter.py`, `data_fetcher.py`,
  `performance_metrics.py`, `base_strategy.py`, `position_tracker.py`,
  `hybrid_parallelism.py`, `evaluator.py`, `hello_world_signal_strategy.py`,
  `strategy_backtester.py`, `sharpe_momentum_portfolio_strategy.py`,
  `signal_price_extractor_interface.py`, `trade_tracker.py`,
  `strategy_config_cross_validator.py`, `double_oos_wfo.py`, `atr_service.py`,
  `base_momentum_portfolio_strategy.py`, `metrics.py`, `ema_crossover_signal_strategy.py`,
  `seasonal_signal_strategy.py`, `numba_optimized.py`, `backtester_logic/optimization.py`,
  `sequential_orchestrator.py`, `scalar_extractor_interface.py`, `price_data_utils.py`,
  `monte_carlo/monte_carlo.py`, `monte_carlo/visual_inspection.py`, `table_generator.py`,
  `strategy_logic.py`, `parameter_generator.py`, `monte_carlo_stage2.py`,
  `testing/strategies/dummy_signal_strategy.py`, `prepared_arrays.py`,
  `universe_data/spy_holdings.py`, `monte_carlo_analyzer.py`, `signal_cache.py`.

- **`# noqa: BLE001` sites** (justify or narrow over time):

  `signal_cache.py`, `constraint_logic.py`, `reporting.py`, `strategy_logic.py`,
  `autogluon_sortino_ml_portfolio_strategy.py`, `evaluator.py`, `backtester_facade.py`,
  `double_oos_wfo.py`, `base_momentum_portfolio_strategy.py`, `monte_carlo_stage2.py`,
  `protocol_config.py`, `parameter_analysis.py`.
