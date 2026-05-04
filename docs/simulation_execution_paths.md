# Portfolio simulation execution paths

This document records **which code paths** produce strategy returns and costs, so changes stay semantically aligned across the backtester, optimizer, and tests.

## Canonical share/cash engine (standard strategies)

**Production entry points**

- [`StrategyBacktester`](../src/portfolio_backtester/backtesting/strategy_backtester.py) and [`backtest_runner`](../src/portfolio_backtester/backtester_logic/backtest_runner.py) call [`calculate_portfolio_returns`](../src/portfolio_backtester/backtester_logic/portfolio_logic.py).
- The optimization stack (`evaluation_engine`, `optimization/*`, `optuna_objective`) does **not** import Numba return kernels directly; it runs scenarios that ultimately use the same portfolio return pipeline as the backtester.

**Call chain**

1. `calculate_portfolio_returns` (non-meta) builds [`PortfolioSimulationInput`](../src/portfolio_backtester/backtester_logic/portfolio_simulation_input.py): dense targets, **`rebalance_mask`** (from sparse execution rows or an explicit override), close panel, and optional **open** panel when `trade_execution_timing=next_bar_open`.
2. [`simulate_portfolio`](../src/portfolio_backtester/simulation/kernel.py) calls [`canonical_portfolio_simulation_kernel`](../src/portfolio_backtester/numba_kernels.py), which uses **execution prices** for fills and **close** prices for mark-to-market, and sets **`daily_returns[0]`** from post-cost day-0 NAV.

**Invariant:** net-of-cost daily returns and positions for standard strategies come only from this chain unless explicitly documented otherwise.

**Return series labeling:** Standard-strategy outputs set ``Series.attrs["portfolio_backtester.execution_model"]`` to ``MetaExecutionMode.CANONICAL_SHARE_CASH_SIMULATION.value`` (``"canonical_share_cash_simulation"``) for the same reporting distinction as meta (see above).

**Execution ledger and trade tracking (standard only):** `simulate_portfolio` emits a canonical **`execution_ledger`** (`EXECUTION_LEDGER_COLUMNS`) aligned with the share/cash kernel. When `track_trades` is enabled, `calculate_portfolio_returns` builds a **`TradeTracker`** via **`TradeTracker.populate_from_execution_ledger`** on that ledger. Treat ledger row semantics (quantities, cash/position before/after, costs) as **canonical-path contracts** for standard strategies.

## Meta strategies (`MetaExecutionMode.TRADE_AGGREGATION`)

Meta strategies **do not** use `simulate_portfolio` / `canonical_portfolio_simulation_kernel`. They aggregate sub-strategy trades and rebuild a timeline; see [`meta_execution.py`](../src/portfolio_backtester/backtester_logic/meta_execution.py) and `_calculate_meta_strategy_portfolio_returns` in [`portfolio_logic.py`](../src/portfolio_backtester/backtester_logic/portfolio_logic.py).

### Current status: intentional alternate execution model

Until meta strategies emit an order or target stream consumable by the canonical share/cash kernel, **`TRADE_AGGREGATION` is a first-class, separate execution model**, not a temporary shim.

- **Do not assume** the same invariants as the canonical path (e.g. share-level rebalance masks, day-0 return accounting, or `next_bar_open` wiring) without dedicated meta tests or docs.
- **Do assume** optimizer and backtest flows branch on [`portfolio_execution_mode_for_strategy`](../src/portfolio_backtester/backtester_logic/meta_execution.py); standard scenarios keep using the canonical simulator only.
- **Future unification** would mean converting aggregated child trades into canonical inputs (or equivalent guarantees) and proving parity with integration tests—not deleting this path silently.

**Return series labeling:** ``calculate_portfolio_returns`` sets ``Series.attrs["portfolio_backtester.execution_model"]`` to ``MetaExecutionMode.TRADE_AGGREGATION.value`` (``"trade_aggregation"``) on meta outputs so downstream reporting can distinguish them from the canonical path without relying on strategy type alone. Downstream code that ``.copy()`` or re-serializes returns may drop ``attrs``—treat labeling as best-effort metadata.

**Implication:** parity tests and cost semantics that assert on the canonical kernel apply to **standard** strategies only unless meta execution is explicitly cross-tested or unified.

**Meta vs canonical execution ledger:** Meta strategies remain **`MetaExecutionMode.TRADE_AGGREGATION`**. They **do not** run `simulate_portfolio` and therefore **do not** produce the canonical **`execution_ledger`**. With `track_trades`, meta paths construct a **`TradeTracker`** from **aggregated sub-strategy trades** (`_create_meta_strategy_trade_tracker` in `portfolio_logic.py`), not from **`populate_from_execution_ledger`**. **Do not compare** meta trade rows or timing to canonical ledger invariants until/unless meta is canonicalized and covered by explicit parity tests.

## Legacy Numba kernels (not the production return path)

[`drifting_weights_returns_kernel`](../src/portfolio_backtester/numba_kernels.py) and [`detailed_commission_slippage_kernel`](../src/portfolio_backtester/numba_kernels.py) remain in the tree for **tests**, equivalence checks, and local verification scripts (e.g. `tests/verify_numba.py`, `tests/unit/test_optimization_path_equivalence.py`). **No module under `src/portfolio_backtester/optimization/` imports them.**

**Footgun:** new features must not reintroduce these kernels as the source of truth for live backtests or optimizer metrics while scenarios use the canonical simulator—behavior would diverge.

## Strategy target generation

Production full-scan generation requires [`generate_target_weights`](../src/portfolio_backtester/backtester_logic/strategy_logic.py). Legacy per-date `generate_signals` loops are isolated behind [`LegacyGenerateSignalsAdapter`](../src/portfolio_backtester/backtester_logic/strategy_logic.py) for tests and migration.

[`StrategyContext`](../src/portfolio_backtester/strategies/_core/target_generation.py) remains pandas-first for authoring compatibility. Optional **NumPy** views (``universe_close_np``, ``rebalance_session_mask_np``) exist for performance-sensitive ``generate_target_weights`` implementations that want contiguous arrays without re-implementing alignment.

## Audit summary (2026-05)

| Area | Uses canonical `simulate_portfolio`? |
|------|----------------------------------------|
| `backtesting/strategy_backtester.py` | Yes |
| `backtester_logic/backtest_runner.py` | Yes |
| `src/.../optimization/` (no `numba_kernels` drift/detailed imports) | Indirectly via scenario runs |
| Meta `TRADE_AGGREGATION` | No (trade ledger path) |
| Legacy drift/detailed kernels | Tests / scripts only |

## Legacy position-based trade inference (isolated)

The following are **not** the canonical portfolio return path and **must not** be wired back into ``calculate_portfolio_returns`` for standard strategies:

- [`trade_lifecycle_kernel`](../src/portfolio_backtester/numba_kernels.py): reconstructs round-trips from **daily positions** and a **single price matrix** (typically closes).
- [`TradeTracker.populate_from_kernel_results`](../src/portfolio_backtester/trading/trade_tracker.py): consumes that kernel’s structured output.
- [`trading/numba_trade_tracker.py`](../src/portfolio_backtester/trading/numba_trade_tracker.py): adapter for tests / ``track_trades_canonical``; **not** imported by production portfolio simulation.

Production standard trade rows must come from the **execution ledger** via ``populate_from_execution_ledger``.

## Open roadmap (methodology, performance, external parity)

Not yet first-class product requirements; track intentionally:

- **Stricter optimizer perf budgets:** tighten warm-run thresholds beyond hang guards in perf-marked tests.
- **Research protocol defaults:** richer anti-overfitting presets (nested WFO, cost stress, reporting) on top of existing ``research_validate`` mode.
- **External engine parity:** optional golden comparisons vs vectorbt / ``bt`` for simple fixed-weight and rebalance schedules to catch semantic drift.

Perf smoke (optional CI: `-m "not perf"`): [`test_seasonal_sim_perf_smoke.py`](../tests/unit/simulation/test_seasonal_sim_perf_smoke.py), [`test_optimizer_step_perf_smoke.py`](../tests/unit/simulation/test_optimizer_step_perf_smoke.py).
