# Research validation vs SPY intramonth seasonal (notes)

## Commands

```bash
./.venv/Scripts/python.exe -m src.portfolio_backtester.backtester \
  --mode research_validate \
  --protocol double_oos_wfo \
  --scenario-filename <scenario.yaml>
```

Optional: `--research-artifact-base-dir`, `--force-new-research-run`, `--research-skip-unseen`, `--mdmp-cache-only`.

Artifacts default layout: `data/reports/<scenario>/research_protocol/<run_id>/`.

Authoritative YAML semantics: `docs/research_validation.md`.

## Features available in `research_validate` (double_oos_wfo)

- WFO **architecture grid** on `global_train_period` (train/test window months, step, rolling/expanding); inner **strategy** search uses scenario `optimize` + orchestrator.
- **Selection**: `constraints`, `selection.metric` (default RobustComposite), optional `scoring.composite_rank`, **robust_selection** (neighbor-smoothed `robust_score`).
- **Lock** + **registry**; **unseen** holdout on `unseen_test_period` (`final_unseen_mode`: reoptimize vs fixed params).
- **cross_validation**: blocked folds inside global train, aggregate per architecture (`cross_validation_summary.yaml`).
- **cost_sensitivity** (post-unseen only): Cartesian `slippage_bps_grid` x `commission_multiplier_grid`, `run_on: unseen` only.
- **bootstrap** (post-unseen, post cost_sensitivity if enabled; does not affect ranking): `random_wfo_architecture`, `block_shuffled_returns`, `block_shuffled_positions`, `random_strategy_parameters` (+ optional `persist_distribution_samples` / plot toggles under `reporting`).
- **execution**: `max_grid_cells` cap, `fail_fast`, `max_parallel_grid_workers`, `resume_partial` (incompatible with cross_validation).

## Not part of `research_validate`

- **Monte Carlo** (`--mode monte_carlo`, `monte_carlo_config`, Stage 2 robustness plots in optimization reporting): separate subsystem; no hook in `src/portfolio_backtester/research/` for MC.
- **`universe_search` / subset meta-search**: `scripts/universe_search.py` + `universe_search.py` module; not composed with the research protocol in one CLI run.
- **Synthetic return batch** (`numba_optimized.generate_synthetic_returns_batch`): Monte Carlo / robustness tooling, not the research protocol.

## Gaps for SeasonalSignalStrategy + entry_day_by_month / hold_days_by_month / subset search

1. **Inner search space**: Protocol assumes Optuna/scenario **`optimize`** lists and WFO on global train. Full per-month dicts (`entry_day_by_month`, `hold_days_by_month`) are not first-class Optuna variables; dev workflows use custom outer grids / subset enumeration.
2. **Bootstrap `random_strategy_parameters`**: Parameter space is `extras.strategy_params_space` or `strategy_params.strategy_params_space` as **flat** `str -> list` maps; sampling picks one value per key independently (`bootstrap.py`). Nested month-key dicts are **not** supported unless you explode into many flat keys (e.g. per-month knobs).
3. **Compute**: Each architecture cell runs full WFO optimization on train (x CV folds if enabled). Pairing that with large discrete grids or subset meta-search multiplies cost; `max_grid_cells` caps architecture combinations only, not inner trial counts.
4. **Scenarios like `is_wfo: false` optimize**: Research still applies its own WFO on `global_train_period`; align dates, `timing_config`, and costs with how you ran seasonal backtests.

## Suggested execution plan

1. Fix **month subset** and calendar parameterization offline (or narrow to one verification scenario) so `strategy_params` + small `optimize` block match production intent.
2. Author `research_protocol` from `config/scenarios/examples/research/double_oos_wfo_robust_composite_minimal.yaml`: periods, coarse `wfo_window_grid`, `constraints`, `robust_selection`, `execution.max_grid_cells`.
3. Run `research_validate`; review `wfo_architecture_grid.csv` and unseen metrics.
4. Enable **cost_sensitivity** on unseen for slippage/commission stress; then **bootstrap** (e.g. block shuffled returns/positions; add `random_strategy_parameters` only with a flat `strategy_params_space`).
5. Treat **Monte Carlo Stage 2** and **universe_search** as separate optional studies on finalists if needed.
