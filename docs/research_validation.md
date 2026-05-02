# Research validation protocol (methodology)

This guide describes the **double out-of-sample walk-forward** (`double_oos_wfo`) layer invoked with `--mode research_validate`. It is scenario-driven via a top-level `research_protocol` YAML block. Behavior and filenames match the current implementation under `src/portfolio_backtester/research/`.

## What the run does

1. **In-sample (global train)**: For each unique cell in the WFO architecture grid, the engine runs walk-forward optimization on `global_train_period`, producing aggregate metrics and best parameters for that architecture.
2. **Selection**: Rows may be filtered by `constraints`, then ranked. With `selection.metric: RobustComposite` (or omitted metric, which defaults to `RobustComposite`), ranking uses weighted ranks over multiple metrics (`scoring.type: composite_rank`). With any other metric, ranking uses that single metric only (omit `scoring`).
3. **Lock**: The winning cell is written to `protocol_lock.yaml` together with hashes of the scenario, global config, and normalized protocol payload.
4. **Unseen holdout**: On `unseen_test_period`, the run performs final validation according to `final_unseen_mode` (see below).
5. **Post hoc analyses** (optional, **never** change step 2): `cost_sensitivity` after unseen, then `bootstrap` if enabled.

CLI entry (protocol defaults to `double_oos_wfo`):

```text
python -m src.portfolio_backtester.backtester --mode research_validate --scenario-filename <path>
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--research-artifact-base-dir <path>` | Artifact root (default: `data/reports/`). Layout stays `<path>/<safe_scenario>/research_protocol/<run_id>/`. |
| `--research-skip-unseen` | Writes grid, selection, and lock; skips unseen validation and post-unseen steps that depend on it. |
| `--force-new-research-run` | Allows lock overwrite (ignores `lock.refuse_overwrite` for that run) and relaxes duplicate-lock checks in `registry.yaml`. |

## Choosing train vs unseen periods

- **`global_train_period`**: Calendar-inclusive range (`start_date` / `end_date`, or legacy `start` / `end`). All architecture comparisons and parameter searches for selection happen here (or inside its CV folds if `cross_validation.enabled`).
- **`unseen_test_period`**: Must **not** overlap the global train range (inclusive overlap is rejected). This is the single forward holdout used to mimic true out-of-sample behavior.
- **Practice**: Hold out a contiguous regime you did not design against; keep train long enough for multiple WFO windows. Very short unseen spans increase variance; very long ones dilute focus but improve stability of averages.

Aliases accepted by the parser: `global_train` for the train block; `unseen_holdout` for unseen.

## WFO architecture grid

Block key: `wfo_window_grid` (legacy: `wfo_grid`). Lists are Cartesian-expanded then deduplicated.

| Key | Meaning |
|-----|---------|
| `train_window_months` | Train window length(s). Legacy: `train_months`. |
| `test_window_months` | Test window length(s). Legacy: `test_months`. |
| `wfo_step_months` | Step between window starts. Legacy: `step_months`. |
| `walk_forward_type` | `rolling` or `expanding`. Legacy: `walk_forward_types`. |

**`execution.max_grid_cells`**: After expansion, if the number of unique cells exceeds this cap, the run raises `ResearchProtocolConfigError` before work starts (default `100`).

**Choosing a grid**: Start coarse (few train/test/step combinations) to estimate sensitivity, then refine. Extremely fine grids multiply multiple testing and compute cost; prefer interpretable neighborhoods rather than exhaustive large meshes.

## RobustComposite, scoring, and constraints

### RobustComposite

- Configure with `selection.metric: RobustComposite` or **omit** `metric` (empty defaults to `RobustComposite`).
- Optional `scoring` must use `type: composite_rank`, with `weights` (positive, normalized internally) and optional `directions` per weighted metric (`higher` or `lower`). Default direction is `higher` for all metrics except `Turnover` (`lower`).
- Allowed metric labels match optimizer display names and aliases (see `canonical_metric_display_key` usage in `src/portfolio_backtester/research/scoring.py`), e.g. `Calmar`, `Max Drawdown`, `total_return`, etc.

### Single-metric selection

Set `selection.metric` to one supported metric and **omit** `scoring`.

### Constraints

`constraints` is a list of rules: each needs `metric`, plus at least one of `min_value`, `max_value` (inclusive bounds). A row must satisfy **all** rules to be eligible. If no row passes, the run raises `ResearchConstraintError`.

Interpretation tips:

- For **Max Drawdown** stored as negative fractions, a **less negative** drawdown is a **larger** number (`-0.20` > `-0.35`). `min_value: -0.35` means "drawdown must not be worse than -35%".
- Failed rules are recorded per row in `wfo_architecture_grid.csv` (`constraint_passed`, `constraint_failures`).

### Robust selection (neighbor smoothing)

When `robust_selection.enabled: true`, ranking uses `robust_score`, a weighted mix of the cell score, neighbor median, and neighbor minimum on the WFO grid (same `walk_forward_type` and `wfo_step_months`; neighbors on sorted train/test axes). Only constraint-passing cells participate as neighbors. Weights default to `cell: 0.5`, `neighbor_median: 0.3`, `neighbor_min: 0.2` when unspecified.

When disabled, `robust_score` in artifacts still mirrors the raw cell score for convenience.

## Final unseen modes

| `final_unseen_mode` | Behavior |
|---------------------|----------|
| `reoptimize_with_locked_architecture` | Re-run WFO on the unseen period using the locked WFO architecture sizes and step. |
| `fixed_selected_params` | Single backtest on unseen using `selected_parameters` merged into strategy params. |

## Overfitting and multiple testing

This protocol **reduces** but does not remove overfitting:

- You still search **strategy parameters** inside each architecture cell on the global train period.
- You still search **over architecture cells** (and optional CV folds average metrics).
- The unseen period is **one** path; strong unseen performance can still be luck, especially with correlated assets or overlapping signals.

Mitigations baked into the tool: explicit holdout, optional constraints, composite ranks, robust neighbor scores, optional temporal CV on the train period, cost stress tests, and bootstrap summaries. Methodological discipline (simple grids, pre-registration of constraints, reporting all runs) remains your responsibility.

## Artifacts and how to read them

Default directory: `data/reports/<scenario>/research_protocol/<run_id>/` (scenario name sanitized for the filesystem).

| File | Contents |
|------|----------|
| `wfo_architecture_grid.csv` | One row per architecture: window fields, `score`, `robust_score`, `n_evaluations`, constraint flags, `best_parameters_json`, metric columns. |
| `selected_protocols.yaml` | Ranked picks with architectures, scores, parameters, metrics. |
| `protocol_lock.yaml` | `protocol_version`, hashes (`scenario_hash`, `global_config_hash`, `protocol_config_hash`), timestamps, winning architecture and `selected_parameters`. |
| `unseen_test_returns.csv` | Daily unseen portfolio returns (when unseen runs). |
| `unseen_test_metrics.yaml` | Aggregated unseen metrics and `mode`. |
| `research_validation_report.md` | Narrative summary when `reporting.enabled` is true. |
| `research_validation_report.html` | Optional when `reporting.generate_html: true`. |
| `research_execution_manifest.yaml` | Written for fresh grid runs: scenario/protocol hashes and full architecture list (supports resume). |
| `grid_architecture_snapshots/` | Per-cell YAML checkpoints when using resume (see below). With `cross_validation.enabled`, checkpoints are nested under `grid_architecture_snapshots/fold_<n>/`. |
| `cross_validation_summary.yaml` | When `cross_validation.enabled`: fold boundaries and metadata. |
| `cost_sensitivity.csv` / `cost_sensitivity_summary.yaml` | Slippage/commission grid on unseen when enabled. |
| `bootstrap_significance.csv` / `bootstrap_summary.yaml` | Bootstrap summaries when enabled. |
| `bootstrap_distribution_<slug>.csv` | Optional per-test distributions when `bootstrap.persist_distribution_samples: true`. |
| `wfo_heatmap_*.png` | When `reporting.generate_heatmaps: true`; metrics driven by `reporting.heatmap_metrics` (`score`, `robust_score`, or metric display names). |

Per-scenario append-only tracking: `<artifact_root>/<scenario>/research_protocol/registry.yaml` records lock paths and hashes to block accidental duplicate completed runs (unless `--force-new-research-run`).

## Reporting options (`research_protocol.reporting`)

- `enabled`: Master toggle for markdown (and related) reporting.
- `generate_heatmaps`, `heatmap_metrics`: PNG heatmaps per metric and WFO subgroup.
- `generate_html`, `html_embed_figures`, `html_navigation`: HTML report behavior.
- `generate_bootstrap_distribution_plots`: Requires `bootstrap.enabled: true`.
- `generate_cost_sensitivity_figure`: Requires `cost_sensitivity.enabled: true`.

## Cost sensitivity

```yaml
cost_sensitivity:
  enabled: true
  slippage_bps_grid: [...]
  commission_multiplier_grid: [...]
  run_on: unseen   # only value supported
```

Runs **after** unseen validation; does not affect ranking. Outputs CSV + YAML summary for stress-testing execution assumptions on the holdout path.

## Bootstrap (post-selection only)

Enable under `research_protocol.bootstrap`. Parsed defaults when the block is omitted: `enabled: false`, `n_samples: 200`, `random_seed: 42`, sub-blocks off, `persist_distribution_samples: false`.

Sub-features (each toggled with `enabled`):

- `random_wfo_architecture`: Compare against random architecture draws (in-sample stress).
- `block_shuffled_returns`: Block bootstrap of unseen returns (`block_size_days`, default 20).
- `block_shuffled_positions`: Block bootstrap of position path from trades (`block_size_days`, default 20).
- `random_strategy_parameters`: Random discrete draws from parameter space (`sample_size`, default 100).

Bootstrap runs **after** unseen and **after** cost sensitivity if that ran. If unseen is skipped, bootstrap is skipped.

## Cross-validation (`cross_validation`)

```yaml
cross_validation:
  enabled: true
  n_folds: 2
  strategy: blocked_global_train_period
```

Splits `global_train_period` into consecutive blocked business-day folds, evaluates the **same** expanded architecture grid on each fold, then averages metrics per architecture across folds (metric keys must appear in **every** fold to be averaged together). Constraint failures propagate: if any fold fails constraints for a cell, the aggregated row is marked failed. Failed slots aggregate **deterministic fold-wise means**: shared metric keys average across folds, `score` averages raw per-fold scores, numeric `best_parameters` keys intersecting all folds average, `n_evaluations` uses the arithmetic mean across folds (rounded on passing-only paths), and all constraint failure messages union.

After aggregation, composite scores (or single-metric scores) are recomputed from merged metrics before ranking **only when constraints pass everywhere** on that aggregated row (failed aggregates keep their averaged bookkeeping fields instead of overwriting with misleading last-fold values).

**Incompatible** with `execution.resume_partial`: enabling both raises `ResearchProtocolConfigError`.

## Execution phase: parallelism, resume, checkpoints

`execution` block:

- `max_grid_cells`: Hard cap before work starts (`ResearchProtocolConfigError` when exceeded).
- `fail_fast`: When parallel grid threads are enabled, any cell exception triggers cancellation of the remaining queued futures (`ThreadPoolExecutor.shutdown(..., cancel_futures=True)`), so stray work does not keep running silently.
- `max_parallel_grid_workers`: Upper bound on concurrent architecture cells (`1` disables parallelism). Effective parallelism uses **fresh `OptimizationOrchestrator` instances** via `optimization_orchestrator_factory` from `ResearchProtocolOrchestrator` (wired from the backtester facade with independent RNG draws). Requests for `max_parallel_grid_workers > 1` without that factory downgrade to serial execution while emitting a WARNING so a single shared orchestrator instance is never used concurrently.

`protocol_config_to_plain()` (and thus `protocol_config_hash`, resume manifests, and registry fingerprints) serializes **`max_parallel_grid_workers`**. Changing worker counts perturbs hashes the same way as other execution fields (`fail_fast`, `max_grid_cells`) even though deterministic math should eventually match serial results.

- `resume_partial`: Continue a previous run from `run_directory` using checkpoints and `research_execution_manifest.yaml`. Hashes and architecture order must match. Checkpoints live under `grid_architecture_snapshots/` (and `fold_<idx>/` when CV is active) inside that run directory. **`resume_partial` itself stays out of YAML hashing**, but mismatched manifests still fail loudly if you resume after tweaking other hashed fields (`max_parallel_grid_workers`, grid lists, composite weights, periods, …).

Resume and cross-validation cannot be used together.

## Interpreting results in practice

1. **Start with `wfo_architecture_grid.csv`**: Check `constraint_passed` rate. If most rows fail, constraints may be too tight or the strategy unstable under WFO.
2. **Compare `score` vs `robust_score`**: Large gaps when robust selection is on highlight cells that depend on neighbor context.
3. **Read `selected_protocols.yaml`**: Confirms parameters you would deploy under each ranked architecture.
4. **Treat `unseen_test_metrics.yaml` as one test**: Compare to in-sample metrics; large degradation is a red flag (though some degradation is expected).
5. **Use cost sensitivity and bootstrap** as **sanity checks** on fragility, not as p-values that "prove" alpha.

## Limitations

- **Single unseen window**: No automatic multi-regime or multi-asset-class holdout matrix.
- **Grid search cost**: Each cell runs a full WFO optimization on the train slice (or per CV fold).
- **Registry / lock semantics**: Designed to prevent silent duplication; use `--force-new-research-run` deliberately when you intend a new study lineage.
- **Cost and bootstrap** do not feed back into model selection; they are exploratory post hoc analyses.
- **Metric definitions** follow the same calculations as standard optimization reports; regime-dependent weaknesses of those metrics apply here as well.

## Small examples in-repo

Scenario snippets and tiny grids: `config/scenarios/examples/research/`.
