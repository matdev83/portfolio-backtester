"""Markdown summaries for research protocol validation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

import yaml

from portfolio_backtester.research.protocol_config import (
    RESEARCH_PROTOCOL_ARTIFACT_VERSION,
    DoubleOOSWFOProtocolConfig,
)
from portfolio_backtester.research.results import ResearchProtocolResult


def resolve_cross_validation_summary(
    run_dir: Path | str,
    result: ResearchProtocolResult,
) -> dict[str, object] | None:
    """Load cross-validation summary from the result object or ``cross_validation_summary.yaml``."""

    root = Path(run_dir)
    yaml_path = root / "cross_validation_summary.yaml"
    blob = result.cross_validation_summary
    if isinstance(blob, Mapping) and blob:
        return dict(blob)
    if yaml_path.is_file():
        try:
            loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError, TypeError, ValueError):
            loaded = None
        if isinstance(loaded, Mapping) and loaded:
            return dict(loaded)
    return None


def generate_research_markdown_report(
    run_dir: Path | str,
    result: ResearchProtocolResult,
    protocol_config: DoubleOOSWFOProtocolConfig,
) -> None:
    """Write ``research_validation_report.md`` describing protocol outcomes."""

    root = Path(run_dir)
    lines: list[str] = []
    lines.append("# Research validation report")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Protocol type: double_oos_wfo")
    lines.append(f"- Protocol artifact version: {RESEARCH_PROTOCOL_ARTIFACT_VERSION}")
    lines.append(f"- Scenario: {result.scenario_name}")
    robust_cfg = protocol_config.robust_selection
    lines.append(
        f"- Robust selection: {'enabled' if robust_cfg.enabled else 'disabled'} "
        f"(weights cell={robust_cfg.cell_weight}, neighbor_median={robust_cfg.neighbor_median_weight}, "
        f"neighbor_min={robust_cfg.neighbor_min_weight})"
    )
    lines.append("")
    lines.append("## Global periods")
    lines.append("")
    gt = protocol_config.global_train_period
    ut = protocol_config.unseen_test_period
    lines.append(
        f"- Global train: {pd.Timestamp(gt.start).isoformat()} "
        f"through {pd.Timestamp(gt.end).isoformat()}"
    )
    lines.append(
        f"- Unseen test: {pd.Timestamp(ut.start).isoformat()} "
        f"through {pd.Timestamp(ut.end).isoformat()}"
    )
    lines.append("")
    if protocol_config.constraints:
        lines.append("## Metric constraints")
        lines.append("")
        lines.append(
            f"- Rules configured: {len(protocol_config.constraints)}; "
            f"architectures passing: {sum(1 for r in result.grid_results if r.constraint_passed)} "
            f"of {len(result.grid_results)}"
        )
        for i, c in enumerate(protocol_config.constraints, start=1):
            bounds: list[str] = []
            if c.min_value is not None:
                bounds.append(f"min={c.min_value}")
            if c.max_value is not None:
                bounds.append(f"max={c.max_value}")
            lines.append(f"- Rule {i}: {c.display_key} ({', '.join(bounds)})")
        failed = [r for r in result.grid_results if not r.constraint_passed]
        if failed:
            lines.append("")
            lines.append("Architectures failing constraints (first row failures):")
            for r in failed[:10]:
                arch = r.architecture
                fails = "; ".join(r.constraint_failures) if r.constraint_failures else "unknown"
                lines.append(
                    f"- train={arch.train_window_months} test={arch.test_window_months} "
                    f"step={arch.wfo_step_months} type={arch.walk_forward_type}: {fails}"
                )
            if len(failed) > 10:
                lines.append(f"- … {len(failed) - 10} additional failing row(s) omitted")
        lines.append("")
    cs_cfg_summary = resolve_cross_validation_summary(root, result)
    if cs_cfg_summary is not None:
        lines.append("## Cross-validation (global train)")
        lines.append("")
        lines.append(f"- Strategy: {cs_cfg_summary.get('strategy')}")
        lines.append(f"- Folds: {cs_cfg_summary.get('n_folds')}")
        fold_periods = cs_cfg_summary.get("fold_periods") or []
        if isinstance(fold_periods, (list, tuple)):
            fp_list = fold_periods
        else:
            fp_list = []
        for i, blk in enumerate(fp_list[:10], start=1):
            if not isinstance(blk, Mapping):
                continue
            lines.append(f"- Fold {i}: {blk.get('start')} -- {blk.get('end')}")
        if len(fp_list) > 10:
            lines.append(f"- … {len(fp_list) - 10} additional fold(s) omitted")
        lines.append("- Artifacts: [cross_validation_summary.yaml](cross_validation_summary.yaml)")
        lines.append("")
    lines.append("## Top selected architectures")
    lines.append("")
    if not result.selected_protocols:
        lines.append("(none)")
    else:
        lines.append("| Rank | Train months | Test months | Step months | Type | Score | Robust |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for sp in result.selected_protocols:
            arch = sp.architecture
            rs_cell = "" if sp.robust_score is None else str(sp.robust_score)
            lines.append(
                f"| {sp.rank} | {arch.train_window_months} | {arch.test_window_months} | "
                f"{arch.wfo_step_months} | {arch.walk_forward_type} | {sp.score} | {rs_cell} |"
            )
    lines.append("")
    lines.append("## Unseen validation")
    lines.append("")
    if result.unseen_result is None:
        lines.append(
            "Note: unseen validation was not run or produced no result " "(unseen_result is None)."
        )
    else:
        ur = result.unseen_result
        lines.append(f"- Mode: {ur.mode}")
        lines.append("")
        lines.append("Metrics:")
        for key in sorted(ur.metrics.keys()):
            lines.append(f"- {key}: {ur.metrics[key]}")
    lines.append("")
    cs_cfg = protocol_config.cost_sensitivity
    if cs_cfg.enabled:
        lines.append("## Cost sensitivity")
        lines.append("")
        lines.append(
            "- Artifacts: [cost_sensitivity.csv](cost_sensitivity.csv), "
            "[cost_sensitivity_summary.yaml](cost_sensitivity_summary.yaml)"
        )
        summ_path = root / "cost_sensitivity_summary.yaml"
        if summ_path.is_file():
            try:
                loaded = yaml.safe_load(summ_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError, TypeError, ValueError):
                loaded = None
            if isinstance(loaded, dict):
                sm = loaded.get("survival_metric")
                lines.append(f"- Survival metric: {sm}")
                cm = loaded.get("chosen_metrics")
                if isinstance(cm, list):
                    lines.append(f"- Recorded metrics: {', '.join(str(x) for x in cm)}")
                be = loaded.get("breakeven_slippage_bps_at_multiplier_1")
                if be is None:
                    lines.append(
                        "- Break-even slippage (commission multiplier 1.0): not determinable"
                    )
                else:
                    lines.append(f"- Break-even slippage (commission multiplier 1.0): {be} bps")
        else:
            lines.append("- Summary file not found for this run directory.")
        cost_png = root / "cost_sensitivity_survival_curve.png"
        if cost_png.is_file():
            lines.append(f"- Survival curve chart: [{cost_png.name}]({cost_png.name})")
        elif protocol_config.reporting.generate_cost_sensitivity_figure:
            lines.append(
                "- Note: survival curve PNG was not produced (CSV/summary may be incompatible)."
            )
        lines.append("")
    bs_cfg = protocol_config.bootstrap
    if bs_cfg.enabled:
        lines.append("## Bootstrap significance")
        lines.append("")
        lines.append(
            "Order: unseen validation, then cost sensitivity (when enabled), "
            "then bootstrap significance (post-selection; does not affect ranking)."
        )
        lines.append("")
        lines.append(
            "- Artifacts: [bootstrap_significance.csv](bootstrap_significance.csv), "
            "[bootstrap_summary.yaml](bootstrap_summary.yaml)"
        )
        distrib_paths = sorted(root.glob("bootstrap_distribution_*.csv"))
        if distrib_paths:
            lines.append("- Distribution sample CSVs:")
            for dp in distrib_paths:
                lines.append(f"  - [{dp.name}]({dp.name})")
        elif bs_cfg.persist_distribution_samples or (
            protocol_config.reporting.generate_bootstrap_distribution_plots
        ):
            lines.append(
                "- Note: Bootstrap distribution CSVs absent "
                "(persist disabled or bootstrap skipped)."
            )
        summ_path = root / "bootstrap_summary.yaml"
        if summ_path.is_file():
            try:
                loaded = yaml.safe_load(summ_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError, TypeError, ValueError):
                loaded = None
            if isinstance(loaded, dict):
                lines.append(f"- n_samples: {loaded.get('n_samples')}")
                lines.append(f"- random_seed: {loaded.get('random_seed')}")
                rw = loaded.get("random_wfo_architecture")
                if isinstance(rw, dict) and rw.get("enabled"):
                    lines.append(f"- random_wfo_architecture p-value: {rw.get('p_value')}")
                br = loaded.get("block_shuffled_returns")
                if isinstance(br, dict) and br.get("enabled"):
                    lines.append(f"- block_shuffled_returns p-value: {br.get('p_value')}")
                    lines.append(f"- block_size_days: {br.get('block_size_days')}")
                bpos = loaded.get("block_shuffled_positions")
                if isinstance(bpos, dict) and bpos.get("enabled"):
                    lines.append(f"- block_shuffled_positions p-value: {bpos.get('p_value')}")
                    lines.append(
                        "- block_shuffled_positions block_size_days: "
                        f"{bpos.get('block_size_days')}"
                    )
                rsp = loaded.get("random_strategy_parameters")
                if isinstance(rsp, dict) and rsp.get("enabled"):
                    lines.append(f"- random_strategy_parameters p-value: {rsp.get('p_value')}")
                    lines.append(
                        f"- random_strategy_parameters sample_size: {rsp.get('sample_size')}"
                    )
        else:
            lines.append("- Summary file not found for this run directory.")
        lines.append("")
    lines.append("## WFO heatmaps")
    lines.append("")
    rep_cfg = protocol_config.reporting
    if not rep_cfg.generate_heatmaps:
        lines.append("Heatmaps are disabled (`reporting.generate_heatmaps: false`).")
    else:
        heatmap_paths = sorted(root.glob("wfo_heatmap_*.png"))
        if heatmap_paths:
            for hp in heatmap_paths:
                lines.append(f"- [{hp.name}]({hp.name})")
        else:
            lines.append("No heatmap PNG files were generated for this run.")
    lines.append("")
    path = root / "research_validation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
