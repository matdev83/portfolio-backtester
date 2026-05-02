"""HTML summaries for research protocol validation (derivative of structured results)."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import yaml

from portfolio_backtester.research.protocol_config import (
    RESEARCH_PROTOCOL_ARTIFACT_VERSION,
    DoubleOOSWFOProtocolConfig,
)
from portfolio_backtester.research.report import resolve_cross_validation_summary
from portfolio_backtester.research.results import ResearchProtocolResult
from portfolio_backtester.research.static_charts import (
    write_bootstrap_distribution_chart_pngs,
    write_cost_sensitivity_line_chart_png,
)


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _append_assets_list_and_embed_images(
    parts: list[str],
    *,
    root: Path,
    glob_patterns: tuple[str, ...],
    embed: bool,
    title: str | None,
) -> None:
    pngs: list[Path] = []
    for gp in glob_patterns:
        pngs.extend(sorted(root.glob(gp)))
    if not pngs:
        parts.append("<p>No matching PNG artifacts were found.</p>")
        return
    parts.append("<ul>")
    for png in pngs:
        nm = png.name
        parts.append(f'<li><a href="{_esc(nm)}">{_esc(nm)}</a>')
        parts.append("</li>")
    parts.append("</ul>")
    if embed:
        parts.append('<div class="figure-grid">')
        for png in pngs:
            nm = png.name
            caption = nm if title is None else title
            parts.append("<figure>")
            parts.append(
                "<img "
                f'alt="{_esc(caption + ": " + nm)}" '
                f'src="{_esc(nm)}" />'
                f"<figcaption>{_esc(nm)}</figcaption>"
                "</figure>"
            )
        parts.append("</div>")


def generate_research_html_report(
    run_dir: Path | str,
    result: ResearchProtocolResult,
    protocol_config: DoubleOOSWFOProtocolConfig,
) -> None:
    """Write ``research_validation_report.html`` alongside markdown artifacts."""

    root = Path(run_dir)
    rep_cfg = protocol_config.reporting

    if rep_cfg.generate_bootstrap_distribution_plots:
        write_bootstrap_distribution_chart_pngs(root)

    cost_summary_yaml: dict[str, object] | None = None
    if rep_cfg.generate_cost_sensitivity_figure:
        ys = root / "cost_sensitivity_summary.yaml"
        if ys.is_file():
            try:
                raw_cs = yaml.safe_load(ys.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError, TypeError, ValueError):
                raw_cs = None
            if isinstance(raw_cs, dict):
                cost_summary_yaml = dict(raw_cs)
        write_cost_sensitivity_line_chart_png(root, summary=cost_summary_yaml)

    bs_cfg = protocol_config.bootstrap
    cv_pay = resolve_cross_validation_summary(root, result)

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append("<title>Research validation report</title>")
    parts.append("<style>")
    parts.append("body{font-family:Arial,sans-serif;margin:16px;line-height:1.35;}")
    parts.append(
        "nav{border:1px solid #ddd;padding:8px 12px;background:#fafafa;margin:12px 0;}"
        "nav ul{margin:4px 0;padding-left:20px;display:flex;"
        "flex-wrap:wrap;gap:12px 20px;list-style:disc}"
    )
    parts.append("table{border-collapse:collapse;margin:8px 0;width:auto;}")
    parts.append("th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;}")
    parts.append("th{background:#f4f4f4;}")
    parts.append("ul{margin:8px 0;padding-left:24px;}")
    parts.append("section{margin-bottom:24px;scroll-margin-top:12px}")
    parts.append(".figure-grid{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}")
    parts.append(
        ".figure-grid figure{border:1px solid #e0e0e0;margin:0;padding:6px;background:#fafafa}"
    )
    parts.append(".figure-grid img{display:block;width:520px;height:auto}")
    parts.append("</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<h1>Research validation report</h1>")

    if rep_cfg.html_navigation:
        toc: list[tuple[str, str]] = []
        toc.append(("protocol", "Protocol"))
        toc.append(("periods", "Global periods"))
        if protocol_config.constraints:
            toc.append(("constraints", "Metric constraints"))
        toc.append(("architectures", "Top architectures"))
        toc.append(("unseen", "Unseen validation"))
        if cv_pay is not None:
            toc.append(("cross-validation", "Cross-validation"))
        if protocol_config.cost_sensitivity.enabled:
            toc.append(("cost-sensitivity", "Cost sensitivity"))
        if bs_cfg.enabled:
            toc.append(("bootstrap", "Bootstrap"))
        toc.append(("heatmaps", "WFO heatmaps"))
        parts.append('<nav aria-label="Report sections"><ul>')
        for sec_id, label in toc:
            parts.append(f'<li><a href="#{_esc(sec_id)}">{_esc(label)}</a></li>')
        parts.append("</ul></nav>")

    parts.append('<section id="protocol">')
    parts.append("<h2>Protocol</h2>")
    parts.append("<ul>")
    parts.append("<li>Protocol type: double_oos_wfo</li>")
    parts.append(f"<li>Protocol artifact version: {_esc(RESEARCH_PROTOCOL_ARTIFACT_VERSION)}</li>")
    parts.append(f"<li>Scenario: {_esc(result.scenario_name)}</li>")
    robust_cfg = protocol_config.robust_selection
    rs_label = "enabled" if robust_cfg.enabled else "disabled"
    parts.append(
        "<li>Robust selection: "
        f"{_esc(rs_label)} "
        f"(weights cell={_esc(robust_cfg.cell_weight)}, "
        f"neighbor_median={_esc(robust_cfg.neighbor_median_weight)}, "
        f"neighbor_min={_esc(robust_cfg.neighbor_min_weight)})"
        "</li>"
    )
    parts.append("</ul>")
    parts.append("</section>")

    parts.append('<section id="periods">')
    parts.append("<h2>Global periods</h2>")
    parts.append("<ul>")
    gt = protocol_config.global_train_period
    ut = protocol_config.unseen_test_period
    parts.append(
        "<li>Global train: "
        f"{_esc(pd.Timestamp(gt.start).isoformat())} through {_esc(pd.Timestamp(gt.end).isoformat())}"
        "</li>"
    )
    parts.append(
        "<li>Unseen test: "
        f"{_esc(pd.Timestamp(ut.start).isoformat())} through {_esc(pd.Timestamp(ut.end).isoformat())}"
        "</li>"
    )
    parts.append("</ul>")
    parts.append("</section>")

    if protocol_config.constraints:
        parts.append('<section id="constraints">')
        parts.append("<h2>Metric constraints</h2>")
        passed_n = sum(1 for r in result.grid_results if r.constraint_passed)
        parts.append("<ul>")
        parts.append(
            "<li>"
            f"Rules configured: {_esc(len(protocol_config.constraints))}; "
            f"architectures passing: {_esc(passed_n)} of {_esc(len(result.grid_results))}"
            "</li>"
        )
        for i, c in enumerate(protocol_config.constraints, start=1):
            bounds: list[str] = []
            if c.min_value is not None:
                bounds.append(f"min={c.min_value}")
            if c.max_value is not None:
                bounds.append(f"max={c.max_value}")
            parts.append(
                f"<li>Rule {_esc(i)}: {_esc(c.display_key)} ({_esc(', '.join(bounds))})</li>"
            )
        parts.append("</ul>")
        failed = [r for r in result.grid_results if not r.constraint_passed]
        if failed:
            parts.append("<p>Architectures failing constraints (first row failures):</p>")
            parts.append("<ul>")
            for r in failed[:10]:
                arch = r.architecture
                fails = "; ".join(r.constraint_failures) if r.constraint_failures else "unknown"
                parts.append(
                    "<li>"
                    f"train={_esc(arch.train_window_months)} test={_esc(arch.test_window_months)} "
                    f"step={_esc(arch.wfo_step_months)} type={_esc(arch.walk_forward_type)}: "
                    f"{_esc(fails)}"
                    "</li>"
                )
            if len(failed) > 10:
                parts.append(f"<li>{_esc(len(failed) - 10)} additional failing row(s) omitted</li>")
            parts.append("</ul>")
        parts.append("</section>")

    parts.append('<section id="architectures">')
    parts.append("<h2>Top selected architectures</h2>")
    if not result.selected_protocols:
        parts.append("<p>(none)</p>")
    else:
        parts.append("<table>")
        parts.append("<thead><tr>")
        for col in (
            "Rank",
            "Train months",
            "Test months",
            "Step months",
            "Type",
            "Score",
            "Robust",
        ):
            parts.append(f"<th>{col}</th>")
        parts.append("</tr></thead>")
        parts.append("<tbody>")
        for sp in result.selected_protocols:
            arch = sp.architecture
            rs_cell = "" if sp.robust_score is None else str(sp.robust_score)
            parts.append("<tr>")
            parts.extend(
                [
                    f"<td>{_esc(sp.rank)}</td>",
                    f"<td>{_esc(arch.train_window_months)}</td>",
                    f"<td>{_esc(arch.test_window_months)}</td>",
                    f"<td>{_esc(arch.wfo_step_months)}</td>",
                    f"<td>{_esc(arch.walk_forward_type)}</td>",
                    f"<td>{_esc(sp.score)}</td>",
                    f"<td>{_esc(rs_cell)}</td>",
                ]
            )
            parts.append("</tr>")
        parts.append("</tbody></table>")
    parts.append("</section>")

    parts.append('<section id="unseen">')
    parts.append("<h2>Unseen validation</h2>")
    if result.unseen_result is None:
        parts.append(
            "<p>Note: unseen validation was not run or produced no result "
            "(unseen_result is None).</p>"
        )
    else:
        ur = result.unseen_result
        parts.append(f"<p>Mode: {_esc(ur.mode)}</p>")
        parts.append("<p>Metrics:</p>")
        parts.append("<ul>")
        for key in sorted(ur.metrics.keys()):
            parts.append(f"<li>{_esc(key)}: {_esc(ur.metrics[key])}</li>")
        parts.append("</ul>")
    parts.append("</section>")

    if cv_pay is not None:
        parts.append('<section id="cross-validation">')
        parts.append("<h2>Cross-validation summary</h2>")
        parts.append("<ul>")
        enabled = cv_pay.get("enabled")
        if enabled is None:
            parts.append("<li>enabled: inferred from archived summary</li>")
        else:
            parts.append(f"<li>enabled: {_esc(enabled)}</li>")
        strat = cv_pay.get("strategy")
        if strat is not None:
            parts.append(f"<li>strategy: {_esc(strat)}</li>")
        nf = cv_pay.get("n_folds")
        if nf is not None:
            parts.append(f"<li>n_folds: {_esc(nf)}</li>")
        fp = cv_pay.get("fold_periods")
        if isinstance(fp, list) and fp:
            parts.append("<li>Fold periods:")
            parts.append("<ol>")
            for entry in fp:
                if isinstance(entry, dict):
                    parts.append(
                        "<li>"
                        f"{_esc(entry.get('start',''))} to {_esc(entry.get('end',''))}"
                        "</li>"
                    )
                else:
                    parts.append(f"<li>{_esc(entry)}</li>")
            parts.append("</ol>")
            parts.append("</li>")
        yaml_link = '<a href="cross_validation_summary.yaml">cross_validation_summary.yaml</a>'
        parts.append(f"<li>Persisted YAML: {yaml_link}</li>")
        parts.append("</ul>")
        parts.append("</section>")

    cs_cfg = protocol_config.cost_sensitivity
    if cs_cfg.enabled:
        parts.append('<section id="cost-sensitivity">')
        parts.append("<h2>Cost sensitivity</h2>")
        parts.append("<ul>")
        parts.append(
            "<li>Artifacts: "
            '<a href="cost_sensitivity.csv">cost_sensitivity.csv</a>, '
            '<a href="cost_sensitivity_summary.yaml">cost_sensitivity_summary.yaml</a>'
            "</li>"
        )
        summ_path = root / "cost_sensitivity_summary.yaml"
        if summ_path.is_file():
            try:
                loaded = yaml.safe_load(summ_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError, TypeError, ValueError):
                loaded = None
            if isinstance(loaded, dict):
                sm = loaded.get("survival_metric")
                parts.append(f"<li>Survival metric: {_esc(sm)}</li>")
                cm = loaded.get("chosen_metrics")
                if isinstance(cm, list):
                    parts.append(
                        "<li>Recorded metrics: " f"{_esc(', '.join(str(x) for x in cm))}" "</li>"
                    )
                be = loaded.get("breakeven_slippage_bps_at_multiplier_1")
                if be is None:
                    parts.append(
                        "<li>Break-even slippage (commission multiplier 1.0): not determinable</li>"
                    )
                else:
                    parts.append(
                        "<li>Break-even slippage (commission multiplier 1.0): "
                        f"{_esc(be)} bps</li>"
                    )
        else:
            parts.append("<li>Summary file not found for this run directory.</li>")
        parts.append("</ul>")
        cost_png = root / "cost_sensitivity_survival_curve.png"
        embed = protocol_config.reporting.html_embed_figures
        if embed and cost_png.is_file():
            nm = cost_png.name
            parts.append("<figure>")
            parts.append(
                f'<img alt="Cost sensitivity survival curve" src="{_esc(nm)}" />'
                f"<figcaption>{_esc(nm)}</figcaption>"
                "</figure>"
            )
        elif cost_png.is_file() and not embed:
            nm = cost_png.name
            parts.append(
                "<p>Cost sensitivity figure: " f'<a href="{_esc(nm)}">{_esc(nm)}</a>' "</p>"
            )
        elif rep_cfg.generate_cost_sensitivity_figure and not cost_png.is_file():
            parts.append(
                "<p>Note: survival curve PNG was not produced (CSV/summary incompatible).</p>"
            )
        parts.append("</section>")

    if bs_cfg.enabled:
        parts.append('<section id="bootstrap">')
        parts.append("<h2>Bootstrap significance</h2>")
        parts.append("<p>")
        parts.append(
            "Order: unseen validation, then cost sensitivity (when enabled), "
            "then bootstrap significance (post-selection; does not affect ranking)."
        )
        parts.append("</p>")
        parts.append("<ul>")
        parts.append(
            "<li>Artifacts: "
            '<a href="bootstrap_significance.csv">bootstrap_significance.csv</a>, '
            '<a href="bootstrap_summary.yaml">bootstrap_summary.yaml</a>'
            "</li>"
        )

        distrib_glob_any = sorted(root.glob("bootstrap_distribution_*.csv"))
        if distrib_glob_any:
            parts.append("<li>")
            inner = ", ".join(
                f'<a href="{_esc(dp.name)}">{_esc(dp.name)}</a>' for dp in distrib_glob_any
            )
            parts.append(f"Distribution samples ({inner}):")
            parts.append("</li>")
        elif bs_cfg.persist_distribution_samples or rep_cfg.generate_bootstrap_distribution_plots:
            parts.append(
                "<li>Note: Bootstrap distribution CSVs absent (persist disabled or bootstrap skipped).</li>"
            )

        summ_path = root / "bootstrap_summary.yaml"
        if summ_path.is_file():
            try:
                loaded = yaml.safe_load(summ_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError, TypeError, ValueError):
                loaded = None
            if isinstance(loaded, dict):
                parts.append(f"<li>n_samples: {_esc(loaded.get('n_samples'))}</li>")
                parts.append(f"<li>random_seed: {_esc(loaded.get('random_seed'))}</li>")
                rw = loaded.get("random_wfo_architecture")
                if isinstance(rw, dict) and rw.get("enabled"):
                    parts.append(
                        "<li>random_wfo_architecture p-value: " f"{_esc(rw.get('p_value'))}</li>"
                    )
                br = loaded.get("block_shuffled_returns")
                if isinstance(br, dict) and br.get("enabled"):
                    parts.append(
                        "<li>block_shuffled_returns p-value: " f"{_esc(br.get('p_value'))}</li>"
                    )
                    parts.append(f"<li>block_size_days: {_esc(br.get('block_size_days'))}</li>")
                bpos = loaded.get("block_shuffled_positions")
                if isinstance(bpos, dict) and bpos.get("enabled"):
                    parts.append(
                        "<li>block_shuffled_positions p-value: " f"{_esc(bpos.get('p_value'))}</li>"
                    )
                    parts.append(
                        "<li>block_shuffled_positions block_size_days: "
                        f"{_esc(bpos.get('block_size_days'))}</li>"
                    )
                rsp = loaded.get("random_strategy_parameters")
                if isinstance(rsp, dict) and rsp.get("enabled"):
                    parts.append(
                        "<li>random_strategy_parameters p-value: "
                        f"{_esc(rsp.get('p_value'))}</li>"
                    )
                    parts.append(
                        "<li>random_strategy_parameters sample_size: "
                        f"{_esc(rsp.get('sample_size'))}</li>"
                    )
        else:
            parts.append("<li>Summary file not found for this run directory.</li>")
        parts.append("</ul>")

        if sorted(root.glob("bootstrap_viz_*.png")):
            _append_assets_list_and_embed_images(
                parts,
                root=root,
                glob_patterns=("bootstrap_viz_*.png",),
                embed=rep_cfg.html_embed_figures,
                title="Bootstrap",
            )

        parts.append("</section>")

    parts.append('<section id="heatmaps">')
    parts.append("<h2>WFO heatmaps</h2>")
    if not rep_cfg.generate_heatmaps:
        parts.append(
            "<p>Heatmaps are disabled (<code>reporting.generate_heatmaps: false</code>).</p>"
        )
    else:
        heatmap_paths = sorted(root.glob("wfo_heatmap_*.png"))
        if not heatmap_paths:
            parts.append("<p>No heatmap PNG files were generated for this run.</p>")
        elif rep_cfg.html_embed_figures:
            parts.append("<ul>")
            for hp in heatmap_paths:
                name = hp.name
                parts.append(f'<li><a href="{_esc(name)}">{_esc(name)}</a></li>')
            parts.append("</ul>")
            parts.append('<div class="figure-grid">')
            for hp in heatmap_paths:
                nm = hp.name
                parts.append("<figure>")
                parts.append(f'<img alt="{_esc(nm)}" src="{_esc(nm)}" />')
                parts.append(f"<figcaption>{_esc(nm)}</figcaption>")
                parts.append("</figure>")
            parts.append("</div>")
        else:
            parts.append("<ul>")
            for hp in heatmap_paths:
                name = hp.name
                parts.append(f'<li><a href="{_esc(name)}">{_esc(name)}</a></li>')
            parts.append("</ul>")
    parts.append("</section>")

    parts.append("</body></html>")
    path = root / "research_validation_report.html"
    path.write_text("\n".join(parts), encoding="utf-8")
