"""Pure helpers and orchestration entry for double OOS walk-forward protocols."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Mapping, Sequence

import numpy as np
import pandas as pd

from ..canonical_config import CanonicalScenarioConfig
from ..optimization.results import OptimizationResult
from ..reporting.performance_metrics import calculate_metrics
from ..scenario_normalizer import ScenarioNormalizer
from .artifacts import (
    ResearchArtifactWriter,
    write_grid_results,
    write_lock_file,
    write_selected_protocols,
    write_unseen_results,
)
from .registry import ResearchRunRegistry, compute_registry_hashes
from .heatmaps import write_wfo_heatmaps
from .constraints import ConstraintEvaluator, ResearchConstraintError
from .bootstrap import run_research_bootstrap, write_bootstrap_artifacts
from .cost_sensitivity import (
    build_cost_sensitivity_summary,
    effective_global_config_for_cost_cell,
    expand_cost_sensitivity_grid,
    row_survives,
    survival_metric_for_selection,
    write_cost_sensitivity_artifacts,
)
from .protocol_config import (
    CostSensitivityRunOn,
    DateRangeConfig,
    DoubleOOSWFOProtocolConfig,
    FinalUnseenMode,
    ResearchProtocolConfigError,
    WFOGridConfig,
)
from .html_report import generate_research_html_report
from .report import generate_research_markdown_report
from .results import (
    ResearchProtocolResult,
    SelectedProtocol,
    UnseenValidationResult,
    WFOArchitecture,
    WFOArchitectureResult,
)
from .scoring import (
    ResearchScoreCalculator,
    assign_robust_scores_to_results,
    compute_composite_rank_scores_for_results,
    extract_metric_value,
    is_robust_composite_metric,
    select_top_selected_protocols,
    select_top_selected_protocols_robust_composite,
)

logger = logging.getLogger(__name__)


def _trade_history_from_backtest_payload(bt_payload: Mapping[str, Any]) -> pd.DataFrame | None:
    raw = bt_payload.get("trade_history")
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        return raw
    return None


def expand_wfo_architecture_grid(
    grid: WFOGridConfig | None = None,
    *,
    train_window_months: Sequence[int] | None = None,
    test_window_months: Sequence[int] | None = None,
    wfo_step_months: Sequence[int] | None = None,
    walk_forward_type: Sequence[str] | None = None,
) -> List[WFOArchitecture]:
    """Expand architecture grid in fixed order: train, test, step, type.

    Deduplicates exact architecture tuples while preserving first-seen order.

    Args:
        grid: Pre-built grid configuration (preferred when available).
        train_window_months: Training window lengths (months).
        test_window_months: Test window lengths (months).
        wfo_step_months: Step sizes (months).
        walk_forward_type: Walk-forward mode names (e.g. rolling, expanding).

    Returns:
        Ordered list of unique ``WFOArchitecture`` rows.
    """

    if grid is not None:
        tr_seq = grid.train_window_months
        te_seq = grid.test_window_months
        st_seq = grid.wfo_step_months
        wft_seq = grid.walk_forward_type
    else:
        if (
            train_window_months is None
            or test_window_months is None
            or wfo_step_months is None
            or walk_forward_type is None
        ):
            msg = "expand_wfo_architecture_grid requires grid= or all window sequences"
            raise TypeError(msg)
        tr_seq = tuple(train_window_months)
        te_seq = tuple(test_window_months)
        st_seq = tuple(wfo_step_months)
        wft_seq = tuple(walk_forward_type)

    seen: set[WFOArchitecture] = set()
    out: List[WFOArchitecture] = []
    for tr in tr_seq:
        for te in te_seq:
            for st in st_seq:
                for wft in wft_seq:
                    arch = WFOArchitecture(
                        train_window_months=int(tr),
                        test_window_months=int(te),
                        wfo_step_months=int(st),
                        walk_forward_type=str(wft),
                    )
                    if arch in seen:
                        continue
                    seen.add(arch)
                    out.append(arch)
    return out


def _timestamp_for_index_compare(ts: pd.Timestamp, index: pd.Index) -> pd.Timestamp:
    """Align ``ts`` to ``index`` timezone so comparisons do not mix naive/aware."""

    t = pd.Timestamp(ts)
    if not isinstance(index, pd.DatetimeIndex):
        return t
    idx_tz = index.tz
    if idx_tz is not None:
        if t.tzinfo is None:
            return t.tz_localize(idx_tz)
        return t.tz_convert(idx_tz)
    if t.tzinfo is not None:
        return t.tz_localize(None)
    return t


def slice_panel_by_dates(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return row slice ``[start, end]`` without mutating ``df``."""

    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy()
    idx = df.index
    start_cmp = _timestamp_for_index_compare(start, idx)
    end_cmp = _timestamp_for_index_compare(end, idx)
    mask = (idx >= start_cmp) & (idx <= end_cmp)
    return df.loc[mask].copy()


def training_scenario_dict_for_architecture(
    scenario_config: CanonicalScenarioConfig,
    period: DateRangeConfig,
    architecture: WFOArchitecture,
) -> dict[str, Any]:
    """Build a normalized scenario dictionary for grid optimization on ``period``."""

    merged = scenario_config.to_dict()
    merged["start_date"] = pd.Timestamp(period.start).strftime("%Y-%m-%d")
    merged["end_date"] = pd.Timestamp(period.end).strftime("%Y-%m-%d")
    merged["train_window_months"] = architecture.train_window_months
    merged["test_window_months"] = architecture.test_window_months
    merged["wfo_step_months"] = architecture.wfo_step_months
    merged["walk_forward_type"] = architecture.walk_forward_type
    merged["wfo_mode"] = "reoptimize"
    return merged


def unseen_scenario_dict_reoptimize(
    scenario_config: CanonicalScenarioConfig,
    period: DateRangeConfig,
    selected: SelectedProtocol,
) -> dict[str, Any]:
    """Scenario dict for unseen holdout optimization with frozen architecture."""

    merged = scenario_config.to_dict()
    merged["start_date"] = pd.Timestamp(period.start).strftime("%Y-%m-%d")
    merged["end_date"] = pd.Timestamp(period.end).strftime("%Y-%m-%d")
    arch = selected.architecture
    merged["train_window_months"] = arch.train_window_months
    merged["test_window_months"] = arch.test_window_months
    merged["wfo_step_months"] = arch.wfo_step_months
    merged["walk_forward_type"] = arch.walk_forward_type
    merged["wfo_mode"] = "reoptimize"
    return merged


def unseen_scenario_dict_fixed_params(
    scenario_config: CanonicalScenarioConfig,
    period: DateRangeConfig,
    selected: SelectedProtocol,
) -> dict[str, Any]:
    """Scenario dict for unseen single backtest using locked parameters."""

    merged = scenario_config.to_dict()
    merged["start_date"] = pd.Timestamp(period.start).strftime("%Y-%m-%d")
    merged["end_date"] = pd.Timestamp(period.end).strftime("%Y-%m-%d")
    arch = selected.architecture
    merged["train_window_months"] = arch.train_window_months
    merged["test_window_months"] = arch.test_window_months
    merged["wfo_step_months"] = arch.wfo_step_months
    merged["walk_forward_type"] = arch.walk_forward_type
    base_params = dict(merged.get("strategy_params") or {})
    base_params.update(dict(selected.selected_parameters))
    merged["strategy_params"] = base_params
    return merged


def extract_strategy_params_space(
    scenario_config: CanonicalScenarioConfig,
) -> dict[str, list[Any]] | None:
    """Resolve ``strategy_params_space`` from scenario extras or ``strategy_params``."""

    raw: Any = None
    extras = scenario_config.extras
    if isinstance(extras, Mapping):
        raw = extras.get("strategy_params_space")
    if raw is None:
        sp = scenario_config.strategy_params
        if isinstance(sp, Mapping):
            raw = sp.get("strategy_params_space")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, list[Any]] = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)) and len(v) > 0:
            out[str(k)] = list(v)
    return out if out else None


def _turnover_proxy_from_returns(stitched: pd.Series) -> float:
    """Single-number turnover proxy from returns (lower = less churn) for composite scoring."""

    if stitched is None or stitched.empty:
        return 0.0
    arr = pd.to_numeric(stitched, errors="coerce").astype(float).to_numpy()
    if arr.size <= 1:
        return float(np.nansum(np.abs(arr)))
    return float(np.nansum(np.abs(np.diff(arr))))


class DoubleOOSWFOProtocol:
    """Double out-of-sample walk-forward research validation protocol."""

    def __init__(
        self,
        optimization_orchestrator: Any,
        backtest_runner: Any,
        artifact_writer: ResearchArtifactWriter | None = None,
        *,
        scenario_normalizer: ScenarioNormalizer | None = None,
    ) -> None:
        """Initialize protocol with injected execution dependencies."""

        self._optimization_orchestrator = optimization_orchestrator
        self._backtest_runner = backtest_runner
        self._artifact_writer = artifact_writer
        self._normalizer = scenario_normalizer or ScenarioNormalizer()

    def _effective_writer(self) -> ResearchArtifactWriter:
        return self._artifact_writer or ResearchArtifactWriter(Path("data/reports"))

    @staticmethod
    def _effective_benchmark_ticker(global_config: Mapping[str, Any]) -> str:
        ticker = global_config.get("benchmark_ticker", global_config.get("benchmark"))
        if ticker is None or not str(ticker):
            return "SPY"
        return str(ticker)

    @staticmethod
    def _optimization_result_to_series(result: OptimizationResult) -> pd.Series:
        stitched = getattr(result, "stitched_returns", None)
        if isinstance(stitched, pd.Series):
            active = stitched.dropna()
            if not active.empty:
                return stitched
        return pd.Series(dtype=float)

    def _benchmark_returns_aligned(
        self,
        stitched: pd.Series,
        daily_slice: pd.DataFrame,
        benchmark_ticker: str,
    ) -> pd.Series:
        if stitched.empty or not isinstance(stitched.index, pd.DatetimeIndex):
            return pd.Series(0.0, index=stitched.index)
        zeros = pd.Series(0.0, index=stitched.index)
        if daily_slice.empty:
            return zeros
        try:
            if isinstance(
                daily_slice.columns, pd.MultiIndex
            ) and "Close" in daily_slice.columns.get_level_values("Field"):
                daily_close = daily_slice.xs("Close", level="Field", axis=1)
            else:
                daily_close = daily_slice
            if benchmark_ticker in daily_close.columns:
                return (
                    daily_close[benchmark_ticker]
                    .pct_change(fill_method=None)
                    .reindex(stitched.index)
                    .fillna(0.0)
                )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Benchmark extraction failed for research protocol; using zeros.", exc_info=True
            )
        return zeros

    def _metrics_from_returns(
        self,
        stitched: pd.Series,
        daily_slice: pd.DataFrame,
        global_config: Mapping[str, Any],
    ) -> dict[str, float]:
        ticker = self._effective_benchmark_ticker(global_config)
        benchmark_returns = (
            pd.Series(0.0, index=stitched.index)
            if stitched.empty
            else self._benchmark_returns_aligned(stitched, daily_slice, ticker)
        )
        metrics_series = calculate_metrics(stitched, benchmark_returns, ticker)
        metrics = {
            k: float(v) if not pd.isna(v) else float("nan") for k, v in metrics_series.items()
        }
        metrics["Turnover"] = _turnover_proxy_from_returns(stitched)
        return metrics

    def run(
        self,
        *,
        scenario_config: CanonicalScenarioConfig,
        protocol_config: DoubleOOSWFOProtocolConfig,
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
        args: argparse.Namespace,
        global_config: Mapping[str, Any],
    ) -> ResearchProtocolResult:
        """Execute the double-OOS protocol (grid selection + optional unseen evaluation)."""

        writer = self._effective_writer()
        run_dir = writer.create_run_directory(scenario_config.name)

        gt = protocol_config.global_train_period
        ut = protocol_config.unseen_test_period
        monthly_tr = slice_panel_by_dates(monthly_data, gt.start, gt.end)
        daily_tr = slice_panel_by_dates(daily_data, gt.start, gt.end)
        rets_tr = slice_panel_by_dates(rets_full, gt.start, gt.end)

        architectures = expand_wfo_architecture_grid(grid=protocol_config.wfo_window_grid)
        max_cells = protocol_config.execution.max_grid_cells
        if len(architectures) > max_cells:
            msg = (
                f"expanded WFO architecture grid has {len(architectures)} unique cell(s), "
                f"which exceeds research_protocol.execution.max_grid_cells={max_cells}"
            )
            raise ResearchProtocolConfigError(msg)
        use_composite = is_robust_composite_metric(protocol_config.selection.metric)
        constraint_evaluator = (
            ConstraintEvaluator(protocol_config.constraints)
            if protocol_config.constraints
            else None
        )

        grid_results: list[WFOArchitectureResult] = []
        for arch in architectures:
            scen_dict = training_scenario_dict_for_architecture(scenario_config, gt, arch)
            canonical_cell = self._normalizer.normalize(
                scenario=scen_dict, global_config=dict(global_config)
            )
            opt_result = self._optimization_orchestrator.run_optimization(
                canonical_cell, monthly_tr, daily_tr, rets_tr, args
            )
            stitched = self._optimization_result_to_series(opt_result)
            metrics = self._metrics_from_returns(stitched, daily_tr, global_config)
            passed, fails = (
                constraint_evaluator.evaluate(metrics)
                if constraint_evaluator is not None
                else (True, ())
            )
            stitched_opt = (
                stitched if isinstance(stitched, pd.Series) and not stitched.empty else None
            )
            if use_composite:
                grid_results.append(
                    WFOArchitectureResult(
                        architecture=arch,
                        metrics=metrics,
                        score=0.0,
                        robust_score=None,
                        best_parameters=dict(opt_result.best_parameters),
                        n_evaluations=int(opt_result.n_evaluations),
                        stitched_returns=stitched_opt,
                        constraint_passed=passed,
                        constraint_failures=fails,
                    )
                )
            else:
                calculator = ResearchScoreCalculator(metric_key=protocol_config.selection.metric)
                raw_metric = extract_metric_value(metrics, protocol_config.selection.metric)
                score = calculator.score_for_ranking(raw_metric)
                grid_results.append(
                    WFOArchitectureResult(
                        architecture=arch,
                        metrics=metrics,
                        score=score,
                        robust_score=None,
                        best_parameters=dict(opt_result.best_parameters),
                        n_evaluations=int(opt_result.n_evaluations),
                        stitched_returns=stitched_opt,
                        constraint_passed=passed,
                        constraint_failures=fails,
                    )
                )

        eligible = [r for r in grid_results if r.constraint_passed]
        if not eligible:
            msg = (
                "no walk-forward architecture satisfied research_protocol.constraints; "
                f"evaluated {len(grid_results)} grid row(s)"
            )
            raise ResearchConstraintError(msg)

        scoring_cfg = protocol_config.composite_scoring

        if use_composite:
            if scoring_cfg is None:
                msg = "composite_scoring is required for RobustComposite selection"
                raise TypeError(msg)
            composite_scores = compute_composite_rank_scores_for_results(eligible, scoring_cfg)
            arch_to_score = {
                r.architecture: cs for r, cs in zip(eligible, composite_scores, strict=True)
            }
            grid_results = [
                WFOArchitectureResult(
                    architecture=gr.architecture,
                    metrics=gr.metrics,
                    score=arch_to_score.get(gr.architecture, float("nan")),
                    robust_score=gr.robust_score,
                    best_parameters=gr.best_parameters,
                    n_evaluations=gr.n_evaluations,
                    stitched_returns=gr.stitched_returns,
                    constraint_passed=gr.constraint_passed,
                    constraint_failures=gr.constraint_failures,
                )
                for gr in grid_results
            ]

        grid_results = list(
            assign_robust_scores_to_results(
                tuple(grid_results),
                protocol_config.robust_selection,
            )
        )
        eligible = [r for r in grid_results if r.constraint_passed]
        rank_robust = protocol_config.robust_selection.enabled

        if use_composite:
            if scoring_cfg is None:
                msg = "composite_scoring is required for RobustComposite selection"
                raise TypeError(msg)
            selected = select_top_selected_protocols_robust_composite(
                eligible,
                composite=scoring_cfg,
                top_n=protocol_config.selection.top_n,
                rank_by_robust=rank_robust,
            )
        else:
            selected = select_top_selected_protocols(
                eligible,
                metric_key=protocol_config.selection.metric,
                top_n=protocol_config.selection.top_n,
                rank_by_robust=rank_robust,
            )

        write_grid_results(run_dir, grid_results)
        if protocol_config.reporting.enabled and protocol_config.reporting.generate_heatmaps:
            write_wfo_heatmaps(run_dir, grid_results, protocol_config)
        write_selected_protocols(run_dir, selected)

        refuse_overwrite_eff = (
            False
            if getattr(args, "force_new_research_run", False)
            else bool(protocol_config.lock.refuse_overwrite)
        )
        winner = selected[0]
        write_lock_file(
            run_dir,
            scenario_config,
            global_config,
            protocol_config,
            winner,
            refuse_overwrite=refuse_overwrite_eff,
        )

        protocol_root = writer.scenario_protocol_root(scenario_config.name)
        run_id = run_dir.name
        lock_rel = Path(run_id) / "protocol_lock.yaml"
        sh, pch, uph = compute_registry_hashes(scenario_config, protocol_config)
        registry = ResearchRunRegistry(protocol_root / "registry.yaml")
        skip_unseen = bool(getattr(args, "research_skip_unseen", False))
        force_run = bool(getattr(args, "force_new_research_run", False))

        if not skip_unseen:
            registry.assert_no_completed_duplicate(sh, pch, uph, force_new_research_run=force_run)
        registry.record_lock(
            run_id=run_id,
            scenario_hash=sh,
            protocol_config_hash=pch,
            unseen_period_hash=uph,
            lock_path=str(lock_rel.as_posix()),
        )

        unseen_result: UnseenValidationResult | None = None
        if not getattr(args, "research_skip_unseen", False):
            monthly_ut = slice_panel_by_dates(monthly_data, ut.start, ut.end)
            daily_ut = slice_panel_by_dates(daily_data, ut.start, ut.end)
            rets_ut = slice_panel_by_dates(rets_full, ut.start, ut.end)
            unseen_result = self._run_final_unseen(
                scenario_config=scenario_config,
                protocol_config=protocol_config,
                selected=winner,
                monthly_ut=monthly_ut,
                daily_ut=daily_ut,
                rets_ut=rets_ut,
                args=args,
                global_config=global_config,
            )
            write_unseen_results(run_dir, unseen_result)
            registry.mark_unseen_completed(run_id)
            cs_cfg = protocol_config.cost_sensitivity
            if (
                cs_cfg.enabled
                and cs_cfg.run_on == CostSensitivityRunOn.UNSEEN
                and unseen_result is not None
            ):
                cost_rows = self._run_cost_sensitivity_rows(
                    scenario_config=scenario_config,
                    protocol_config=protocol_config,
                    selected=winner,
                    monthly_ut=monthly_ut,
                    daily_ut=daily_ut,
                    rets_ut=rets_ut,
                    args=args,
                    global_config=global_config,
                )
                cs_summary = build_cost_sensitivity_summary(
                    rows=cost_rows,
                    cost_config=cs_cfg,
                    baseline_global=global_config,
                    selection_metric=protocol_config.selection.metric,
                )
                write_cost_sensitivity_artifacts(run_dir, cost_rows, cs_summary)

            bs_cfg = protocol_config.bootstrap
            if bs_cfg.enabled and unseen_result is not None:
                param_space = extract_strategy_params_space(scenario_config)

                def _run_with_sampled_params(
                    sampled: Mapping[str, Any],
                ) -> Mapping[str, float]:
                    ur_inner = self._run_final_unseen(
                        scenario_config=scenario_config,
                        protocol_config=protocol_config,
                        selected=winner,
                        monthly_ut=monthly_ut,
                        daily_ut=daily_ut,
                        rets_ut=rets_ut,
                        args=args,
                        global_config=global_config,
                        strategy_parameter_overrides=sampled,
                    )
                    return ur_inner.metrics

                payload = run_research_bootstrap(
                    cfg=bs_cfg,
                    grid_results=grid_results,
                    selected=winner,
                    unseen_result=unseen_result,
                    selection_metric=protocol_config.selection.metric,
                    metrics_from_returns=lambda s: self._metrics_from_returns(
                        s, daily_ut, global_config
                    ),
                    param_space=param_space,
                    run_with_params_fn=_run_with_sampled_params,
                    trade_history=unseen_result.trade_history,
                    asset_returns=rets_ut,
                )
                if payload is not None:
                    bs_summary, bs_rows = payload
                    write_bootstrap_artifacts(run_dir, bs_summary, bs_rows)

        result = ResearchProtocolResult(
            scenario_name=scenario_config.name,
            grid_results=tuple(grid_results),
            selected_protocols=selected,
            unseen_result=unseen_result,
            artifact_dir=run_dir,
        )
        if protocol_config.reporting.enabled:
            generate_research_markdown_report(run_dir, result, protocol_config)
            if protocol_config.reporting.generate_html:
                generate_research_html_report(run_dir, result, protocol_config)
        return result

    def _run_final_unseen(
        self,
        *,
        scenario_config: CanonicalScenarioConfig,
        protocol_config: DoubleOOSWFOProtocolConfig,
        selected: SelectedProtocol,
        monthly_ut: pd.DataFrame,
        daily_ut: pd.DataFrame,
        rets_ut: pd.DataFrame,
        args: argparse.Namespace,
        global_config: Mapping[str, Any],
        strategy_parameter_overrides: Mapping[str, Any] | None = None,
    ) -> UnseenValidationResult:
        ut = protocol_config.unseen_test_period

        if strategy_parameter_overrides is not None:
            merged_params = dict(selected.selected_parameters)
            merged_params.update(dict(strategy_parameter_overrides))
            selected_eff = replace(selected, selected_parameters=merged_params)
            scen_dict = unseen_scenario_dict_fixed_params(scenario_config, ut, selected_eff)
            unseen_canon = self._normalizer.normalize(
                scenario=scen_dict, global_config=dict(global_config)
            )
            study_name = getattr(args, "study_name", None)
            bt_payload = self._backtest_runner.run_backtest_mode(
                unseen_canon, monthly_ut, daily_ut, rets_ut, study_name=study_name
            )
            rets_bt = bt_payload.get("returns")
            stitched = pd.Series(dtype=float)
            if isinstance(rets_bt, pd.Series):
                stitched = rets_bt
            metrics = self._metrics_from_returns(stitched, daily_ut, global_config)
            return UnseenValidationResult(
                selected_protocol=selected_eff,
                metrics=metrics,
                returns=stitched,
                mode=FinalUnseenMode.FIXED_SELECTED_PARAMS.value,
                trade_history=_trade_history_from_backtest_payload(bt_payload),
            )

        mode = protocol_config.final_unseen_mode

        if mode == FinalUnseenMode.REOPTIMIZE_WITH_LOCKED_ARCHITECTURE:
            scen_dict = unseen_scenario_dict_reoptimize(scenario_config, ut, selected)
            unseen_canon = self._normalizer.normalize(
                scenario=scen_dict, global_config=dict(global_config)
            )
            opt_result = self._optimization_orchestrator.run_optimization(
                unseen_canon, monthly_ut, daily_ut, rets_ut, args
            )
            stitched = self._optimization_result_to_series(opt_result)
            metrics = self._metrics_from_returns(stitched, daily_ut, global_config)
            return UnseenValidationResult(
                selected_protocol=selected,
                metrics=metrics,
                returns=stitched,
                mode=mode.value,
                trade_history=None,
            )

        scen_dict = unseen_scenario_dict_fixed_params(scenario_config, ut, selected)
        unseen_canon = self._normalizer.normalize(
            scenario=scen_dict, global_config=dict(global_config)
        )
        study_name = getattr(args, "study_name", None)
        bt_payload = self._backtest_runner.run_backtest_mode(
            unseen_canon, monthly_ut, daily_ut, rets_ut, study_name=study_name
        )
        rets_bt = bt_payload.get("returns")
        stitched = pd.Series(dtype=float)
        if isinstance(rets_bt, pd.Series):
            stitched = rets_bt
        metrics = self._metrics_from_returns(stitched, daily_ut, global_config)
        return UnseenValidationResult(
            selected_protocol=selected,
            metrics=metrics,
            returns=stitched,
            mode=mode.value,
            trade_history=_trade_history_from_backtest_payload(bt_payload),
        )

    def _run_cost_sensitivity_rows(
        self,
        *,
        scenario_config: CanonicalScenarioConfig,
        protocol_config: DoubleOOSWFOProtocolConfig,
        selected: SelectedProtocol,
        monthly_ut: pd.DataFrame,
        daily_ut: pd.DataFrame,
        rets_ut: pd.DataFrame,
        args: argparse.Namespace,
        global_config: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Re-run unseen validation for each cost grid cell (does not change selection)."""

        cs = protocol_config.cost_sensitivity
        cells = expand_cost_sensitivity_grid(cs)
        survival = survival_metric_for_selection(protocol_config.selection.metric)
        rows: list[dict[str, Any]] = []
        for slip, mult in cells:
            eff = effective_global_config_for_cost_cell(
                global_config, slippage_bps=slip, commission_multiplier=mult
            )
            unseen = self._run_final_unseen(
                scenario_config=scenario_config,
                protocol_config=protocol_config,
                selected=selected,
                monthly_ut=monthly_ut,
                daily_ut=daily_ut,
                rets_ut=rets_ut,
                args=args,
                global_config=eff,
            )
            row: dict[str, Any] = {
                "slippage_bps": slip,
                "commission_multiplier": mult,
                "survives": row_survives(unseen.metrics, survival),
            }
            for k, v in unseen.metrics.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    fv = float("nan")
                row[str(k)] = fv
            rows.append(row)
        return rows
