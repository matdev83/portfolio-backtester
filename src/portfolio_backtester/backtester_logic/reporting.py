"""portfolio_backtester.backtester_logic.reporting
================================================
Light-weight compatibility shim that preserves the original public API of
`backtester_logic.reporting` while delegating real work to the refactored
modules living in `portfolio_backtester.reporting`.

All helper functions that external code (including the test-suite) imports from
here are *re-exported* so that nothing breaks even though the monolithic file
has been split into smaller, maintainable pieces.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Mapping

import pandas as pd
from rich.console import Console
from pathlib import Path

# ---------------------------------------------------------------------------
# Import helpers from their new homes and expose them under the old names
# ---------------------------------------------------------------------------
from ..reporting.plot_generator import (
    plot_performance_summary as _plot_performance_summary,
    plot_price_with_trades as _plot_price_with_trades,
)
from ..reporting.table_generator import (
    generate_performance_table as _generate_performance_table,
    generate_transaction_history_csv as _generate_transaction_history_csv,
)
from ..reporting.monte_carlo_analyzer import (
    plot_stability_measures as _plot_stability_measures,
)
from ..reporting.monte_carlo_stage2 import (
    _plot_monte_carlo_robustness_analysis,
    _create_monte_carlo_robustness_plot,
)
from ..reporting.parameter_analysis import (
    _plot_parameter_impact_analysis,
    _create_parameter_heatmaps,
    _create_parameter_sensitivity_analysis,
    _create_parameter_stability_analysis,
    _create_parameter_correlation_analysis,
    _create_parameter_importance_ranking,
    _create_parameter_robustness_analysis,
)
from ..reporting.report_directory_utils import (
    create_report_directory,
    generate_content_hash,
)

__all__ = [
    # public API
    "display_results",
    # helper re-exports (kept for backwards-compatibility)
    "_plot_stability_measures",
    "_plot_parameter_impact_analysis",
    "_create_parameter_heatmaps",
    "_create_parameter_sensitivity_analysis",
    "_create_parameter_stability_analysis",
    "_create_parameter_correlation_analysis",
    "_create_parameter_importance_ranking",
    "_create_parameter_robustness_analysis",
    "_plot_monte_carlo_robustness_analysis",
    "_create_monte_carlo_robustness_plot",
    "_generate_performance_table",
    "_generate_transaction_history_csv",
    "_plot_performance_summary",
    "_plot_price_with_trades",
]

# ---------------------------------------------------------------------------
# Thin facade – identical external behaviour, minimal internal code
# ---------------------------------------------------------------------------


def _resolve_benchmark_ticker(backtester: Any) -> str:
    try:
        if hasattr(backtester, "scenarios") and len(backtester.scenarios) == 1:
            scenario = backtester.scenarios[0]
            bench = getattr(scenario, "benchmark_ticker", None)
            if bench:
                return str(bench)
            if hasattr(scenario, "get"):
                bench = scenario.get("benchmark_ticker")
                if bench:
                    return str(bench)
    except Exception:
        pass
    return str(backtester.global_config.get("benchmark", "SPY"))


def _benchmark_returns(daily_data_for_display: pd.DataFrame, benchmark_ticker: str) -> pd.Series:
    """Extract benchmark price series and convert to returns."""
    candidates = [benchmark_ticker]
    if ":" in benchmark_ticker:
        candidates.append(benchmark_ticker.split(":")[-1])

    prices = None
    if isinstance(daily_data_for_display.columns, pd.MultiIndex):
        tickers = daily_data_for_display.columns.get_level_values("Ticker")
        for candidate in candidates:
            if candidate in tickers and (candidate, "Close") in daily_data_for_display.columns:
                prices = daily_data_for_display[(candidate, "Close")]
                break
    else:
        for candidate in candidates:
            if candidate in daily_data_for_display.columns:
                prices = daily_data_for_display[candidate]
                break

    if prices is None:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Benchmark %s not found in price data; skipping benchmark returns.", benchmark_ticker
        )
        return pd.Series(dtype=float)

    rets = prices.pct_change(fill_method=None).fillna(0.0)
    return rets.iloc[:, 0] if isinstance(rets, pd.DataFrame) else rets


def display_results(self: Any, daily_data_for_display: pd.DataFrame) -> None:  # noqa: D401
    """Generate all textual & graphical reports for the finished backtest run."""
    logger = self.logger
    logger.info("Generating performance report …")
    console = Console()

    if not self.results:
        logger.warning("No results available – skipping report generation.")
        return

    # ------------------------------------------------------------------
    # 1) Optional optimization analysis
    # ------------------------------------------------------------------
    adv_cfg: Dict[str, Any] = self.global_config.get("advanced_reporting_config", {})
    if adv_cfg.get("enable_optimization_reports", True):
        stage2_processed_scenarios: set[str] = set()
        for name, result in self.results.items():
            logger.info("Processing optimization report for: %s", name)
            best_trial = result.get("best_trial_obj")
            if best_trial is None:
                logger.info("  - No best_trial_obj found, skipping.")
                continue

            _plot_stability_measures(self, name, best_trial, result["returns"])

            opt_params = result.get("optimal_params")
            if opt_params is None:
                continue

            # locate matching scenario config
            scenario_name = name.removesuffix("_Optimized")
            scenario_cfg = next((s for s in self.scenarios if s.get("name") == scenario_name), None)
            if scenario_cfg is not None:
                logger.info("  - Found matching scenario config.")
                # Stage 2 MC is disabled by default; enable only if CLI flag is set
                if getattr(self.args, "enable_stage2_mc", False):
                    if scenario_name in stage2_processed_scenarios:
                        logger.info(
                            "  - Stage 2 MC already generated for %s, skipping.", scenario_name
                        )
                    else:
                        stage2_processed_scenarios.add(scenario_name)
                        _plot_monte_carlo_robustness_analysis(
                            self,
                            scenario_name,
                            scenario_cfg,
                            opt_params,
                            self.monthly_data,
                            daily_data_for_display,
                            self.rets_full,
                        )
                else:
                    logger.info(
                        "  - Stage 2 MC is disabled. Use --enable-stage2-mc to generate robustness plots."
                    )
            else:
                logger.info("  - No matching scenario config found.")
    else:
        logger.info("Advanced optimization reports disabled – skipping.")

    # ------------------------------------------------------------------
    # 2) Always show basic performance table & plots
    # ------------------------------------------------------------------
    try:
        logger.info("--- Generating Basic Performance Table & Plots ---")
        period_returns = {n: d["returns"] for n, d in self.results.items()}
        benchmark_ticker = _resolve_benchmark_ticker(self)
        benchmark_rets = _benchmark_returns(daily_data_for_display, benchmark_ticker)
        trials_map = {n: d.get("num_trials_for_dsr", 1) for n, d in self.results.items()}

        # unique report directory per run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Handle scenario name: use args.scenario_name if available, otherwise derive from scenario config or results
        scenario_name = None
        if hasattr(self.args, "scenario_name") and self.args.scenario_name:
            scenario_name = self.args.scenario_name
            logger.info("Scenario name from args: %s", scenario_name)
        elif self.scenarios and len(self.scenarios) > 0:
            # Use first scenario's name
            scenario_name = self.scenarios[0].get("name", "unknown_scenario")
            logger.info("Scenario name from scenarios[0]: %s", scenario_name)
        elif self.results:
            # Use the first result key, removing any "_Optimized" suffix
            first_result_name = next(iter(self.results.keys()))
            scenario_name = first_result_name.replace("_Optimized", "")
            logger.info("Scenario name from results key: %s", scenario_name)
        else:
            scenario_name = "unknown_scenario"
            logger.warning("Could not determine scenario name, defaulting to 'unknown_scenario'.")

        scenario_slug = (
            scenario_name.replace(" ", "_").replace("(", "").replace(")", "").replace('"', "")
        )

        # Generate content hash for version tracking
        content_hash = None
        scenario_path = None
        strategy_class = None

        # Try to get scenario filename from args (for hash generation)
        if hasattr(self.args, "scenario_filename") and self.args.scenario_filename:
            scenario_path = Path(self.args.scenario_filename)

        # Try to get strategy class from scenario config
        if self.scenarios and len(self.scenarios) > 0:
            scenario_config = self.scenarios[0]
            strategy_name = scenario_config.get("strategy")
            if strategy_name:
                try:
                    from ..strategies._core.registry.registry.strategy_registry import (
                        get_strategy_registry,
                    )

                    registry = get_strategy_registry()
                    strategy_class = registry.get_strategy_class(strategy_name)
                    if strategy_class:
                        logger.debug(f"Found strategy class: {strategy_class.__name__}")
                except Exception as e:
                    logger.warning(f"Could not resolve strategy class for hash generation: {e}")

        # Generate hash if both sources are available
        if scenario_path or strategy_class:
            try:
                content_hash = generate_content_hash(
                    strategy_class=strategy_class, config_file_path=scenario_path
                )
                if content_hash:
                    logger.info(f"Generated content hash: {content_hash}")
            except Exception as e:
                logger.warning(f"Could not generate content hash: {e}")

        # Create report directory with hash-based structure
        report_dir = create_report_directory(
            Path("data") / "reports", scenario_slug, content_hash, timestamp
        )

        perf_title = "Full-Period Performance"
        if hasattr(self, "scenarios") and len(self.scenarios) == 1:
            scenario_cfg = self.scenarios[0]
            overlay_cfg = None
            if hasattr(scenario_cfg, "get"):
                overlay_cfg = scenario_cfg.get("risk_overlay_config")
            elif isinstance(scenario_cfg, dict):
                overlay_cfg = scenario_cfg.get("risk_overlay_config")
            if isinstance(overlay_cfg, Mapping) and overlay_cfg.get("metrics_window") == "wfo_test":
                perf_title = "WFO Test-Window Performance"

        _generate_performance_table(
            self,
            console,
            period_returns,
            benchmark_rets,
            perf_title,
            trials_map,
            str(report_dir),
        )
        _generate_transaction_history_csv(self.results, str(report_dir))

        plot_path = report_dir / "equity_curve.png"
        _plot_performance_summary(self, benchmark_rets, secondary_save_path=str(plot_path))

        # --------------------------------------------------------------
        # 3) Price charts with trade markers for strategies trading ≤2 symbols
        # --------------------------------------------------------------
        for name, result_data in self.results.items():
            trade_hist = result_data.get("trade_history")
            if trade_hist is None or trade_hist.empty:
                continue
            unique_symbols = trade_hist["ticker"].unique()
            if len(unique_symbols) > 2:
                continue  # Skip – too many symbols
            for sym in unique_symbols:
                output_file = report_dir / f"price_with_trades_{name}_{sym}.png"
                _plot_price_with_trades(
                    self,
                    daily_data_for_display,
                    trade_hist,
                    sym,
                    str(output_file),
                    getattr(self.args, "interactive", False),
                )

    except Exception as exc:  # noqa: BLE001
        # Check if this is the common scenario_name None error
        if "'NoneType' object has no attribute 'replace'" in str(exc):
            logger.error(
                "Optimization reporting failed: scenario name could not be determined. "
                "This typically indicates the optimization process did not complete successfully. "
                "Check for earlier error messages about data availability or parameter space issues."
            )
        else:
            logger.warning("Failed to generate summary table/plot: %s", exc)
