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

import os
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console

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


def _benchmark_returns(daily_data_for_display: pd.DataFrame, benchmark_ticker: str) -> pd.Series:
    """Extract benchmark price series and convert to returns."""
    if isinstance(daily_data_for_display.columns, pd.MultiIndex):
        prices = daily_data_for_display.xs(
            (benchmark_ticker, "Close"), level=("Ticker", "Field"), axis=1
        )
    else:
        prices = daily_data_for_display[benchmark_ticker]

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
        for name, result in self.results.items():
            best_trial = result.get("best_trial_obj")
            if best_trial is None:
                continue

            _plot_stability_measures(self, name, best_trial, result["returns"])

            opt_params = result.get("optimal_params")
            if opt_params is None:
                continue

            # locate matching scenario config
            scenario_cfg = next((s for s in self.scenarios if s["name"] in name), None)
            if scenario_cfg is not None:
                _plot_monte_carlo_robustness_analysis(
                    self,
                    name,
                    scenario_cfg,
                    opt_params,
                    self.monthly_data,
                    daily_data_for_display,
                    self.rets_full,
                )
    else:
        logger.info("Advanced optimization reports disabled – skipping.")

    # ------------------------------------------------------------------
    # 2) Always show basic performance table & plots
    # ------------------------------------------------------------------
    try:
        period_returns = {n: d["returns"] for n, d in self.results.items()}
        benchmark_rets = _benchmark_returns(daily_data_for_display, self.global_config["benchmark"])
        trials_map = {n: d.get("num_trials_for_dsr", 1) for n, d in self.results.items()}

        # unique report directory per run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Handle scenario name: use args.scenario_name if available, otherwise derive from scenario config or results
        scenario_name = None
        if hasattr(self.args, "scenario_name") and self.args.scenario_name:
            scenario_name = self.args.scenario_name
        elif self.scenarios and len(self.scenarios) > 0:
            # Use the first scenario's name
            scenario_name = self.scenarios[0].get("name", "unknown_scenario")
        elif self.results:
            # Use the first result key, removing any "_Optimized" suffix
            first_result_name = next(iter(self.results.keys()))
            scenario_name = first_result_name.replace("_Optimized", "")
        else:
            scenario_name = "unknown_scenario"

        scenario_slug = (
            scenario_name.replace(" ", "_").replace("(", "").replace(")", "").replace('"', "")
        )
        report_dir = os.path.join("data", "reports", f"{scenario_slug}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)

        _generate_performance_table(
            self,
            console,
            period_returns,
            benchmark_rets,
            "Full-Period Performance",
            trials_map,
            report_dir,
        )
        _generate_transaction_history_csv(self.results, report_dir)

        _plot_performance_summary(self, benchmark_rets, None)
        plt.tight_layout()
        plot_path = os.path.join(report_dir, "equity_curve.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info("Performance summary saved to %s", plot_path)

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
                output_file = os.path.join(report_dir, f"price_with_trades_{name}_{sym}.png")
                _plot_price_with_trades(
                    self,
                    daily_data_for_display,
                    trade_hist,
                    sym,
                    output_file,
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
