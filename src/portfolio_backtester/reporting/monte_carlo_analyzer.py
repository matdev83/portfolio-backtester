import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..optimization.results import OptimizationResult
from ..interfaces.attribute_accessor_interface import (
    IAttributeAccessor,
    create_attribute_accessor,
)

from .parameter_analysis import _plot_parameter_impact_analysis

logger = logging.getLogger(__name__)


def plot_stability_measures(
    backtester,
    scenario_name: str,
    optimization_result: OptimizationResult,
    optimal_returns: pd.Series,
    attribute_accessor: Optional[IAttributeAccessor] = None,
):
    """
    Create a Monte Carlo-style visualization showing P&L curves from all optimization trials.

    Safe by design: if insufficient history or incompatible trial objects, it logs and skips without error.
    """
    logger = backtester.logger

    try:
        optimization_history = getattr(optimization_result, "optimization_history", None)
        if not optimization_history or len(optimization_history) < 2:
            logger.info(
                "Skipping trial P&L visualization: insufficient optimization history (need >=2)."
            )
            return

        trial_returns_data = []
        for trial in optimization_history:
            # Expect dict-like entries created by our orchestrators; ignore raw FrozenTrial objects
            if not isinstance(trial, dict):
                continue
            metrics = trial.get("metrics") or {}
            returns_dict = metrics.get("trial_returns")
            if not returns_dict:
                continue
            try:
                dates = pd.to_datetime(returns_dict["dates"])  # expect list-like structure
                returns = pd.Series(returns_dict["returns"], index=dates)
            except Exception:
                continue

            trial_value = trial.get("objective_value")
            if trial_value is None:
                continue

            trial_returns_data.append(
                {
                    "trial_number": trial.get("evaluation"),
                    "returns": returns,
                    "params": trial.get("parameters", {}),
                    "value": float(trial_value),
                }
            )

        if len(trial_returns_data) < 2:
            logger.info("Skipping trial P&L visualization: <2 trials with stored returns data.")
            return

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Creating Monte Carlo-style trial P&L visualization with {len(trial_returns_data)} trials..."
            )

        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots(figsize=(14, 8))

        all_cumulative_returns = []
        trial_values = []

        for trial_data in trial_returns_data:
            returns = trial_data["returns"]
            cumulative_returns = (1 + returns).cumprod()
            all_cumulative_returns.append(cumulative_returns)
            trial_values.append(trial_data["value"])

            ax.plot(
                cumulative_returns.index,
                cumulative_returns.values,
                color="lightgray",
                alpha=0.3,
                linewidth=0.8,
                zorder=1,
            )

        if isinstance(optimal_returns, pd.Series) and not optimal_returns.empty:
            optimal_cumulative: pd.Series = (1 + optimal_returns).cumprod()
            ax.plot(
                optimal_cumulative.index,
                optimal_cumulative.values,
                color="black",
                linewidth=2.5,
                label="Optimized Strategy",
                zorder=3,
            )

        if len(all_cumulative_returns) >= 5:
            common_start = max(cr.index.min() for cr in all_cumulative_returns)
            common_end = min(cr.index.max() for cr in all_cumulative_returns)

            aligned_series = []
            for cr in all_cumulative_returns:
                aligned = cr.loc[common_start:common_end]
                if len(aligned) > 10:
                    aligned_series.append(aligned)

            if len(aligned_series) >= 5:
                aligned_data = {}
                for i, series in enumerate(aligned_series):
                    series_reset = series.reset_index(drop=True)
                    aligned_data[f"trial_{i}"] = series_reset

                aligned_df = pd.DataFrame(aligned_data)

                percentile_5 = aligned_df.quantile(0.05, axis=1)
                percentile_95 = aligned_df.quantile(0.95, axis=1)
                median = aligned_df.median(axis=1)

                common_index = aligned_series[0].index[: len(aligned_df)]

                ax.fill_between(
                    common_index,
                    percentile_5.values.astype(float),
                    percentile_95.values.astype(float),
                    alpha=0.2,
                    color="blue",
                    label="90% Confidence Band",
                    zorder=2,
                )
                ax.plot(
                    common_index,
                    median.values.astype(float),
                    color="blue",
                    linewidth=1.5,
                    linestyle="--",
                    label="Median Trial Performance",
                    zorder=2,
                )
        ax.set_title(
            f"Optimization Trial P&L Curves: {scenario_name}",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Cumulative Returns", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        stats_text = f"""Trial Statistics:
Total Trials: {len(trial_returns_data)}
Best Trial Value: {max(trial_values):.3f}
Worst Trial Value: {min(trial_values):.3f}
Median Trial Value: {np.median(trial_values):.3f}
Std Dev of Values: {np.std(trial_values):.3f}"""

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10,
            fontfamily="monospace",
        )

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trial_pnl_curves_{scenario_name}_{timestamp}.png"
        filepath = os.path.join("plots", filename)

        os.makedirs("plots", exist_ok=True)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Trial P&L curves plot saved to: {filepath}")

        advanced_reporting_config = backtester.global_config.get("advanced_reporting_config", {})
        if advanced_reporting_config.get("enable_advanced_parameter_analysis", False):
            # Dependency injection for attribute access (DIP)
            _attribute_accessor = attribute_accessor or create_attribute_accessor()
            best_trial_obj = _attribute_accessor.get_attribute(
                optimization_result, "best_trial", None
            )
            if best_trial_obj is not None:
                _plot_parameter_impact_analysis(
                    backtester, scenario_name, best_trial_obj, timestamp
                )
            else:
                logger.info("No best_trial available for parameter analysis.")
        else:
            logger.info(
                "Advanced parameter analysis is disabled. Skipping hyperparameter correlation/sensitivity analysis."
            )

    except Exception as e:
        logger.error(f"Error creating trial P&L visualization: {e}")
        try:
            import traceback

            logger.debug(traceback.format_exc())
        except Exception:
            pass
