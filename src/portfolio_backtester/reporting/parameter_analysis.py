"""Parameter-impact visualisation helpers.

Extracted untouched from the legacy `backtester_logic.reporting` implementation
so that the behaviour remains identical.  The only change is the module path.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from ..core import Backtester

__all__ = [
    "_plot_parameter_impact_analysis",
    "_create_parameter_heatmaps",
    "_create_parameter_sensitivity_analysis",
    "_create_parameter_stability_analysis",
    "_create_parameter_correlation_analysis",
    "_create_parameter_importance_ranking",
    "_create_parameter_robustness_analysis",
]

# ---------------------------------------------------------------------------
# The following blocks are verbatim copies of the corresponding functions in
# the original 1 700-line file.  Only long docstrings / comments trimmed.
# ---------------------------------------------------------------------------


def _plot_parameter_impact_analysis(
    self: "Backtester", scenario_name: str, best_trial_obj: Any, timestamp: str
) -> None:
    logger = self.logger
    try:
        if not hasattr(best_trial_obj, "study") or best_trial_obj.study is None:
            logger.warning("No study object found. Cannot create parameter impact analysis.")
            return
        study = best_trial_obj.study
        completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
        if len(completed_trials) < 10:
            logger.warning("Need ≥10 completed trials for meaningful parameter analysis.")
            return

        param_data = []
        for t in completed_trials:
            params = t.params.copy()
            try:
                val = t.value
            except RuntimeError:
                val = t.values[0] if t.values else 0.0
            params["objective_value"] = val
            params["trial_number"] = t.number
            param_data.append(params)

        df = pd.DataFrame(param_data)
        param_names = [c for c in df.columns if c not in ["objective_value", "trial_number"]]
        if len(param_names) < 2:
            logger.warning("Need at least 2 parameters for analysis.")
            return

        _create_parameter_heatmaps(self, df, param_names, scenario_name, timestamp)
        _create_parameter_sensitivity_analysis(self, df, param_names, scenario_name, timestamp)
        _create_parameter_stability_analysis(self, df, param_names, scenario_name, timestamp)
        _create_parameter_correlation_analysis(self, df, param_names, scenario_name, timestamp)
        _create_parameter_importance_ranking(self, df, param_names, scenario_name, timestamp)
        _create_parameter_robustness_analysis(self, df, param_names, scenario_name, timestamp)
        logger.info("Parameter impact analysis completed successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Error creating parameter impact analysis: %s", exc)


# ---------------------------- heat-maps ------------------------------------


def _create_parameter_heatmaps(
    self: "Backtester", df: pd.DataFrame, param_names: list[str], scenario_name: str, timestamp: str
) -> None:
    logger = self.logger
    try:
        if len(param_names) < 2:
            return
        num_pairs = min(6, len(param_names) * (len(param_names) - 1) // 2)
        if num_pairs == 0:
            return
        cols = min(3, num_pairs)
        rows = (num_pairs + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        pairs = []
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                p1, p2 = param_names[i], param_names[j]
                imp = abs(df[p1].corr(df["objective_value"])) + abs(
                    df[p2].corr(df["objective_value"])
                )
                pairs.append((p1, p2, imp))
        pairs.sort(key=lambda x: x[2], reverse=True)
        pairs = pairs[:num_pairs]

        for idx, (p1, p2, _) in enumerate(pairs):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            try:
                v1, v2 = df[p1].copy(), df[p2].copy()
                if len(v1.unique()) > 10:
                    v1 = pd.cut(v1, bins=10, precision=2)
                if len(v2.unique()) > 10:
                    v2 = pd.cut(v2, bins=10, precision=2)
                pivot = (
                    df.groupby([v1, v2], observed=True)["objective_value"]
                    .mean()
                    .unstack(fill_value=None)
                )
                # Convert None to NaN after unstacking
                pivot = pivot.fillna(np.nan)
                sns.heatmap(
                    pivot,
                    annot=True,
                    fmt=".3f",
                    cmap="viridis",
                    ax=ax,
                    cbar_kws={"label": "Objective"},
                )
                ax.set_title(f"{p1} vs {p2}")
                ax.set_xlabel(p2)
                ax.set_ylabel(p1)
                ax.tick_params(axis="x", rotation=45)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Heatmap failed for %s vs %s: %s", p1, p2, exc)
                ax.text(0.5, 0.5, "heatmap failed", ha="center", va="center")

        for idx in range(num_pairs, rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].set_visible(False)

        plt.suptitle(f"Parameter Heatmaps: {scenario_name}")
        plt.tight_layout()
        fname = f"parameter_heatmaps_{scenario_name}_{timestamp}.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(os.path.join("plots", fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Parameter heatmaps saved: %s", fname)
    except Exception as exc:
        logger.error("Error creating heatmaps: %s", exc)


# ---------------------- sensitivity analysis --------------------------------


def _create_parameter_sensitivity_analysis(
    self: "Backtester", df: pd.DataFrame, param_names: list[str], scenario_name: str, timestamp: str
) -> None:
    logger = self.logger
    try:
        if not param_names:
            return
        cols = min(3, len(param_names))
        rows = (len(param_names) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, param in enumerate(param_names):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            try:
                x = df[param]
                y = df["objective_value"]
                ax.scatter(x, y, alpha=0.6, s=30, color="steelblue")
                z = np.polyfit(x, y, 1)
                ax.plot(sorted(x), np.poly1d(z)(sorted(x)), "r--", alpha=0.8)
                correlation = x.corr(y)
                ax.text(0.05, 0.95, f"corr={correlation:.3f}", transform=ax.transAxes, va="top")
                ax.set_title(f"Sensitivity: {param}")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Sensitivity plot failed for %s: %s", param, exc)
                ax.text(0.5, 0.5, "plot failed", ha="center", va="center")
        for idx in range(len(param_names), rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].set_visible(False)
        plt.suptitle(f"Parameter Sensitivity: {scenario_name}")
        plt.tight_layout()
        fname = f"parameter_sensitivity_{scenario_name}_{timestamp}.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(os.path.join("plots", fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Parameter sensitivity saved: %s", fname)
    except Exception as exc:
        logger.error("Error creating sensitivity analysis: %s", exc)


# ---------------------- stability analysis ----------------------------------


def _create_parameter_stability_analysis(
    self: "Backtester", df: pd.DataFrame, param_names: list[str], scenario_name: str, timestamp: str
) -> None:
    logger = self.logger
    try:
        if not param_names:
            return
        fig = plt.figure(figsize=(16, 10))
        ax1 = plt.subplot(2, 2, (1, 2))
        for param in param_names[:6]:
            ax1.plot(
                df["trial_number"], df[param], marker="o", markersize=3, alpha=0.7, label=param
            )
        ax1.set_title("Parameter Evolution Over Trials")
        ax1.legend()

        ax2 = plt.subplot(2, 2, 3)
        variances = []
        numeric_param_names = []
        for p in param_names:
            try:
                # Check if the column is numeric
                if pd.api.types.is_numeric_dtype(df[p]):
                    var = df[p].var()
                    if pd.notna(var) and isinstance(var, (int, float, np.number)):
                        variances.append(float(var))
                        numeric_param_names.append(p)
            except (TypeError, AttributeError, ValueError):
                # Skip non-numeric parameters
                continue

        max_var = max(variances) if variances else 1.0
        ax2.bar(range(len(numeric_param_names)), [v / max_var for v in variances])
        ax2.set_xticks(range(len(numeric_param_names)))
        ax2.set_xticklabels(numeric_param_names, rotation=45, ha="right")
        ax2.set_title("Normalised Variance (lower = stable)")

        ax3 = plt.subplot(2, 2, 4)
        if len(param_names) >= 2:
            ax3.scatter(df[param_names[0]], df[param_names[1]], alpha=0.6)
            ax3.set_xlabel(param_names[0])
            ax3.set_ylabel(param_names[1])
        else:
            ax3.text(0.5, 0.5, "Need 2+ params", ha="center", va="center")
        plt.suptitle(f"Parameter Stability: {scenario_name}")
        plt.tight_layout()
        fname = f"parameter_stability_{scenario_name}_{timestamp}.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(os.path.join("plots", fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Parameter stability saved: %s", fname)
    except Exception as exc:
        logger.error("Error creating stability analysis: %s", exc)


# ---------------------- correlation analysis --------------------------------


def _create_parameter_correlation_analysis(
    self: "Backtester", df: pd.DataFrame, param_names: list[str], scenario_name: str, timestamp: str
) -> None:
    logger = self.logger
    try:
        if len(param_names) < 2:
            return
        corr = df[param_names + ["objective_value"]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="coolwarm", center=0)
        plt.title(f"Correlation Matrix – {scenario_name}")
        plt.tight_layout()
        fname = f"parameter_correlation_{scenario_name}_{timestamp}.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(os.path.join("plots", fname), dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Parameter correlation saved: %s", fname)
    except Exception as exc:
        logger.error("Error creating correlation analysis: %s", exc)


# ---------------------- importance ranking ----------------------------------


def _create_parameter_importance_ranking(
    self: "Backtester", df: pd.DataFrame, param_names: list[str], scenario_name: str, timestamp: str
) -> None:
    logger = self.logger
    try:
        if not param_names:
            return
        scores = {p: abs(df[p].corr(df["objective_value"])) for p in param_names}
        params, vals = zip(*sorted(scores.items(), key=lambda x: x[1], reverse=True))
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(params)), vals, color=sns.color_palette("viridis", len(params)))
        plt.xticks(range(len(params)), params, rotation=45, ha="right")
        plt.title(f"Parameter Importance – {scenario_name}")
        plt.tight_layout()
        fname = f"parameter_importance_{scenario_name}_{timestamp}.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(os.path.join("plots", fname), dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Parameter importance saved: %s", fname)
    except Exception as exc:
        logger.error("Error creating importance ranking: %s", exc)


# ---------------------- robustness analysis ---------------------------------


def _create_parameter_robustness_analysis(
    self: "Backtester", df: pd.DataFrame, param_names: list[str], scenario_name: str, timestamp: str
) -> None:
    logger = self.logger
    try:
        if not param_names:
            return
        mean_perf = df["objective_value"].mean()
        std_perf = df["objective_value"].std()
        high = mean_perf + 0.5 * std_perf
        low = mean_perf - 0.5 * std_perf

        if len(param_names) >= 2:
            p1, p2 = param_names[:2]
            x, y, z = df[p1], df[p2], df["objective_value"]
            xi = np.linspace(x.min(), x.max(), 50)
            yi = np.linspace(y.min(), y.max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            from scipy.interpolate import griddata

            Zi = griddata((x, y), z, (Xi, Yi), method="cubic", fill_value=np.nan)
            plt.figure(figsize=(8, 6))
            plt.contourf(Xi, Yi, Zi, levels=20, cmap="viridis", alpha=0.8)
            plt.colorbar(label="Objective")
            plt.contour(Xi, Yi, Zi, levels=[high], colors="red", linestyles="--")
            plt.contour(Xi, Yi, Zi, levels=[low], colors="blue", linestyles="--")
            plt.title(f"Parameter Robustness Landscape: {p1} vs {p2}")
            fname = f"parameter_robustness_{scenario_name}_{timestamp}.png"
            os.makedirs("plots", exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join("plots", fname), dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Parameter robustness saved: %s", fname)
    except Exception as exc:
        logger.error("Error creating robustness analysis: %s", exc)
