"""Stage-2 Monte-Carlo robustness analysis helpers.

Copied verbatim from the original monolithic `backtester_logic.reporting` so
behaviour is unchanged; only the physical file location differs.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# NOTE: relative import still works because we are inside
# `portfolio_backtester.reporting` – the same level as `monte_carlo`.
from ..monte_carlo.asset_replacement import AssetReplacementManager  # noqa: E402

__all__ = [
    "_plot_monte_carlo_robustness_analysis",
    "_create_monte_carlo_robustness_plot",
]


def _plot_monte_carlo_robustness_analysis(
    self,
    scenario_name: str,
    scenario_config: dict,
    optimal_params: dict,
    monthly_data,
    daily_data,
    rets_full,
):
    """Stage-2 comprehensive stress-test after optimisation completes."""
    logger = self.logger

    try:
        monte_carlo_config = self.global_config.get("monte_carlo_config", {})
        if not monte_carlo_config.get("enable_synthetic_data", False):
            logger.warning(
                "Stage 2 MC: Synthetic data generation is disabled. Cannot create robustness analysis."
            )
            return
        if not monte_carlo_config.get("enable_stage2_stress_testing", True):
            logger.info(
                "Stage 2 MC: Stage 2 stress testing is disabled for faster optimization. Skipping robustness analysis."
            )
            return

        universe = scenario_config.get("universe", self.global_config.get("universe", []))
        universe_size = len(universe)

        base_percentages = [0.05, 0.075, 0.10]
        replacement_counts = [max(1, int(np.ceil(universe_size * p))) for p in base_percentages]
        replacement_percentages = [c / universe_size for c in replacement_counts]

        num_simulations_per_level = monte_carlo_config.get("num_simulations_per_level", 10)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        optimized_scenario = scenario_config.copy()
        optimized_scenario["strategy_params"] = optimal_params

        simulation_results: dict[float, list[pd.Series]] = {}
        total_sims = len(replacement_percentages) * num_simulations_per_level


        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Stage 2 Monte Carlo Stress Testing…", total=total_sims)

            timeout_manager = getattr(self, 'timeout_manager', None)
            timed_out = False

            for rep_pct in replacement_percentages:
                if timed_out:
                    logger.warning("Stage 2 MC: Timeout detected, aborting further simulations.")
                    break
                level_results: list[pd.Series] = []
                for sim in range(num_simulations_per_level):
                    # Check for timeout before each simulation
                    if timeout_manager is not None and timeout_manager.check_timeout():
                        logger.warning("Stage 2 MC: Timeout detected during simulation (rep_pct=%.3f, sim=%d). Aborting Stage 2 MC.", rep_pct, sim)
                        timed_out = True
                        break
                    try:
                        # naive synthetic: bootstrap returns with noise
                        synthetic = rets_full.copy()
                        rng = np.random.default_rng(sim + int(rep_pct * 1000))
                        assets_to_replace = rng.choice(universe, max(1, int(len(universe) * rep_pct)), replace=False)
                        for ticker in assets_to_replace:
                            if ticker in synthetic.columns:
                                orig = synthetic[ticker].dropna()
                                if len(orig) < 50:
                                    continue
                                boot = rng.choice(orig.to_numpy(), size=len(orig))
                                noise = rng.normal(0, orig.std() * rng.uniform(0.15, 0.25), len(orig))
                                synthetic[ticker] = pd.Series(boot + noise, index=orig.index)

                        sim_rets = self.run_scenario(
                            optimized_scenario,
                            monthly_data,
                            daily_data,
                            synthetic,
                            verbose=False,
                        )
                        level_results.append(sim_rets)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Stage 2 MC simulation failed: %s", exc)
                    finally:
                        progress.advance(task)

                if level_results:
                    simulation_results[rep_pct] = level_results

        original_returns = self.run_scenario(
            optimized_scenario, monthly_data, daily_data, rets_full, verbose=False
        )

        if simulation_results:
            _create_monte_carlo_robustness_plot(
                self,
                scenario_name,
                simulation_results,
                replacement_percentages,
                colors,
                optimal_params,
                original_returns,
            )
        else:
            logger.warning("Stage 2 MC: No simulation results available for robustness plot")

    except Exception as exc:  # noqa: BLE001
        logger.error("Stage 2 MC error: %s", exc)


def _create_monte_carlo_robustness_plot(
    self,
    scenario_name: str,
    simulation_results: dict,
    replacement_percentages: list[float],
    colors: list[str],
    optimal_params: dict,
    original_strategy_returns: pd.Series | None = None,
):
    """Visualise robustness results produced by `_plot_monte_carlo_robustness_analysis`."""
    logger = self.logger

    try:
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, (ax, ax_params) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [5, 1]})

        for i, rep_pct in enumerate(replacement_percentages):
            for j, sim_rets in enumerate(simulation_results.get(rep_pct, [])):
                if sim_rets is None or sim_rets.empty:
                    continue
                cum = (1 + sim_rets).cumprod()
                ax.plot(
                    cum.index,
                    cum.values,
                    color=colors[i % len(colors)],
                    alpha=0.6 if j else 0.8,
                    linewidth=1.5 if j == 0 else 1.0,
                    label=f"{rep_pct:.1%} Replacement" if j == 0 else None,
                )

        if original_strategy_returns is not None and not original_strategy_returns.empty:
            cum_orig = (1 + original_strategy_returns).cumprod()
            ax.plot(cum_orig.index, cum_orig.values, color="black", linewidth=3.0, label="Original Strategy", zorder=10)

        ax.set_title(
            f"Monte Carlo Robustness Analysis: {scenario_name}", fontsize=16, fontweight="bold", pad=20
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

        # parameter summary in second subplot
        ax_params.axis("off")
        key_params = ["lookback_months", "num_holdings", "atr_length", "atr_multiple", "leverage"]
        summary = ", ".join(f"{p}={optimal_params[p]}" for p in key_params if p in optimal_params)
        ax_params.text(0.5, 0.5, summary, ha="center", va="center", fontsize=10, wrap=True)

        plt.tight_layout()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"monte_carlo_robustness_{scenario_name.replace(' ', '_')}_{ts}.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(os.path.join("plots", fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Monte Carlo robustness plot saved: %s", fname)

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to create robustness plot: %s", exc) 