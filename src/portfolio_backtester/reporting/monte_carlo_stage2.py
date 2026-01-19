"""Stage-2 Monte-Carlo robustness analysis helpers.

Copied verbatim from the original monolithic `backtester_logic.reporting` so
behaviour is unchanged; only the physical file location differs.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..numba_optimized import generate_ohlc_from_prices_fast


# Removed multiprocessing to ensure reliability across platforms
# Stage 2 MC now runs sequentially to avoid pickling/serialization issues
# and to provide deterministic progress and logs
# from joblib import Parallel, delayed  # type: ignore
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ..interfaces.attribute_accessor_interface import IAttributeAccessor

# NOTE: relative import still works because we are inside
# `portfolio_backtester.reporting` – the same level as `monte_carlo`.

__all__ = [
    "_plot_monte_carlo_robustness_analysis",
    "_create_monte_carlo_robustness_plot",
]


def _plot_series(ax, x, y, color, alpha, lw, label) -> None:
    try:
        ax.plot(x, y, color=color, alpha=alpha, linewidth=lw, label=label)
    except Exception:
        pass


def _plot_original_series(ax, series: pd.Series | None) -> None:
    if series is None:
        return
    if series.empty:
        return
    cum: pd.Series = (1 + series).cumprod()
    ax.plot(
        cum.index,
        cum.values,
        color="black",
        linewidth=3.0,
        label="Original Strategy",
        zorder=10,
    )


def _block_bootstrap_returns(
    returns: pd.Series,
    target_length: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block-bootstrap returns to preserve short-term autocorrelation."""
    if target_length <= 0:
        return np.array([], dtype=float)

    clean = returns.dropna()
    if clean.empty:
        return np.array([], dtype=float)

    values = clean.to_numpy(dtype=float)
    block_size = max(1, min(int(block_size), len(values)))
    samples: list[float] = []

    while len(samples) < target_length:
        start = int(rng.integers(0, len(values)))
        end = start + block_size
        if end <= len(values):
            block = values[start:end]
        else:
            wrap = end - len(values)
            block = np.concatenate([values[start:], values[:wrap]])
        samples.extend(block.tolist())

    return np.asarray(samples[:target_length], dtype=float)


def _apply_synthetic_prices(
    daily_data: pd.DataFrame,
    ticker: str,
    synthetic_prices: pd.Series,
    random_seed: int,
) -> None:
    """Replace daily OHLC (or close) prices with synthetic series in-place."""
    if isinstance(daily_data.columns, pd.MultiIndex):
        ohlc = generate_ohlc_from_prices_fast(
            synthetic_prices.to_numpy(dtype=float), random_seed=random_seed
        )
        synthetic_ohlc = pd.DataFrame(
            ohlc,
            columns=["Open", "High", "Low", "Close"],
            index=synthetic_prices.index,
        )
        for field in ["Open", "High", "Low", "Close"]:
            col = (ticker, field)
            if col in daily_data.columns:
                daily_data.loc[synthetic_ohlc.index, col] = synthetic_ohlc[field].values
    else:
        if ticker in daily_data.columns:
            daily_data.loc[synthetic_prices.index, ticker] = synthetic_prices.values


def _plot_monte_carlo_robustness_analysis(
    self,
    scenario_name: str,
    scenario_config: dict,
    optimal_params: dict,
    monthly_data,
    daily_data,
    rets_full,
    attribute_accessor: Optional[IAttributeAccessor] = None,
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

        # Resolve a universe consistent with this scenario (do not fall back to global universe
        # unless the scenario doesn't define one). Falling back to GLOBAL_CONFIG.universe can
        # inflate the replacement counts and skew the stress test.
        universe: list[str] = []
        available_tickers: list[str] = []
        try:
            if isinstance(monthly_data, pd.DataFrame) and not monthly_data.empty:
                available_tickers = [str(c) for c in monthly_data.columns if isinstance(c, str)]
        except Exception:
            available_tickers = []

        if isinstance(scenario_config.get("universe"), list) and scenario_config.get("universe"):
            universe = list(scenario_config["universe"])
        elif "universe_config" in scenario_config:
            try:
                from ..universe_resolver import resolve_universe_config

                current_date = None
                try:
                    if isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                        current_date = pd.Timestamp(daily_data.index.max())
                except Exception:
                    current_date = None

                universe = resolve_universe_config(
                    scenario_config["universe_config"], current_date=current_date
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Stage 2 MC: failed to resolve scenario universe_config: {exc}")
                universe = []

        # Exclude the benchmark from the replacement pool if present.
        benchmark_ticker = self.global_config.get("benchmark")
        if isinstance(benchmark_ticker, str) and benchmark_ticker:
            universe = [t for t in universe if t != benchmark_ticker]

        # Prefer the tickers actually present in the backtest data before falling back to the
        # global universe. This avoids skewing the replacement percentages and eliminates
        # "tickers not found" warnings during Stage 2 runs.
        if not universe and available_tickers:
            if isinstance(benchmark_ticker, str) and benchmark_ticker:
                universe = [t for t in available_tickers if t != benchmark_ticker]
            else:
                universe = list(available_tickers)

        if not universe:
            universe = list(self.global_config.get("universe", []))
            if isinstance(benchmark_ticker, str) and benchmark_ticker:
                universe = [t for t in universe if t != benchmark_ticker]

        universe_size = len(universe)
        if universe_size <= 0:
            logger.warning(
                "Stage 2 MC: empty universe after resolution; skipping robustness analysis."
            )
            return

        base_percentages = [0.05, 0.075, 0.10]
        replacement_counts = [max(1, int(np.ceil(universe_size * p))) for p in base_percentages]
        replacement_percentages = [c / universe_size for c in replacement_counts]

        num_simulations_per_level = monte_carlo_config.get("num_simulations_per_level", 10)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        optimized_scenario = scenario_config.copy()
        optimized_scenario["strategy_params"] = optimal_params
        if available_tickers:
            optimized_scenario["universe"] = list(available_tickers)
        elif universe:
            optimized_scenario["universe"] = (
                [benchmark_ticker, *universe]
                if isinstance(benchmark_ticker, str) and benchmark_ticker
                else list(universe)
            )

        simulation_results: dict[float, list[pd.Series]] = {}
        total_sims = len(replacement_percentages) * num_simulations_per_level

        def _run_simulation_task(rep_pct: float, sim_seed: int) -> Optional[pd.Series]:
            """Worker function for a single MC simulation, designed for joblib."""
            try:
                rng = np.random.default_rng(sim_seed)

                synthetic_daily = daily_data.copy(deep=True)
                if isinstance(daily_data.columns, pd.MultiIndex):
                    if "Close" in daily_data.columns.get_level_values("Field"):
                        close_prices = daily_data.xs("Close", level="Field", axis=1)
                    else:
                        close_prices = daily_data
                else:
                    close_prices = daily_data

                target_index = close_prices.index
                block_size = int(monte_carlo_config.get("stage2_block_size_days", 20))

                assets_to_replace = rng.choice(
                    universe, max(1, int(len(universe) * rep_pct)), replace=False
                )

                for ticker in assets_to_replace:
                    if ticker not in close_prices.columns:
                        continue

                    close_series = close_prices[ticker].dropna()
                    if close_series.empty:
                        continue

                    if isinstance(rets_full, pd.DataFrame) and ticker in rets_full.columns:
                        returns_source = rets_full[ticker].dropna()
                    else:
                        returns_source = close_series.pct_change(fill_method=None).dropna()

                    if len(returns_source) < max(5, block_size):
                        continue

                    synthetic_returns = _block_bootstrap_returns(
                        returns_source, len(target_index), block_size, rng
                    )
                    start_price = float(close_series.iloc[0])
                    synthetic_prices = pd.Series(
                        start_price * np.cumprod(1 + synthetic_returns),
                        index=target_index,
                    )

                    _apply_synthetic_prices(
                        synthetic_daily, ticker, synthetic_prices, random_seed=sim_seed
                    )

                return cast(
                    Optional[pd.Series],
                    self.run_scenario(
                        optimized_scenario,
                        monthly_data,
                        synthetic_daily,
                        None,
                        verbose=False,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Stage 2 MC simulation failed (rep_pct={rep_pct}): {exc}")
                return None

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Stage 2 Monte Carlo Stress Testing…", total=total_sims)

            # Create a list of all simulation jobs to run
            simulation_jobs = [
                (rep_pct, sim + int(rep_pct * 1000))
                for rep_pct in replacement_percentages
                for sim in range(num_simulations_per_level)
            ]

            logger.info(f"Running {total_sims} Stage 2 MC simulations sequentially...")

            simulation_results = {rep_pct: [] for rep_pct in replacement_percentages}
            for rep_pct, seed in simulation_jobs:
                result = _run_simulation_task(rep_pct, seed)
                progress.advance(task)
                if result is not None:
                    simulation_results[rep_pct].append(result)

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
        fig, (ax, ax_params) = plt.subplots(
            2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [5, 1]}
        )

        for i, rep_pct in enumerate(replacement_percentages):
            for j, sim_rets in enumerate(simulation_results.get(rep_pct, [])):
                if sim_rets is None or sim_rets.empty:
                    continue
                cum = (1 + sim_rets).cumprod()
                if not hasattr(cum, "index") or not hasattr(cum, "values"):
                    continue
                x = cum.index
                y_arr = np.asarray(cum.values)
                if y_arr.size == 0:
                    continue
                _plot_series(
                    ax,
                    x,
                    y_arr,
                    colors[i % len(colors)],
                    0.6 if j else 0.8,
                    1.5 if j == 0 else 1.0,
                    f"{rep_pct:.1%} Replacement" if j == 0 else None,
                )

        _plot_original_series(ax, original_strategy_returns)

        ax.set_title(
            f"Monte Carlo Robustness Analysis: {scenario_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

        # parameter summary in second subplot
        ax_params.axis("off")
        key_params = [
            "lookback_months",
            "num_holdings",
            "atr_length",
            "atr_multiple",
            "leverage",
        ]
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
