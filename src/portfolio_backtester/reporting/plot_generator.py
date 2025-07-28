import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def plot_performance_summary(backtester, bench_rets_full: pd.Series, train_end_date: pd.Timestamp | None):
    logger = backtester.logger
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    ax1.set_title("Cumulative Returns (Net of Costs)", fontsize=16)
    ax1.set_ylabel("Cumulative Returns (Log Scale)", fontsize=12)
    ax1.set_yscale('log')
    
    all_cumulative_returns_plotting = []
    for name, result_data in backtester.results.items():
        cumulative_strategy_returns = (1 + result_data["returns"]).cumprod()
        cumulative_strategy_returns.plot(ax=ax1, label=result_data["display_name"])
        all_cumulative_returns_plotting.append(cumulative_strategy_returns)
        
        ath_mask = cumulative_strategy_returns.expanding().max() == cumulative_strategy_returns
        ath_dates = cumulative_strategy_returns.index[ath_mask]
        ath_values = cumulative_strategy_returns[ath_mask]
        ax1.scatter(ath_dates, ath_values, color='green', s=20, alpha=0.7, zorder=5)
    
    cumulative_bench_returns = (1 + bench_rets_full).cumprod()
    cumulative_bench_returns.plot(ax=ax1, label=backtester.global_config["benchmark"], linestyle='--')
    all_cumulative_returns_plotting.append(cumulative_bench_returns)
    
    bench_ath_mask = cumulative_bench_returns.expanding().max() == cumulative_bench_returns
    bench_ath_dates = cumulative_bench_returns.index[bench_ath_mask]
    bench_ath_values = cumulative_bench_returns[bench_ath_mask]
    ax1.scatter(bench_ath_dates, bench_ath_values, color='green', s=20, alpha=0.7, zorder=5)

    if all_cumulative_returns_plotting:
        combined_cumulative_returns = pd.concat(all_cumulative_returns_plotting)
        max_val = combined_cumulative_returns.max().max()
        min_val = combined_cumulative_returns.min().min()
        ax1.set_ylim(bottom=min(0.9, min_val * 0.9) if min_val > 0 else 0.1, top=max_val * 1.1)

    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    ax2.set_ylabel("Drawdown", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)

    def calculate_drawdown(returns_series):
        cumulative = (1 + returns_series).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        return drawdown

    for name, result_data in backtester.results.items():
        drawdown = calculate_drawdown(result_data["returns"])
        drawdown.plot(ax=ax2, label=result_data["display_name"])
    
    bench_drawdown = calculate_drawdown(bench_rets_full)
    bench_drawdown.plot(ax=ax2, label=backtester.global_config["benchmark"], linestyle='--')

    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.fill_between(bench_drawdown.index, 0, bench_drawdown, color='gray', alpha=0.2)

    plt.tight_layout()
    
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = backtester.args.scenario_name if backtester.args.scenario_name else list(backtester.results.keys())[0]
    scenario_name_for_filename = base_filename.replace(" ", "_").replace("(", "").replace(")", "").replace('"', "")

    filename = f"performance_summary_{scenario_name_for_filename}_{timestamp}.png"
    filepath = os.path.join(plots_dir, filename)
    
    plt.savefig(filepath)
    if logger.isEnabledFor(logging.INFO):

        logger.info(f"Performance plot saved to: {filepath}")

    if getattr(backtester.args, 'interactive', False):
        plt.show(block=False)
        logger.info("Performance plots displayed interactively.")
    else:
        plt.close(fig)
        logger.info("Performance plots generated and saved.")
    
    plt.close('all')