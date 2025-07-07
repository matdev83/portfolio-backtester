import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table
import os
from datetime import datetime

from ..reporting.performance_metrics import calculate_metrics

def display_results(self, daily_data_for_display):
    logger = self.logger
    logger.info("Generating performance report.")
    console = Console()

    benchmark_ticker = self.global_config["benchmark"]
    if isinstance(daily_data_for_display.columns, pd.MultiIndex):
        if benchmark_ticker in daily_data_for_display.columns.get_level_values(0):
            if ('Close' in daily_data_for_display[benchmark_ticker].columns):
                bench_prices = daily_data_for_display[(benchmark_ticker, 'Close')]
            elif ('Adj Close' in daily_data_for_display[benchmark_ticker].columns):
                 bench_prices = daily_data_for_display[(benchmark_ticker, 'Adj Close')]
            else:
                logger.error(f"Could not find 'Close' or 'Adj Close' for benchmark {benchmark_ticker} in multi-indexed data.")
                bench_prices = pd.Series(dtype=float)
        elif benchmark_ticker in daily_data_for_display.columns.get_level_values(1):
            if ('Close', benchmark_ticker) in daily_data_for_display.columns:
                bench_prices = daily_data_for_display[('Close', benchmark_ticker)]
            elif ('Adj Close', benchmark_ticker) in daily_data_for_display.columns:
                bench_prices = daily_data_for_display[('Adj Close', benchmark_ticker)]
            else:
                logger.error(f"Could not find ('Close'/'Adj Close', {benchmark_ticker}) in multi-indexed data.")
                bench_prices = pd.Series(dtype=float)
        else:
            logger.error(f"Benchmark ticker {benchmark_ticker} not found in multi-indexed columns.")
            bench_prices = pd.Series(dtype=float)
    else:
        if benchmark_ticker in daily_data_for_display.columns:
            bench_prices = daily_data_for_display[benchmark_ticker]
        else:
            logger.error(f"Benchmark ticker {benchmark_ticker} not found in single-level column data.")
            bench_prices = pd.Series(dtype=float)

    bench_rets_full = bench_prices.pct_change(fill_method=None).fillna(0)

    first_result_key = list(self.results.keys())[0]
    train_end_date = self.results[first_result_key].get("train_end_date")

    periods_data = []
    num_trials_full = {name: res.get("num_trials_for_dsr", 1) for name, res in self.results.items()}

    if train_end_date:
        periods_data.append({
            "title": "In-Sample Performance (Net of Costs)",
            "returns_map": {name: res["returns"][res["returns"].index <= train_end_date] for name, res in self.results.items()},
            "bench_returns": bench_rets_full[bench_rets_full.index <= train_end_date],
            "num_trials_map": {name: 1 for name in self.results.keys()}
        })
        periods_data.append({
            "title": "Out-of-Sample Performance (Net of Costs)",
            "returns_map": {name: res["returns"][res["returns"].index > train_end_date] for name, res in self.results.items()},
            "bench_returns": bench_rets_full[bench_rets_full.index > train_end_date],
            "num_trials_map": num_trials_full
        })

    periods_data.append({
        "title": "Full Period Performance (Net of Costs)",
        "returns_map": {name: res["returns"] for name, res in self.results.items()},
        "bench_returns": bench_rets_full,
        "num_trials_map": num_trials_full
    })

    for period_data in periods_data:
        _generate_performance_table(
            self,
            console,
            period_data["returns_map"],
            period_data["bench_returns"],
            period_data["title"],
            period_data["num_trials_map"]
        )
    
    logger.info("Performance tables displayed.")
    _plot_performance_summary(self, bench_rets_full, train_end_date)

def _generate_performance_table(self, console: Console, period_returns: dict,
                                  bench_period_rets: pd.Series, title: str,
                                  num_trials_map: dict):
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)

    bench_metrics = calculate_metrics(bench_period_rets, bench_period_rets, self.global_config["benchmark"], name=self.global_config["benchmark"], num_trials=1)

    all_period_metrics = {self.global_config["benchmark"]: bench_metrics}

    for name in period_returns.keys():
        display_name = self.results[name]["display_name"]
        table.add_column(display_name, style="magenta")
    table.add_column(self.global_config["benchmark"], style="green")

    for name, rets in period_returns.items():
        display_name = self.results[name]["display_name"]
        strategy_num_trials = num_trials_map.get(name, 1)
        metrics = calculate_metrics(rets, bench_period_rets, self.global_config["benchmark"], name=display_name, num_trials=strategy_num_trials)
        all_period_metrics[display_name] = metrics

    percentage_metrics = ["Total Return", "Ann. Return"]
    high_precision_metrics = ["ADF p-value"]

    if not bench_metrics.empty:
        for metric_name in bench_metrics.index:
            row_values = [metric_name]
            for strategy_name_key in period_returns.keys():
                display_name = self.results[strategy_name_key]["display_name"]
                value = all_period_metrics[display_name].loc[metric_name]
                if metric_name in percentage_metrics:
                    row_values.append(f"{value:.2%}")
                elif metric_name in high_precision_metrics:
                    row_values.append(f"{value:.6f}")
                else:
                    row_values.append(f"{value:.4f}")
            
            bench_value = bench_metrics.loc[metric_name]
            if metric_name in percentage_metrics:
                row_values.append(f"{bench_value:.2%}")
            elif metric_name in high_precision_metrics:
                row_values.append(f"{bench_value:.6f}")
            else:
                row_values.append(f"{bench_value:.4f}")
            table.add_row(*row_values)
    
    console.print(table)

def _plot_performance_summary(self, bench_rets_full: pd.Series, train_end_date: pd.Timestamp | None):
    logger = self.logger
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    ax1.set_title("Cumulative Returns (Net of Costs)", fontsize=16)
    ax1.set_ylabel("Cumulative Returns (Log Scale)", fontsize=12)
    ax1.set_yscale('log')
    
    all_cumulative_returns_plotting = []
    for name, result_data in self.results.items():
        cumulative_strategy_returns = (1 + result_data["returns"]).cumprod()
        cumulative_strategy_returns.plot(ax=ax1, label=result_data["display_name"])
        all_cumulative_returns_plotting.append(cumulative_strategy_returns)
    
    cumulative_bench_returns = (1 + bench_rets_full).cumprod()
    cumulative_bench_returns.plot(ax=ax1, label=self.global_config["benchmark"], linestyle='--')
    all_cumulative_returns_plotting.append(cumulative_bench_returns)

    if all_cumulative_returns_plotting:
        combined_cumulative_returns = pd.concat(all_cumulative_returns_plotting)
        max_val = combined_cumulative_returns.max().max()
        min_val = combined_cumulative_returns.min().min()
        ax1.set_ylim(bottom=min(0.9, min_val * 0.9) if min_val > 0 else 0.1, top=max_val * 1.1)

    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    if train_end_date and (self.args.mode == "optimize" or self.args.mode == "backtest"):
        ax1.axvline(train_end_date, color='gray', linestyle='--', lw=2, label='Train/Test Split')
        ax1.legend()

    ax2.set_ylabel("Drawdown", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)

    def calculate_drawdown(returns_series):
        cumulative = (1 + returns_series).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        return drawdown

    for name, result_data in self.results.items():
        drawdown = calculate_drawdown(result_data["returns"])
        drawdown.plot(ax=ax2, label=result_data["display_name"])
    
    bench_drawdown = calculate_drawdown(bench_rets_full)
    bench_drawdown.plot(ax=ax2, label=self.global_config["benchmark"], linestyle='--')

    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.fill_between(bench_drawdown.index, 0, bench_drawdown, color='gray', alpha=0.2)

    plt.tight_layout()
    
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = self.args.scenario_name if self.args.scenario_name else list(self.results.keys())[0]
    scenario_name_for_filename = base_filename.replace(" ", "_").replace("(", "").replace(")", "")

    filename = f"performance_summary_{scenario_name_for_filename}_{timestamp}.png"
    filepath = os.path.join(plots_dir, filename)
    
    plt.savefig(filepath)
    logger.info(f"Performance plot saved to: {filepath}")

    if getattr(self.args, 'interactive', False):
        plt.show(block=False)
        logger.info("Performance plots displayed interactively.")
    else:
        plt.close(fig)
        logger.info("Performance plots generated and saved.")