import logging
import os
from rich.console import Console
from rich.table import Table
import pandas as pd
from ..reporting.performance_metrics import calculate_metrics

logger = logging.getLogger(__name__)

def generate_performance_table(backtester, console: Console, period_returns: dict,
                                  bench_period_rets: pd.Series, title: str,
                                  num_trials_map: dict, report_dir: str):
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)

    bench_metrics = calculate_metrics(bench_period_rets, bench_period_rets, backtester.global_config["benchmark"], name=backtester.global_config["benchmark"], num_trials=1)

    all_period_metrics = {backtester.global_config["benchmark"]: bench_metrics}

    for name in period_returns.keys():
        display_name = backtester.results[name]["display_name"]
        table.add_column(display_name, style="magenta")
    table.add_column(backtester.global_config["benchmark"], style="green")

    for name, rets in period_returns.items():
        display_name = backtester.results[name]["display_name"]
        strategy_num_trials = num_trials_map.get(name, 1)
        metrics = calculate_metrics(rets, bench_period_rets, backtester.global_config["benchmark"], name=display_name, num_trials=strategy_num_trials)
        all_period_metrics[display_name] = metrics

    percentage_metrics = ["Total Return", "Ann. Return"]
    high_precision_metrics = ["ADF p-value"]

    if not bench_metrics.empty:
        for metric_name in bench_metrics.index:
            row_values = [metric_name]
            for strategy_name_key in period_returns.keys():
                display_name = backtester.results[strategy_name_key]["display_name"]
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

    all_metrics_df = pd.DataFrame(all_period_metrics)
    metrics_path = os.path.join(report_dir, "performance_metrics.csv")
    all_metrics_df.to_csv(metrics_path)
    logger.info("Performance metrics saved to %s", metrics_path)

    for name in period_returns.keys():
        result_data = backtester.results.get(name, {})
        optimal_params = result_data.get("optimal_params")
        display_name = result_data.get("display_name", name)
        constraint_status = result_data.get("constraint_status", "UNKNOWN")

        if optimal_params:
            params_path = os.path.join(report_dir, f"optimal_params_{name}.txt")
            with open(params_path, "w") as f:
                for param, value in optimal_params.items():
                    f.write(f"{param}: {value}\n")
            logger.info("Optimal parameters for %s saved to %s", name, params_path)

            title_suffix = ""
            if constraint_status == "VIOLATED":
                title_suffix = " ⚠️ CONSTRAINT VIOLATION"
            elif constraint_status == "ADJUSTED":
                title_suffix = " ✅ CONSTRAINT ADJUSTED"
            elif constraint_status == "FALLBACK_OK":
                title_suffix = " ⚠️ FALLBACK USED"
            
            params_table = Table(
                title=f"Optimal Parameters for {display_name}{title_suffix}",
                show_header=True,
                header_style="bold magenta"
            )
            params_table.add_column("Parameter", style="cyan")
            params_table.add_column("Value", style="green")
            
            for param, value in optimal_params.items():
                params_table.add_row(str(param), str(value))
            
            constraints_config = result_data.get("constraints_config", [])
            if constraints_config:
                params_table.add_row("", "")
                params_table.add_row("Constraint Status", constraint_status)
                
                constraint_message = result_data.get("constraint_message", "")
                if constraint_message:
                    if len(constraint_message) > 50:
                        words = constraint_message.split()
                        lines = []
                        current_line = []
                        for word in words:
                            if len(" ".join(current_line + [word])) <= 50:
                                current_line.append(word)
                            else:
                                lines.append(" ".join(current_line))
                                current_line = [word]
                        if current_line:
                            lines.append(" ".join(current_line))
                        
                        for i, line in enumerate(lines):
                            if i == 0:
                                params_table.add_row("Constraint Details", line)
                            else:
                                params_table.add_row("", line)
                    else:
                        params_table.add_row("Constraint Details", constraint_message)
            
            console.print(params_table)