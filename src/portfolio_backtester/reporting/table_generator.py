import logging
import os
from rich.console import Console
from rich.table import Table
import pandas as pd
from ..reporting.performance_metrics import calculate_metrics

logger = logging.getLogger(__name__)


def _format_metric_value(metric_name: str, value):
    percentage_metrics = ["Total Return", "Ann. Return"]
    high_precision_metrics = ["ADF p-value"]
    for direction in ["All", "Long", "Short"]:
        if metric_name == f"Win Rate % ({direction})":
            return f"{value:.2f}%"
    if metric_name in percentage_metrics:
        return f"{value:.2%}"
    if metric_name in high_precision_metrics:
        return f"{value:.6f}"
    for direction in ["All", "Long", "Short"]:
        if metric_name in {
            f"Total P&L Net ({direction})",
            f"Largest Single Profit ({direction})",
            f"Largest Single Loss ({direction})",
            f"Mean Profit ({direction})",
            f"Mean Loss ({direction})",
            f"Mean Trade P&L ({direction})",
            f"Commissions Paid ({direction})",
            f"Avg MFE ({direction})",
            f"Avg MAE ({direction})",
        }:
            return f"${value:,.2f}"
    integer_metrics = {"Max DD Recovery Time (days)", "Max Flat Period (days)"}
    for direction in ["All", "Long", "Short"]:
        integer_metrics |= {
            f"Number of Trades ({direction})",
            f"Number of Winners ({direction})",
            f"Number of Losers ({direction})",
            f"Min Trade Duration Days ({direction})",
            f"Max Trade Duration Days ({direction})",
        }
    if metric_name in integer_metrics:
        return "N/A" if pd.isna(value) else f"{int(value)}"
    if "Margin Load" in metric_name:
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"


def _add_metrics_rows(
    table: Table,
    bench_metrics: pd.Series,
    period_returns: dict,
    backtester,
    all_period_metrics: dict,
):
    if bench_metrics.empty:
        return
    for metric_name in bench_metrics.index:
        row_values = [metric_name]
        for strategy_name_key in period_returns.keys():
            display_name = backtester.results[strategy_name_key]["display_name"]
            value = all_period_metrics[display_name].loc[metric_name]
            row_values.append(_format_metric_value(metric_name, value))
        bench_value = bench_metrics.loc[metric_name]
        row_values.append(_format_metric_value(metric_name, bench_value))
        table.add_row(*row_values)


def _collect_metrics(
    backtester, period_returns: dict, bench_period_rets: pd.Series, num_trials_map: dict
) -> dict:
    bench_name = backtester.global_config["benchmark"]
    bench_metrics = calculate_metrics(
        bench_period_rets, bench_period_rets, bench_name, name=bench_name, num_trials=1
    )
    all_period_metrics = {bench_name: bench_metrics}
    for name, rets in period_returns.items():
        display_name = backtester.results[name]["display_name"]
        strategy_num_trials = num_trials_map.get(name, 1)
        trade_stats = None
        if hasattr(backtester, "results") and name in backtester.results:
            trade_stats = backtester.results[name].get("trade_stats")
        metrics = calculate_metrics(
            rets,
            bench_period_rets,
            bench_name,
            name=display_name,
            num_trials=strategy_num_trials,
            trade_stats=trade_stats,
        )
        all_period_metrics[display_name] = metrics
    return all_period_metrics


def _save_metrics_csv(all_period_metrics: dict, report_dir: str):
    all_metrics_df = pd.DataFrame(all_period_metrics)
    metrics_path = os.path.join(report_dir, "performance_metrics.csv")
    all_metrics_df.to_csv(metrics_path)
    logger.info("Performance metrics saved to %s", metrics_path)


def _print_params_tables(backtester, console: Console, period_returns: dict, report_dir: str):
    for name in period_returns.keys():
        result_data = backtester.results.get(name, {})
        optimal_params = result_data.get("optimal_params")
        display_name = result_data.get("display_name", name)
        constraint_status = result_data.get("constraint_status", "UNKNOWN")
        if not optimal_params:
            continue
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
            header_style="bold magenta",
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
                lines: list[str] = []
                current_line: list[str] = []
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


def generate_performance_table(
    backtester,
    console: Console,
    period_returns: dict,
    bench_period_rets: pd.Series,
    title: str,
    num_trials_map: dict,
    report_dir: str,
):
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)
    all_period_metrics = _collect_metrics(
        backtester, period_returns, bench_period_rets, num_trials_map
    )
    for name in period_returns.keys():
        display_name = backtester.results[name]["display_name"]
        table.add_column(display_name, style="magenta")
    table.add_column(backtester.global_config["benchmark"], style="green")
    bench_metrics = all_period_metrics[backtester.global_config["benchmark"]]
    _add_metrics_rows(table, bench_metrics, period_returns, backtester, all_period_metrics)
    console.print(table)
    for name, rets in period_returns.items():
        display_name = backtester.results[name]["display_name"]
        if hasattr(backtester, "results") and name in backtester.results:
            trade_stats = backtester.results[name].get("trade_stats")
            if trade_stats:
                generate_trade_statistics_table(console, display_name, trade_stats)
    _save_metrics_csv(all_period_metrics, report_dir)
    _print_params_tables(backtester, console, period_returns, report_dir)


def generate_trade_statistics_table(console: Console, strategy_name: str, trade_stats: dict):
    """Generate a detailed trade statistics table with All/Long/Short columns."""
    if not trade_stats:
        return

    # Check if we have any trades
    if trade_stats.get("all_num_trades", 0) == 0:
        console.print(f"\n[yellow]No trades found for {strategy_name}[/yellow]")
        return

    # Create trade statistics table
    trade_table = Table(
        title=f"Trade Statistics - {strategy_name}", show_header=True, header_style="bold magenta"
    )

    trade_table.add_column("Metric", style="cyan", no_wrap=True)
    trade_table.add_column("All Trades", style="white", justify="right")
    trade_table.add_column("Long Trades", style="green", justify="right")
    trade_table.add_column("Short Trades", style="red", justify="right")

    # Define metrics to display
    metrics_config = [
        ("num_trades", "Number of Trades", "int"),
        ("num_winners", "Number of Winners", "int"),
        ("num_losers", "Number of Losers", "int"),
        ("win_rate_pct", "Win Rate (%)", "pct"),
        ("total_pnl_net", "Total P&L Net", "currency"),
        ("largest_profit", "Largest Single Profit", "currency"),
        ("largest_loss", "Largest Single Loss", "currency"),
        ("mean_profit", "Mean Profit", "currency"),
        ("mean_loss", "Mean Loss", "currency"),
        ("mean_trade_pnl", "Mean Trade P&L", "currency"),
        ("reward_risk_ratio", "Reward/Risk Ratio", "ratio"),
        ("total_commissions_paid", "Commissions Paid", "currency"),
        ("avg_mfe", "Avg MFE", "currency"),
        ("avg_mae", "Avg MAE", "currency"),
        ("min_trade_duration_days", "Min Duration (days)", "int"),
        ("max_trade_duration_days", "Max Duration (days)", "int"),
        ("mean_trade_duration_days", "Mean Duration (days)", "float"),
        ("information_score", "Information Score", "ratio"),
        ("trades_per_month", "Trades per Month", "float"),
    ]

    # Add rows for each metric
    for metric_key, metric_name, format_type in metrics_config:
        row_values = [metric_name]

        for direction in ["all", "long", "short"]:
            value = trade_stats.get(f"{direction}_{metric_key}", 0)

            # Format the value based on type
            if format_type == "int":
                formatted_value = f"{int(value)}"
            elif format_type == "pct":
                formatted_value = f"{value:.2f}%"
            elif format_type == "currency":
                formatted_value = f"${value:,.2f}"
            elif format_type == "ratio":
                if value == float("inf"):
                    formatted_value = "∞"
                elif value == float("-inf"):
                    formatted_value = "-∞"
                else:
                    formatted_value = f"{value:.3f}"
            elif format_type == "float":
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)

            row_values.append(formatted_value)

        trade_table.add_row(*row_values)

    # Add portfolio-level metrics (apply to all columns)
    portfolio_metrics = [
        ("max_margin_load", "Max Margin Load", "pct"),
        ("mean_margin_load", "Mean Margin Load", "pct"),
    ]

    if portfolio_metrics:
        # Add separator
        trade_table.add_row("", "", "", "")

        for metric_key, metric_name, format_type in portfolio_metrics:
            value = trade_stats.get(metric_key, 0)

            if format_type == "pct":
                formatted_value = f"{value * 100:.2f}%"
            else:
                formatted_value = f"{value:.4f}"

            # Portfolio metrics apply to all directions
            trade_table.add_row(metric_name, formatted_value, formatted_value, formatted_value)

    console.print("\n")
    console.print(trade_table)


def generate_enhanced_performance_table(
    backtester,
    console: Console,
    period_returns: dict,
    bench_period_rets: pd.Series,
    title: str,
    num_trials_map: dict,
    report_dir: str,
):
    """Enhanced version that separates main performance metrics from trade statistics."""
    # Generate main performance table (excluding trade metrics)
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)

    bench_metrics = calculate_metrics(
        bench_period_rets,
        bench_period_rets,
        backtester.global_config["benchmark"],
        name=backtester.global_config["benchmark"],
        num_trials=1,
    )

    all_period_metrics = {backtester.global_config["benchmark"]: bench_metrics}

    for name in period_returns.keys():
        display_name = backtester.results[name]["display_name"]
        table.add_column(display_name, style="magenta")
    table.add_column(backtester.global_config["benchmark"], style="green")

    for name, rets in period_returns.items():
        display_name = backtester.results[name]["display_name"]
        strategy_num_trials = num_trials_map.get(name, 1)

        # Get trade statistics if available
        trade_stats = None
        if hasattr(backtester, "results") and name in backtester.results:
            trade_stats = backtester.results[name].get("trade_stats")

        metrics = calculate_metrics(
            rets,
            bench_period_rets,
            backtester.global_config["benchmark"],
            name=display_name,
            num_trials=strategy_num_trials,
            trade_stats=trade_stats,
        )
        all_period_metrics[display_name] = metrics

    # Define which metrics are trade-related and should be excluded from main table
    trade_metric_patterns = [
        "Number of Trades",
        "Win Rate %",
        "Number of Winners",
        "Number of Losers",
        "Total P&L Net",
        "Largest Single Profit",
        "Largest Single Loss",
        "Mean Profit",
        "Mean Loss",
        "Mean Trade P&L",
        "Reward/Risk Ratio",
        "Commissions Paid",
        "Avg MFE",
        "Avg MAE",
        "Information Score",
        "Min Trade Duration",
        "Max Trade Duration",
        "Mean Trade Duration",
        "Trades per Month",
        "Max Margin Load",
        "Mean Margin Load",
    ]

    percentage_metrics = ["Total Return", "Ann. Return"]
    high_precision_metrics = ["ADF p-value"]

    if not bench_metrics.empty:
        for metric_name in bench_metrics.index:
            # Skip trade-related metrics for main table
            if any(pattern in metric_name for pattern in trade_metric_patterns):
                continue

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

    # Generate trade statistics tables for each strategy
    for name, rets in period_returns.items():
        display_name = backtester.results[name]["display_name"]
        if hasattr(backtester, "results") and name in backtester.results:
            trade_stats = backtester.results[name].get("trade_stats")
            if trade_stats:
                generate_trade_statistics_table(console, display_name, trade_stats)

    # Save metrics to CSV
    all_metrics_df = pd.DataFrame(all_period_metrics)
    metrics_path = os.path.join(report_dir, "performance_metrics.csv")
    all_metrics_df.to_csv(metrics_path)
    logger.info("Performance metrics saved to %s", metrics_path)

    # Generate parameter tables (existing code)
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
                header_style="bold magenta",
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
                        lines: list[str] = []
                        current_line: list[str] = []
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


def generate_transaction_history_csv(backtest_results: dict, report_dir: str):
    """Generate CSV file with transaction history for all strategies.

    Args:
        backtest_results: Dictionary of backtest results
        report_dir: Directory to save the CSV file
    """
    for name, result_data in backtest_results.items():
        if "trade_history" not in result_data or result_data["trade_history"].empty:
            continue

        trade_history = result_data["trade_history"]
        transactions = []

        # Process each trade to create entry and exit transactions
        for _, trade in trade_history.iterrows():
            ticker = trade["ticker"]
            quantity = trade["quantity"]
            entry_date = trade["entry_date"]
            exit_date = trade["exit_date"]
            entry_price = trade["entry_price"]
            exit_price = trade["exit_price"]
            commission_entry = trade["commission_entry"]
            commission_exit = trade["commission_exit"]
            pnl_net = trade["pnl_net"]

            # Determine trade types
            if quantity > 0:
                entry_type = "BTO"  # Buy To Open
                exit_type = "STC"  # Sell To Close
            else:
                entry_type = "STO"  # Sell To Open
                exit_type = "BTC"  # Buy To Close

            # Add entry transaction
            transactions.append(
                {
                    "datetime": entry_date,
                    "symbol": ticker,
                    "trade_type": entry_type,
                    "amount_of_shares": abs(quantity),
                    "trade_price": entry_price,
                    "calculated_commissions": commission_entry,
                    "profit_loss": None,  # No P&L for entry trades
                }
            )

            # Add exit transaction
            if pd.notna(exit_date) and exit_price is not None:
                transactions.append(
                    {
                        "datetime": exit_date,
                        "symbol": ticker,
                        "trade_type": exit_type,
                        "amount_of_shares": abs(quantity),
                        "trade_price": exit_price,
                        "calculated_commissions": commission_exit,
                        "profit_loss": pnl_net,  # P&L only for exit trades
                    }
                )

        # Create DataFrame and save to CSV
        if transactions:
            transactions_df = pd.DataFrame(transactions)
            csv_path = os.path.join(report_dir, f"transaction_history_{name}.csv")
            transactions_df.to_csv(csv_path, index=False)
            logger.info("Transaction history for %s saved to %s", name, csv_path)
