import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_performance_summary(
    backtester, bench_rets_full: pd.Series, train_end_date: pd.Timestamp | None
):
    logger = backtester.logger
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    ax1.set_title("Cumulative Returns (Net of Costs)", fontsize=16)
    ax1.set_ylabel("Cumulative Returns (Log Scale)", fontsize=12)
    ax1.set_yscale("log")

    all_cumulative_returns_plotting = []
    for name, result_data in backtester.results.items():
        cumulative_strategy_returns = (1 + result_data["returns"]).cumprod()
        cumulative_strategy_returns.plot(ax=ax1, label=result_data["display_name"])
        all_cumulative_returns_plotting.append(cumulative_strategy_returns)

        ath_mask = cumulative_strategy_returns.expanding().max() == cumulative_strategy_returns
        ath_dates = cumulative_strategy_returns.index[ath_mask]
        ath_values = cumulative_strategy_returns[ath_mask]
        ax1.scatter(ath_dates, ath_values, color="green", s=20, alpha=0.7, zorder=5)

        cumulative_bench_returns: pd.Series = (1 + bench_rets_full).cumprod()
    cumulative_bench_returns.plot(
        ax=ax1, label=backtester.global_config["benchmark"], linestyle="--"
    )

    all_cumulative_returns_plotting.append(cumulative_bench_returns)

    bench_ath_mask = cumulative_bench_returns.expanding().max() == cumulative_bench_returns
    bench_ath_dates = cumulative_bench_returns.index[bench_ath_mask]
    bench_ath_values = cumulative_bench_returns[bench_ath_mask]
    ax1.scatter(bench_ath_dates, bench_ath_values, color="green", s=20, alpha=0.7, zorder=5)

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
    bench_drawdown.plot(ax=ax2, label=backtester.global_config["benchmark"], linestyle="--")

    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.fill_between(bench_drawdown.index, 0, bench_drawdown, color="gray", alpha=0.2)

    plt.tight_layout()

    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = (
        backtester.args.scenario_name
        if backtester.args.scenario_name
        else list(backtester.results.keys())[0]
    )
    scenario_name_for_filename = (
        base_filename.replace(" ", "_").replace("(", "").replace(")", "").replace('"', "")
    )

    filename = f"performance_summary_{scenario_name_for_filename}_{timestamp}.png"
    filepath = os.path.join(plots_dir, filename)

    plt.savefig(filepath)
    if logger.isEnabledFor(logging.INFO):

        logger.info(f"Performance plot saved to: {filepath}")

    # Display plot interactively if requested, but silence warnings when using non-interactive backends such as Agg.
    if getattr(backtester.args, "interactive", False):
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.filterwarnings(
                "ignore",
                message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
                category=UserWarning,
            )
            plt.show(block=False)
        logger.info("Performance plots displayed interactively.")
    else:
        # Close the figure to free memory when not displaying it.
        plt.close(fig)
        logger.info("Performance plots generated and saved (no interactive display).")

    plt.close("all")


def _extract_price_series(daily_data: pd.DataFrame, symbol: str) -> pd.Series | None:
    """Helper to extract closing price series for a symbol from potential MultiIndex/flat DataFrame."""
    if daily_data is None or symbol not in str(daily_data.columns):
        # quick check to skip heavy operations if symbol not present
        pass  # will attempt more thorough extraction below

    try:
        # Case 1: MultiIndex with (Ticker, Field)
        if isinstance(daily_data.columns, pd.MultiIndex):
            if (symbol, "Close") in daily_data.columns:
                series = daily_data[(symbol, "Close")]
            else:
                # fall back: take first level match then close field by position 0 maybe
                level0_matches = [col for col in daily_data.columns if col[0] == symbol]
                if level0_matches:
                    # prefer Close if exists
                    close_cols = [col for col in level0_matches if col[1].lower() == "close"]
                    col_to_use = close_cols[0] if close_cols else level0_matches[0]
                    series = daily_data[col_to_use]
                else:
                    return None
        else:
            # Case 2: Flat columns where each column is already close price per symbol
            if symbol in daily_data.columns:
                series = daily_data[symbol]
            else:
                return None
    except Exception:
        return None

    # Ensure Series type
    try:
        df = pd.DataFrame(series)
        if not df.empty:
            series = df.iloc[:, 0]
    except Exception:
        pass
    return series


def plot_price_with_trades(
    backtester,
    daily_data: pd.DataFrame,
    trade_history: pd.DataFrame,
    symbol: str,
    output_path: str,
    interactive: bool = False,
):
    """Generate price chart with trade entry/exit markers for a single symbol.

    Args:
        backtester: Backtester instance for logging.
        daily_data: Daily price data (OHLC with potentially MultiIndex columns).
        trade_history: DataFrame containing trade history (must include entry_date, exit_date, entry_price, exit_price, quantity, ticker).
        symbol: The symbol to plot.
        output_path: Full path where the PNG will be saved.
        interactive: If True, show the plot interactively.
    """
    logger = backtester.logger
    price_series = _extract_price_series(daily_data, symbol)
    if price_series is None or price_series.empty:
        logger.warning("Price series for %s not found – skipping trade plot.", symbol)
        return

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(14, 6))

    price_series.plot(ax=ax, label=f"{symbol} Close", color="black")

    # Plot trade markers
    # Filter trade_history for the symbol
    symbol_trades = trade_history[trade_history["ticker"] == symbol]

    # Helper to add scatter
    def _scatter(date, price, marker, color, label):
        ax.scatter(date, price, marker=marker, color=color, s=60, alpha=0.8, label=label)

    plotted_labels = set()
    for _, trade in symbol_trades.iterrows():
        qty = trade["quantity"] if "quantity" in trade else trade.get("position", 0)
        # Some trade_history implementations may not have entry_price / exit_price if missing; fallback to price_series.loc[date]
        entry_date = trade.get("entry_date") or trade.get("date")
        exit_date = trade.get("exit_date")
        entry_price = trade.get("entry_price")
        exit_price = trade.get("exit_price")

        if pd.isna(entry_price) and entry_date is not None and entry_date in price_series.index:
            entry_price = price_series.loc[entry_date]
        if pd.isna(exit_price) and exit_date in price_series.index:
            exit_price = price_series.loc[exit_date] if exit_date is not None else None

        if qty > 0:
            entry_marker, entry_color, entry_lbl = "^", "green", "Long Entry"
            exit_marker, exit_color, exit_lbl = "v", "red", "Sell to Close"
        else:
            entry_marker, entry_color, entry_lbl = "v", "orange", "Short Entry"
            exit_marker, exit_color, exit_lbl = "^", "blue", "Buy to Cover"

        if entry_date is not None and entry_price is not None:
            lbl = entry_lbl if entry_lbl not in plotted_labels else None
            _scatter(entry_date, entry_price, entry_marker, entry_color, lbl)
            plotted_labels.add(entry_lbl)
        if exit_date is not None and exit_price is not None:
            lbl = exit_lbl if exit_lbl not in plotted_labels else None
            _scatter(exit_date, exit_price, exit_marker, exit_color, lbl)
            plotted_labels.add(exit_lbl)

    ax.set_title(f"{symbol} Price with Trade Markers")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    logger.info("Trade price plot saved to %s", output_path)

    # Display plot interactively if requested, but silence warnings from non-interactive backends
    if interactive:
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.filterwarnings(
                "ignore",
                message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
                category=UserWarning,
            )
            plt.show(block=False)
    # Always close the figure to free resources
    plt.close(fig)
