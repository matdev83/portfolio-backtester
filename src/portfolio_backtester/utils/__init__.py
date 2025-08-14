import logging  # Assuming logger might be useful here or for other utils
import signal
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd


# Get a logger for this module (or use a more general one if available)
logger = logging.getLogger(__name__)

# Global flag to indicate if an interrupt signal (Ctrl+C) has been received.
INTERRUPTED = False


def handle_interrupt(signum, frame):
    """
    Signal handler for SIGINT (Ctrl+C).
    Sets the global INTERRUPTED flag and logs a message.
    """
    global INTERRUPTED
    INTERRUPTED = True
    # Using print as logger might not be configured when signal occurs early
    print("Interrupt signal received. Attempting to terminate gracefully...")
    logger.warning("Interrupt signal received. Attempting to terminate gracefully...")


def register_signal_handler():
    """Registers the interrupt handler for SIGINT."""
    try:
        signal.signal(signal.SIGINT, handle_interrupt)
        logger.debug("SIGINT handler registered successfully.")
    except Exception as e:
        # This might happen in environments where signal handling is restricted
        logger.error(f"Failed to register SIGINT handler: {e}")


def _resolve_strategy(name: object, strategy_params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Resolve strategy specification to a strategy class using polymorphic interfaces.

    Args:
        name: Strategy specification - can be string name, dict with strategy config, or other types
        strategy_params: Parameters for strategy initialization

    Returns:
        Strategy class if found, None otherwise

    Note:
        Uses polymorphic interfaces to eliminate isinstance violations while supporting
        dict specifications (with 'name', 'strategy', or 'type' keys) and string specifications.
    """
    from ..interfaces import create_strategy_resolver
    from ..interfaces.strategy_resolver_interface import PolymorphicStrategyResolver

    # Check if there are any mocked strategies for testing
    mocked_strategies = PolymorphicStrategyResolver.enumerate_strategies_with_params()
    if mocked_strategies:
        # For tests, use the mocked strategies
        if isinstance(name, dict):
            strategy_name = name.get("name") or name.get("strategy") or name.get("type")
            if strategy_name in mocked_strategies:
                return mocked_strategies[strategy_name]
        elif isinstance(name, str) and name in mocked_strategies:
            return mocked_strategies[name]

    # Use polymorphic strategy resolver to eliminate isinstance violations
    resolver = create_strategy_resolver()
    return resolver.resolve_strategy(name, strategy_params)


def _run_scenario_static(
    global_cfg,
    scenario_cfg,
    price_monthly,
    price_daily,
    rets_daily,
    benchmark_daily,
    features_slice,
):
    """Lightweight version of Backtester.run_scenario() suitable for Optuna.

    Parameters
    ----------
    price_monthly : pd.DataFrame
        Business-month-end close prices used for signal generation.
    price_daily : pd.DataFrame
        Daily close prices used for portfolio valuation.
    rets_daily : pd.DataFrame
        Daily returns (same index as *price_daily*).
    benchmark_daily : pd.Series
        Daily benchmark prices (needed by *calculate_metrics* in the caller).
    """

    from ..portfolio.rebalancing import rebalance

    strat_cls = _resolve_strategy(scenario_cfg["strategy"], scenario_cfg.get("strategy_params", {}))
    if not strat_cls:
        raise ValueError(f"Could not resolve strategy: {scenario_cfg['strategy']}")
    strategy = strat_cls

    # --- 1) Generate signals on monthly data --------------------------------
    universe_cols = [c for c in price_monthly.columns if c != global_cfg["benchmark"]]
    signals = strategy.generate_signals(
        price_monthly[universe_cols],
        features_slice,
        price_monthly[global_cfg["benchmark"]],
    )

    # Use strategy's position sizer provider instead of direct config access
    position_sizer_provider = strategy.get_position_sizer_provider()
    sizer = position_sizer_provider.get_position_sizer()
    sizer_config = position_sizer_provider.get_position_sizer_config()

    # Extract position sizer parameters from provider config
    sizer_params = {k: v for k, v in sizer_config.items() if k != "position_sizer"}

    sizer_kwargs = {
        "signals": signals,
        "prices": price_monthly[universe_cols],
        "benchmark": price_monthly[global_cfg["benchmark"]],
        **sizer_params,
    }

    sized = sizer.calculate_weights(**sizer_kwargs)
    weights_monthly = rebalance(sized, scenario_cfg["rebalance_frequency"])

    # --- 2) Expand weights to daily frequency -------------------------------
    weights_daily = weights_monthly.reindex(price_daily.index, method="ffill").fillna(0.0)

    # --- 3) Compute daily portfolio returns ---------------------------------
    gross = (weights_daily.shift(1).fillna(0.0) * rets_daily).sum(axis=1)
    turn = (weights_daily - weights_daily.shift(1)).abs().sum(axis=1)

    # Use realistic transaction costs (13 bps for liquid S&P 500 stocks)
    realistic_cost_bps = 13.0  # Conservative estimate for retail trading liquid large caps
    tc = turn * (realistic_cost_bps / 10_000)

    portfolio_returns = (gross - tc).reindex(price_daily.index).fillna(0)

    # Calculate benchmark-relative metrics if benchmark data is provided
    if benchmark_daily is not None and len(benchmark_daily) > 0:
        # Calculate benchmark returns for comparison
        benchmark_returns = benchmark_daily.pct_change().fillna(0)
        aligned_benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).fillna(0)

        # Calculate excess returns and tracking error
        excess_returns = portfolio_returns - aligned_benchmark_returns

        # Return enhanced data structure with benchmark information
        return {
            "portfolio_returns": portfolio_returns,
            "benchmark_returns": aligned_benchmark_returns,
            "excess_returns": excess_returns,
            "benchmark_daily": benchmark_daily.reindex(portfolio_returns.index),
        }

    return portfolio_returns


def _get_trading_days_in_month(year: int, month: int) -> pd.DatetimeIndex:
    """Helper to get all trading days in a given month."""
    start_of_month = pd.Timestamp(year=year, month=month, day=1)
    end_of_month = start_of_month + pd.offsets.MonthEnd(1)
    return pd.bdate_range(start=start_of_month, end=end_of_month)


def generate_randomized_wfo_windows(
    monthly_data_index: pd.DatetimeIndex,
    scenario_config: dict,
    global_config: dict,
    rng: np.random.Generator,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate walk-forward optimization windows with optional randomization for robustness.
    Windows are aligned to calendar months and adjusted to business days.
    """
    base_train_window_m = int(scenario_config.get("train_window_months", 36))
    base_test_window_m = int(scenario_config.get("test_window_months", 48))
    wf_type = str(scenario_config.get("walk_forward_type", "expanding")).lower()

    robustness_config = global_config.get("wfo_robustness_config", {}) or {}
    enable_window_randomization = bool(robustness_config.get("enable_window_randomization", False))
    enable_start_date_randomization = bool(
        robustness_config.get("enable_start_date_randomization", False)
    )

    train_rand_config = robustness_config.get("train_window_randomization", {}) or {}
    test_rand_config = robustness_config.get("test_window_randomization", {}) or {}
    start_rand_config = robustness_config.get("start_date_randomization", {}) or {}

    train_min_offset = int(train_rand_config.get("min_offset", 3))
    train_max_offset = int(train_rand_config.get("max_offset", 14))
    test_min_offset = int(test_rand_config.get("min_offset", 3))
    test_max_offset = int(test_rand_config.get("max_offset", 14))
    start_min_offset = int(start_rand_config.get("min_offset", 0))
    start_max_offset = int(start_rand_config.get("max_offset", 12))

    if len(monthly_data_index) < (base_train_window_m + base_test_window_m):
        logger.warning(
            "Not enough monthly data for base windows. Required: %d months, Available: %d months.",
            base_train_window_m + base_test_window_m,
            len(monthly_data_index),
        )
        return []

    def _month_start_bday(ts: pd.Timestamp) -> pd.Timestamp:
        month_start = ts.to_period("M").to_timestamp()
        return pd.bdate_range(start=month_start, end=month_start + pd.offsets.MonthEnd(0))[0]

    def _month_end_bday(ts: pd.Timestamp) -> pd.Timestamp:
        month_start = ts.to_period("M").to_timestamp()
        return pd.bdate_range(start=month_start, end=month_start + pd.offsets.MonthEnd(0))[-1]

    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    current_window_start_idx = 0

    while True:
        start_offset = (
            int(rng.integers(start_min_offset, start_max_offset, endpoint=True))
            if enable_start_date_randomization
            else 0
        )
        if enable_window_randomization:
            train_win = base_train_window_m + int(
                rng.integers(train_min_offset, train_max_offset, endpoint=True)
            )
            test_win = base_test_window_m + int(
                rng.integers(test_min_offset, test_max_offset, endpoint=True)
            )
        else:
            train_win = base_train_window_m
            test_win = base_test_window_m

        if wf_type == "rolling":
            train_start_idx = current_window_start_idx + start_offset
            train_end_idx = train_start_idx + train_win - 1
        else:
            train_start_idx = start_offset
            train_end_idx = current_window_start_idx + train_win - 1

        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_win - 1

        if test_end_idx >= len(monthly_data_index):
            break

        train_start_date = _month_start_bday(monthly_data_index[train_start_idx])
        train_end_date = _month_end_bday(monthly_data_index[train_end_idx])
        test_start_date = _month_start_bday(monthly_data_index[test_start_idx])
        test_end_date = _month_end_bday(monthly_data_index[test_end_idx])

        windows.append((train_start_date, train_end_date, test_start_date, test_end_date))

        step = int(test_win)
        current_window_start_idx += step
        if current_window_start_idx + base_train_window_m + base_test_window_m > len(
            monthly_data_index
        ):
            break

    if logger.isEnabledFor(logging.DEBUG):
        for i, (ts, te, vs, ve) in enumerate(windows):
            logger.debug(
                "Window %d: Train=%s to %s, Test=%s to %s",
                i + 1,
                ts.date(),
                te.date(),
                vs.date(),
                ve.date(),
            )

    return windows


def generate_enhanced_wfo_windows(
    monthly_data_index: pd.DatetimeIndex,
    scenario_config: dict,
    global_config: dict,
    rng: np.random.Generator,
) -> List:
    """Generate enhanced WFO windows with strategy-appropriate evaluation frequency.

    Args:
        monthly_data_index: DatetimeIndex of monthly data (assumed to be month-end dates)
        scenario_config: Scenario configuration dictionary
        global_config: Global configuration dictionary
        rng: Random number generator instance for reproducibility

    Returns:
        List of WFOWindow objects with appropriate evaluation frequency
    """
    # Import here to avoid circular imports
    try:
        from ..optimization.wfo_window import WFOWindow
    except ImportError:
        # Fallback to regular window generation if WFOWindow not available
        return generate_randomized_wfo_windows(
            monthly_data_index, scenario_config, global_config, rng
        )

    # Generate base windows using existing logic
    base_windows = generate_randomized_wfo_windows(
        monthly_data_index, scenario_config, global_config, rng
    )

    # Determine evaluation frequency
    evaluation_frequency = _determine_evaluation_frequency(scenario_config)

    # Convert to enhanced windows
    enhanced_windows = []
    for window in base_windows:
        enhanced_windows.append(
            WFOWindow(
                train_start=window[0],
                train_end=window[1],
                test_start=window[2],
                test_end=window[3],
                evaluation_frequency=evaluation_frequency,
                strategy_name=scenario_config.get("name", "unknown"),
            )
        )

    return enhanced_windows


def _determine_evaluation_frequency(scenario_config: dict) -> str:
    """Determine required evaluation frequency based on strategy configuration.

    Args:
        scenario_config: Scenario configuration dictionary

    Returns:
        Evaluation frequency ('D', 'W', or 'M')
    """
    strategy_class = scenario_config.get("strategy_class", "")
    strategy_name = scenario_config.get("strategy", "")
    timing_config = scenario_config.get("timing_config", {})

    # Intramonth and seasonal strategies need daily evaluation
    if (
        "intramonth" in strategy_class.lower()
        or "intramonth" in strategy_name.lower()
        or "seasonalsignal" in strategy_class.lower()
        or "seasonalsignal" in strategy_name.lower()
    ):
        return "D"

    # Signal-based timing with daily scanning
    if timing_config.get("mode") == "signal_based":
        scan_freq = timing_config.get("scan_frequency", "D")
        if scan_freq == "D":
            return "D"

    # Check rebalance frequency
    rebalance_freq = scenario_config.get("rebalance_frequency", "M")
    if rebalance_freq == "D":
        return "D"

    # Default to monthly for backward compatibility
    return "M"


def calculate_stability_metrics(metric_values_per_objective, metrics_to_optimize, global_config):
    """
    Calculate stability-focused metrics across WFO windows.

    Args:
        metric_values_per_objective: List of lists containing metric values for each objective across windows
        metrics_to_optimize: List of metric names being optimized
        global_config: Global configuration dictionary

    Returns:
        Dictionary of stability metrics
    """
    stability_config = global_config.get("wfo_robustness_config", {}).get("stability_metrics", {})
    worst_percentile = stability_config.get("worst_percentile", 10)
    consistency_threshold = stability_config.get("consistency_threshold", 0.0)

    stability_metrics = {}

    for i, metric_name in enumerate(metrics_to_optimize):
        values = metric_values_per_objective[i]
        valid_values = [v for v in values if np.isfinite(v)]

        if not valid_values:
            stability_metrics[f"{metric_name}_Std"] = np.nan
            stability_metrics[f"{metric_name}_CV"] = np.nan
            stability_metrics[f"{metric_name}_Worst_{worst_percentile}pct"] = np.nan
            stability_metrics[f"{metric_name}_Consistency_Ratio"] = np.nan
            continue

        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)

        # Coefficient of variation (stability measure)
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-9 else np.inf

        # Worst case performance (downside protection)
        worst_case = np.percentile(valid_values, worst_percentile)

        # Consistency ratio (percentage of windows above threshold)
        above_threshold = [v for v in valid_values if v > consistency_threshold]
        consistency_ratio = len(above_threshold) / len(valid_values)

        stability_metrics[f"stability_{metric_name}_Std"] = std_val
        stability_metrics[f"stability_{metric_name}_CV"] = cv
        stability_metrics[f"stability_{metric_name}_Worst_{worst_percentile}pct"] = worst_case
        stability_metrics[f"stability_{metric_name}_Consistency_Ratio"] = consistency_ratio

    return stability_metrics


# --------------------------------------------------------------------------- #
# DataFrame → float32 NumPy helper (for Numba kernels)
# --------------------------------------------------------------------------- #


def _df_to_float32_array(
    df: pd.DataFrame, *, column_names: Optional[List[str]] = None
) -> tuple[np.ndarray, list[str]]:
    """Convert a (potentially Multi-Index) DataFrame to a contiguous
    ``float32`` NumPy ndarray suitable for Numba kernels.

    Parameters
    ----------
    df : pd.DataFrame
        Price or returns data. Index must be monotonic and unique.
    column_names : Optional[List[str]], default None
        If *df* has a Multi-Index with levels (Ticker, Field) supply the
        desired *Field* (e.g. ["Close"]) to extract.  When None the function
        assumes *df* already has one column per asset.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        A 2-D ``float32`` array of shape (n_periods, n_assets) and the list
        of tickers in column order.  Missing values are represented as
        ``np.nan``.
    """
    from ..interfaces import create_array_converter

    # Use polymorphic array converter to eliminate isinstance violations
    converter = create_array_converter()
    matrix, tickers = converter.convert_to_array(df, column_names)
    return matrix.astype(np.float32), tickers
