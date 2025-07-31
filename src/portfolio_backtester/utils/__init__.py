import logging  # Assuming logger might be useful here or for other utils
import random
import signal

import numpy as np
import pandas as pd

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from .. import strategies
from ..api_stability import api_stable

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


@api_stable(version="1.0", strict_params=True, strict_return=False)
def _resolve_strategy(name: str):
    class_name = "".join(w.capitalize() for w in name.split('_')) + "Strategy"
    if name == "momentum_unfiltered_atr":
        class_name = "MomentumUnfilteredAtrStrategy"
    elif name == "vams_momentum":
        class_name = "VAMSMomentumStrategy"
    elif name == "vams_no_downside":
        class_name = "VAMSNoDownsideStrategy"
    elif name == "ema_crossover":
        class_name = "EMAStrategy"
    elif name == "low_volatility_factor":
        class_name = "LowVolatilityFactorStrategy"
    elif name == "uvxy_rsi":
        class_name = "UvxyRsiStrategy"
    return getattr(strategies, class_name, None)


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

    from ..portfolio.position_sizer import get_position_sizer_from_config
    from ..portfolio.rebalancing import rebalance

    strat_cls = _resolve_strategy(scenario_cfg["strategy"])
    if not strat_cls:
        raise ValueError(f"Could not resolve strategy: {scenario_cfg['strategy']}")
    strategy = strat_cls(scenario_cfg["strategy_params"])

    # --- 1) Generate signals on monthly data --------------------------------
    universe_cols = [c for c in price_monthly.columns if c != global_cfg["benchmark"]]
    signals = strategy.generate_signals(
        price_monthly[universe_cols],
        features_slice,
        price_monthly[global_cfg["benchmark"]],
    )

    sizer = get_position_sizer_from_config(scenario_cfg)

    sizer_params = scenario_cfg.get("strategy_params", {}).copy()
    # Map sizer-specific parameters from strategy_params to expected sizer argument names
    sizer_param_mapping = {
        "sizer_sharpe_window": "window",
        "sizer_sortino_window": "window",
        "sizer_beta_window": "window",
        "sizer_corr_window": "window",
        "sizer_dvol_window": "window",
        "sizer_target_return": "target_return",  # For Sortino sizer
    }
    for old_key, new_key in sizer_param_mapping.items():
        if old_key in sizer_params:
            sizer_params[new_key] = sizer_params.pop(old_key)

    sizer_kwargs = {
        "signals": signals,
        "prices": price_monthly[universe_cols],
        "benchmark": price_monthly[global_cfg["benchmark"]],
        **sizer_params
    }

    sized = sizer.calculate_weights(**sizer_kwargs)
    weights_monthly = rebalance(sized, scenario_cfg["rebalance_frequency"])

    # --- 2) Expand weights to daily frequency -------------------------------
    weights_daily = (
        weights_monthly.reindex(price_daily.index, method="ffill").fillna(0.0)
    )

    # --- 3) Compute daily portfolio returns ---------------------------------
    gross = (weights_daily.shift(1).fillna(0.0) * rets_daily).sum(axis=1)
    turn = (weights_daily - weights_daily.shift(1)).abs().sum(axis=1)
    
    # Use realistic transaction costs (13 bps for liquid S&P 500 stocks)
    realistic_cost_bps = 13.0  # Conservative estimate for retail trading liquid large caps
    tc = turn * (realistic_cost_bps / 10_000)

    return (gross - tc).reindex(price_daily.index).fillna(0)




def _get_trading_days_in_month(year: int, month: int) -> pd.DatetimeIndex:
    """Helper to get all trading days in a given month."""
    start_of_month = pd.Timestamp(year=year, month=month, day=1)
    end_of_month = start_of_month + pd.offsets.MonthEnd(1)
    return pd.bdate_range(start=start_of_month, end=end_of_month)

def generate_randomized_wfo_windows(monthly_data_index, scenario_config, global_config, random_state=None):
    """
    Generate walk-forward optimization windows with optional randomization for robustness.
    Windows are aligned to calendar months and ensure valid trading days.
    
    Args:
        monthly_data_index: DatetimeIndex of monthly data (assumed to be month-end dates)
        scenario_config: Scenario configuration dictionary
        global_config: Global configuration dictionary
        random_state: Random seed for reproducibility
        
    Returns:
        List of tuples: (train_start, train_end, test_start, test_end)
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Get base window sizes
    base_train_window_m = scenario_config.get("train_window_months", 36)
    base_test_window_m = scenario_config.get("test_window_months", 48)
    wf_type = scenario_config.get("walk_forward_type", "expanding").lower()
    
    # Get robustness configuration
    robustness_config = global_config.get("wfo_robustness_config", {})
    enable_window_randomization = robustness_config.get("enable_window_randomization", False)
    enable_start_date_randomization = robustness_config.get("enable_start_date_randomization", False)
    
    # Window randomization parameters
    train_rand_config = robustness_config.get("train_window_randomization", {})
    test_rand_config = robustness_config.get("test_window_randomization", {})
    start_rand_config = robustness_config.get("start_date_randomization", {})
    
    train_min_offset = train_rand_config.get("min_offset", 3)
    train_max_offset = train_rand_config.get("max_offset", 14)
    test_min_offset = test_rand_config.get("min_offset", 3)
    test_max_offset = test_rand_config.get("max_offset", 14)
    start_min_offset = start_rand_config.get("min_offset", 0)
    start_max_offset = start_rand_config.get("max_offset", 12)
    
    windows = []
    
    # Determine the effective start date for the first window
    # This should be the earliest date from which a full train_window_m can be formed
    # and then potentially offset by start_offset
    
    # Find the first valid month-end date that can serve as a start for a train window
    # The monthly_data_index is assumed to be month-end dates.
    # We need at least (base_train_window_m + base_test_window_m) months of data
    if len(monthly_data_index) < (base_train_window_m + base_test_window_m):
        logger.warning(f"Not enough monthly data for base windows. Required: {base_train_window_m + base_test_window_m} months, Available: {len(monthly_data_index)} months.")
        return []

    # Calculate the first possible start of a train window
    # This is the date that allows for a full train_window_m + test_window_m to exist
    first_possible_train_start_idx = 0
    
    # Iterate through the monthly data index to generate windows
    current_window_start_idx = first_possible_train_start_idx
    
    while True:
        # Apply start date randomization for the current iteration
        if enable_start_date_randomization:
            current_start_offset = random.randint(start_min_offset, start_max_offset)
        else:
            current_start_offset = 0
        
        # Apply window randomization for the current iteration
        if enable_window_randomization:
            current_train_offset = random.randint(train_min_offset, train_max_offset)
            current_test_offset = random.randint(test_min_offset, test_max_offset)
            current_train_window_m = base_train_window_m + current_train_offset
            current_test_window_m = base_test_window_m + current_test_offset
        else:
            current_train_window_m = base_train_window_m
            current_test_window_m = base_test_window_m

        # Calculate window boundaries based on calendar months
        # train_end is the end of the training period (month-end)
        # test_end is the end of the testing period (month-end)
        
        # Determine the actual start of the training period based on the current_window_start_idx
        # and the randomized start offset
        
        # For rolling windows, train_start moves with the window
        if wf_type == "rolling":
            train_start_month_idx = current_window_start_idx + current_start_offset
            train_end_month_idx = train_start_month_idx + current_train_window_m - 1
        else: # Expanding window
            train_start_month_idx = current_start_offset # Start from the beginning, offset by random
            train_end_month_idx = current_window_start_idx + current_train_window_m - 1
            
        test_start_month_idx = train_end_month_idx + 1
        test_end_month_idx = test_start_month_idx + current_test_window_m - 1
        
        # Ensure indices are within bounds of monthly_data_index
        if test_end_month_idx >= len(monthly_data_index):
            break # No more full test windows

        train_start_date = monthly_data_index[train_start_month_idx]
        train_end_date = monthly_data_index[train_end_month_idx]
        test_start_date = monthly_data_index[test_start_month_idx]
        test_end_date = monthly_data_index[test_end_month_idx]

        # Adjust dates to be actual business days.
        # For start dates, we want the first business day of that month.
        # For end dates, we want the last business day of that month.

        # For train_start_date, get the first day of its month, then find the first business day on or after it.
        train_start_month = train_start_date.to_period('M').to_timestamp()
        train_start_date = pd.bdate_range(start=train_start_month, end=train_start_month + pd.offsets.MonthEnd(0))[0]
        
        # For train_end_date, get the last day of its month, then find the last business day on or before it.
        train_end_month = train_end_date.to_period('M').to_timestamp()
        train_end_date = pd.bdate_range(start=train_end_month, end=train_end_month + pd.offsets.MonthEnd(0))[-1]
        
        # For test_start_date, get the first day of its month, then find the first business day on or after it.
        test_start_month = test_start_date.to_period('M').to_timestamp()
        test_start_date = pd.bdate_range(start=test_start_month, end=test_start_month + pd.offsets.MonthEnd(0))[0]
        
        # For test_end_date, get the last day of its month, then find the last business day on or before it.
        test_end_month = test_end_date.to_period('M').to_timestamp()
        test_end_date = pd.bdate_range(start=test_end_month, end=test_end_month + pd.offsets.MonthEnd(0))[-1]
        
        windows.append((
            train_start_date,
            train_end_date,
            test_start_date,
            test_end_date,
        ))
        
        # Move to the next window
        # For expanding windows, we increment the end of the training window
        # For rolling windows, we increment the start of the training window
        if wf_type == "rolling":
            current_window_start_idx += current_test_window_m
        else:  # Expanding window
            current_window_start_idx += current_test_window_m
        
        # Break if the next window would exceed the data
        if current_window_start_idx + base_train_window_m + base_test_window_m > len(monthly_data_index):
            break

    if logger.isEnabledFor(logging.DEBUG):
        for i, (ts, te, vs, ve) in enumerate(windows):
            logger.debug(f"Window {i+1}: Train={ts.date()} to {te.date()}, Test={vs.date()} to {ve.date()}")

    return windows


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

def _df_to_float32_array(df: pd.DataFrame, *, field: str | None = None) -> tuple[np.ndarray, list[str]]:
    """Convert a (potentially Multi-Index) DataFrame to a contiguous
    ``float32`` NumPy ndarray suitable for Numba kernels.

    Parameters
    ----------
    df : pd.DataFrame
        Price or returns data. Index must be monotonic and unique.
    field : str | None, default None
        If *df* has a Multi-Index with levels (Ticker, Field) supply the
        desired *Field* (e.g. "Close") to extract.  When None the function
        assumes *df* already has one column per asset.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        A 2-D ``float32`` array of shape (n_periods, n_assets) and the list
        of tickers in column order.  Missing values are represented as
        ``np.nan``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Handle Multi-Index columns (Ticker, Field)
    if isinstance(df.columns, pd.MultiIndex):
        if field is None:
            raise ValueError("Multi-Index DataFrame requires *field* parameter")
        if 'Field' in df.columns.names:
            level_name = 'Field'
        else:
            # Assume the last level holds the field
            level_name = df.columns.names[-1]
        if field not in df.columns.get_level_values(str(level_name)):
            raise KeyError(f"Field '{field}' not found in DataFrame columns")
        extracted = df.xs(field, level=str(level_name), axis=1)
    else:
        extracted = df.copy()

    # Ensure column order is deterministic (sorted tickers)
    tickers = list(extracted.columns)
    extracted = extracted.astype(np.float32)

    # Pandas to NumPy (contiguous)
    matrix = np.ascontiguousarray(extracted.values, dtype=np.float32)
    return matrix, tickers
