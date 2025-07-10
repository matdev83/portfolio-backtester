from . import strategies
import signal
import logging # Assuming logger might be useful here or for other utils
import random
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

    from .portfolio.position_sizer import get_position_sizer
    from .portfolio.rebalancing import rebalance

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

    sizer_name = scenario_cfg.get("position_sizer", "equal_weight")
    sizer_func = get_position_sizer(sizer_name)

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

    sized = sizer_func(
        signals,
        price_monthly[universe_cols],
        price_monthly[global_cfg["benchmark"]],
        **sizer_params,
    )
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




def generate_randomized_wfo_windows(monthly_data_index, scenario_config, global_config, random_state=None):
    """
    Generate walk-forward optimization windows with optional randomization for robustness.
    
    Args:
        monthly_data_index: DatetimeIndex of monthly data
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
    
    idx = monthly_data_index
    windows = []
    
    # Apply start date randomization
    if enable_start_date_randomization:
        start_offset = random.randint(start_min_offset, start_max_offset)
    else:
        start_offset = 0
    
    # Apply window randomization if enabled
    # Randomization only EXTENDS windows, never shrinks them below specified minimums
    if enable_window_randomization:
        train_offset = random.randint(train_min_offset, train_max_offset)
        test_offset = random.randint(test_min_offset, test_max_offset)
        train_window_m = base_train_window_m + train_offset  # Always >= base size
        test_window_m = base_test_window_m + test_offset     # Always >= base size
    else:
        train_window_m = base_train_window_m
        test_window_m = base_test_window_m
    
    # Generate windows with randomized parameters
    start_idx = train_window_m + start_offset
    
    while start_idx + test_window_m <= len(idx):
        train_end_idx = start_idx - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_window_m - 1
        
        if test_end_idx >= len(idx):
            break
            
        if wf_type == "rolling":
            train_start_idx = train_end_idx - train_window_m + 1
        else:
            train_start_idx = start_offset  # Apply start offset for expanding windows
            
        windows.append((
            idx[train_start_idx],
            idx[train_end_idx], 
            idx[test_start_idx],
            idx[test_end_idx],
        ))
        start_idx += test_window_m

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

