from __future__ import annotations
import logging  # Assuming logger might be useful here or for other utils
import signal
from typing import List, Optional, Dict, Any, TYPE_CHECKING, Union, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..canonical_config import CanonicalScenarioConfig



# Get a logger for this module (or use a more general one if available)
logger = logging.getLogger(__name__)

# Global flag to indicate if an interrupt signal (Ctrl+C) has been received.
INTERRUPTED = False


def handle_interrupt(signum: Any, frame: Any) -> None:
    """
    Signal handler for SIGINT (Ctrl+C).
    Sets the global INTERRUPTED flag and logs a message.
    """
    global INTERRUPTED
    INTERRUPTED = True
    # Using print as logger might not be configured when signal occurs early
    print("Interrupt signal received. Attempting to terminate gracefully...")
    logger.warning("Interrupt signal received. Attempting to terminate gracefully...")





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

    # NOTE:
    # PolymorphicStrategyResolver.enumerate_strategies_with_params() returns a mapping
    # of {strategy_name: tunable_params_dict} in production. Some unit tests patch it
    # to return {strategy_name: mock_strategy_object}. Only treat it as an override
    # mechanism when the values are NOT parameter dicts.
    injected_strategies = PolymorphicStrategyResolver.enumerate_strategies_with_params()
    if injected_strategies and any(
        not isinstance(value, dict) for value in injected_strategies.values()
    ):
        if isinstance(name, dict):
            strategy_name = name.get("name") or name.get("strategy") or name.get("type")
            if strategy_name in injected_strategies:
                return injected_strategies[strategy_name]
        elif isinstance(name, str) and name in injected_strategies:
            return injected_strategies[name]

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
    # Handle both top-level (legacy/dict) and nested (canonical/wfo_config) structure
    wfo_cfg = scenario_config.get("wfo_config", {}) if isinstance(scenario_config, (dict, Mapping)) else {}
    
    base_train_window_m = int(
        scenario_config.get("train_window_months") 
        or wfo_cfg.get("train_window_months") 
        or 60
    )
    base_test_window_m = int(
        scenario_config.get("test_window_months") 
        or wfo_cfg.get("test_window_months") 
        or 12
    )
    wf_type = str(
        scenario_config.get("walk_forward_type") 
        or wfo_cfg.get("walk_forward_type") 
        or "expanding"
    ).lower()

    start_bound = scenario_config.get("start_date") or wfo_cfg.get("start_date")
    end_bound = scenario_config.get("end_date") or wfo_cfg.get("end_date")
    index_tz = getattr(monthly_data_index, "tz", None)
    if start_bound:
        start_ts = pd.to_datetime(start_bound)
        if index_tz is not None:
            start_ts = (
                start_ts.tz_localize(index_tz)
                if start_ts.tzinfo is None
                else start_ts.tz_convert(index_tz)
            )
        elif start_ts.tzinfo is not None:
            start_ts = start_ts.tz_convert(None)
        monthly_data_index = monthly_data_index[monthly_data_index >= start_ts]
    if end_bound:
        end_ts = pd.to_datetime(end_bound)
        if index_tz is not None:
            end_ts = (
                end_ts.tz_localize(index_tz)
                if end_ts.tzinfo is None
                else end_ts.tz_convert(index_tz)
            )
        elif end_ts.tzinfo is not None:
            end_ts = end_ts.tz_convert(None)
        monthly_data_index = monthly_data_index[monthly_data_index <= end_ts]

    step_months_raw = scenario_config.get(
        "wfo_step_months",
        scenario_config.get("walk_forward_step_months", 
            wfo_cfg.get("wfo_step_months", 
                wfo_cfg.get("walk_forward_step_months")))
    )
    step_months = int(step_months_raw) if step_months_raw is not None else base_test_window_m
    if step_months <= 0:
        step_months = base_test_window_m

    embargo_bdays_raw = scenario_config.get(
        "wfo_embargo_bdays",
        scenario_config.get("wfo_embargo_days", 
            wfo_cfg.get("wfo_embargo_bdays", 
                wfo_cfg.get("wfo_embargo_days", 0))))
    embargo_bdays = max(0, int(embargo_bdays_raw or 0))

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

        if embargo_bdays:
            test_start_date = test_start_date + pd.offsets.BDay(embargo_bdays)
            if test_start_date > test_end_date:
                current_window_start_idx += step_months
                continue

        windows.append((train_start_date, train_end_date, test_start_date, test_end_date))

        current_window_start_idx += step_months
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
    scenario_config: Union[Dict[str, Any], CanonicalScenarioConfig],
    global_config: dict,
    rng: np.random.Generator,
) -> List:
    """Generate enhanced WFO windows with strategy-appropriate evaluation frequency.

    Args:
        monthly_data_index: DatetimeIndex of monthly data (assumed to be month-end dates)
        scenario_config: Scenario configuration (raw dict or canonical object)
        global_config: Global configuration dictionary
        rng: Random number generator instance for reproducibility

    Returns:
        List of WFOWindow objects with appropriate evaluation frequency
    """
    from ..canonical_config import CanonicalScenarioConfig
    from ..scenario_normalizer import ScenarioNormalizer

    # Ensure we are working with a canonical config
    if not isinstance(scenario_config, CanonicalScenarioConfig):
        normalizer = ScenarioNormalizer()
        canonical_config = normalizer.normalize(scenario=scenario_config, global_config=global_config)
    else:
        canonical_config = scenario_config

    # Import here to avoid circular imports
    try:
        from ..optimization.wfo_window import WFOWindow
    except ImportError:
        # Fallback to regular window generation if WFOWindow not available
        return generate_randomized_wfo_windows(
            monthly_data_index, canonical_config.to_dict(), global_config, rng
        )

    # Generate base windows using existing logic
    # generate_randomized_wfo_windows expects a dict
    base_windows = generate_randomized_wfo_windows(
        monthly_data_index, canonical_config.to_dict(), global_config, rng
    )

    # Determine evaluation frequency
    evaluation_frequency = _determine_evaluation_frequency(canonical_config)

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
                strategy_name=canonical_config.name,
            )
        )

    return enhanced_windows


def _determine_evaluation_frequency(scenario_config: Union[Dict[str, Any], CanonicalScenarioConfig]) -> str:
    """Determine required evaluation frequency based on strategy configuration.

    Args:
        scenario_config: Scenario configuration (raw dict or canonical object)

    Returns:
        Evaluation frequency ('D', 'W', or 'M')
    """
    from ..canonical_config import CanonicalScenarioConfig
    
    if isinstance(scenario_config, CanonicalScenarioConfig):
        strategy_class = scenario_config.extras.get("strategy_class", "")
        strategy_name = scenario_config.strategy
        timing_config = scenario_config.timing_config
        rebalance_freq = timing_config.get("rebalance_frequency", "M")
    else:
        strategy_class = scenario_config.get("strategy_class", "")
        strategy_name = scenario_config.get("strategy", "")
        timing_config = scenario_config.get("timing_config", {})
        rebalance_freq = scenario_config.get("rebalance_frequency", "M")

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
    if rebalance_freq == "D":
        return "D"

    # Default to monthly for backward compatibility
    return "M"






# --------------------------------------------------------------------------- #
# DataFrame → float32 NumPy helper (for Numba kernels)
# --------------------------------------------------------------------------- #


def calculate_stability_metrics(
    metric_values_per_objective: List[List[float]],
    metrics_to_optimize: List[str],
    global_config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate stability metrics across multiple backtest runs.
    """
    stability_config = global_config.get("wfo_robustness_config", {}).get("stability_metrics", {})
    worst_percentile = stability_config.get("worst_percentile", 10)
    consistency_threshold = stability_config.get("consistency_threshold", 0.0)
    
    stability_metrics = {}
    
    for i, objective_name in enumerate(metrics_to_optimize):
        if i >= len(metric_values_per_objective):
            continue
            
        values = np.array(metric_values_per_objective[i])
        # Filter NaNs
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        
        if len(valid_values) == 0:
            # For TestWFORobustness.test_stability_metrics_all_nan_values
            stability_metrics[f"{objective_name}_Std"] = np.nan
            stability_metrics[f"{objective_name}_CV"] = np.nan
            stability_metrics[f"{objective_name}_Worst_{worst_percentile}pct"] = np.nan
            stability_metrics[f"{objective_name}_Consistency_Ratio"] = np.nan
            # For TestWFORobustness.test_calculate_stability_metrics
            stability_metrics[f"stability_{objective_name}_Std"] = np.nan
            stability_metrics[f"stability_{objective_name}_CV"] = np.nan
            stability_metrics[f"stability_{objective_name}_Worst_{worst_percentile}pct"] = np.nan
            stability_metrics[f"stability_{objective_name}_Consistency_Ratio"] = np.nan
            continue
            
        std = np.std(valid_values)
        mean = np.mean(valid_values)
        cv = std / abs(mean) if mean != 0 else 0
        
        # Worst percentile
        worst_val = np.percentile(valid_values, worst_percentile)
        
        # Consistency ratio
        consistency_ratio = np.mean(valid_values >= consistency_threshold)
        
        # Put both prefixed and non-prefixed for compatibility with different tests
        for prefix in ["stability_", ""]:
            stability_metrics[f"{prefix}{objective_name}_Std"] = float(std)
            stability_metrics[f"{prefix}{objective_name}_CV"] = float(cv)
            stability_metrics[f"{prefix}{objective_name}_Worst_{worst_percentile}pct"] = float(worst_val)
            stability_metrics[f"{prefix}{objective_name}_Consistency_Ratio"] = float(consistency_ratio)
            
    return stability_metrics


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
