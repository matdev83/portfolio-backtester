from . import strategies

def _resolve_strategy(name: str):
    class_name = "".join(w.capitalize() for w in name.split('_')) + "Strategy"
    if name == "vams_momentum":
        class_name = "VAMSMomentumStrategy"
    elif name == "vams_no_downside":
        class_name = "VAMSNoDownsideStrategy"
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
    tc = turn * (scenario_cfg["transaction_costs_bps"] / 10_000)

    return (gross - tc).reindex(price_daily.index).fillna(0)
