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

    from .portfolio.position_sizer import POSITION_SIZER_REGISTRY
    from .portfolio.rebalancing import rebalance

    strat_cls = _resolve_strategy(scenario_cfg["strategy"])
    strategy = strat_cls(scenario_cfg["strategy_params"])

    # --- 1) Generate signals on monthly data --------------------------------
    universe_cols = [c for c in price_monthly.columns if c != global_cfg["benchmark"]]
    signals = strategy.generate_signals(
        price_monthly[universe_cols],
        features_slice,
        price_monthly[global_cfg["benchmark"]],
    )

    sizer_name = scenario_cfg.get("position_sizer", "equal_weight")
    sizer = POSITION_SIZER_REGISTRY.get(sizer_name)

    rets_monthly = price_monthly[universe_cols].pct_change().fillna(0)
    bench_rets_monthly = price_monthly[global_cfg["benchmark"]].pct_change().fillna(0)

    if sizer_name == "equal_weight" or sizer is None:
        if sizer is None and sizer_name != "none":
            print(f"Unsupported position sizer: {sizer_name}. Using equal_weight.")
        sized = POSITION_SIZER_REGISTRY["equal_weight"](signals)
    elif sizer_name == "rolling_sharpe":
        window = scenario_cfg["strategy_params"].get("sharpe_sizer_window", 6)
        sized = sizer(signals, rets_monthly, window)
    elif sizer_name == "rolling_sortino":
        window = scenario_cfg["strategy_params"].get("sortino_sizer_window", 6)
        target = scenario_cfg["strategy_params"].get("sizer_target_return", 0.0)
        sized = sizer(signals, rets_monthly, window, target)
    elif sizer_name == "rolling_beta":
        window = scenario_cfg["strategy_params"].get("beta_sizer_window", 6)
        sized = sizer(signals, rets_monthly, bench_rets_monthly, window)
    elif sizer_name == "rolling_corr":
        window = scenario_cfg["strategy_params"].get("corr_sizer_window", 6)
        sized = sizer(signals, rets_monthly, bench_rets_monthly, window)
    else:
        sized = signals

    weights_monthly = rebalance(
        sized, scenario_cfg["rebalance_frequency"]
    )

    # --- 2) Expand weights to daily frequency -------------------------------
    weights_daily = (
        weights_monthly.reindex(price_daily.index, method="ffill").fillna(0.0)
    )

    # --- 3) Compute daily portfolio returns ---------------------------------
    gross = (weights_daily.shift(1).fillna(0.0) * rets_daily).sum(axis=1)
    turn = (weights_daily - weights_daily.shift(1)).abs().sum(axis=1)
    tc = turn * (scenario_cfg["transaction_costs_bps"] / 10_000)

    return (gross - tc).reindex(price_daily.index).fillna(0)
