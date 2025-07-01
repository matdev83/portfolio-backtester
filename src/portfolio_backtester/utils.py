from . import strategies

def _resolve_strategy(name: str):
    class_name = "".join(w.capitalize() for w in name.split('_')) + "Strategy"
    if name == "vams_momentum":
        class_name = "VAMSMomentumStrategy"
    elif name == "vams_no_downside":
        class_name = "VAMSNoDownsideStrategy"
    return getattr(strategies, class_name, None)

def _run_scenario_static(global_cfg, scenario_cfg, data_slice, rets_slice, benchmark_data, features_slice):
    """A trimmed-down, picklable version of Backtester.run_scenario()."""
    from .portfolio.position_sizer import equal_weight_sizer
    from .portfolio.rebalancing import rebalance
    strat_cls = _resolve_strategy(scenario_cfg["strategy"])
    strategy   = strat_cls(scenario_cfg["strategy_params"])

    d = data_slice.loc[rets_slice.index]
    bench = d[global_cfg["benchmark"]]
    
    signals = strategy.generate_signals(d.drop(columns=[global_cfg["benchmark"]]), features_slice, bench)
    weights = rebalance(equal_weight_sizer(signals), scenario_cfg["rebalance_frequency"])

    aligned = rets_slice.loc[weights.index]
    gross   = (weights.shift(1) * aligned).sum(axis=1).dropna()
    turn    = (weights - weights.shift(1)).abs().sum(axis=1)
    tc      = turn * (scenario_cfg["transaction_costs_bps"] / 10_000)
    return (gross - tc).reindex(gross.index).fillna(0)
