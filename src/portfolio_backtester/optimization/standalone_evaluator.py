import numpy as np
import pandas as pd


def evaluate_walk_forward(
    scenario_config,
    monthly_data,
    daily_data,
    rets_full,
    metrics_to_optimize,
    is_multi_objective,
    windows,
    run_scenario_method,
    get_strategy_method,
    global_config,
    args,
    data_cache,
    random_state,
):
    from ..reporting.performance_metrics import calculate_metrics

    all_window_returns = []
    for window_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        m_slice = monthly_data.loc[tr_start:tr_end]
        d_slice = daily_data.loc[tr_start:te_end]
        r_slice = rets_full.loc[tr_start:te_end]

        window_returns = run_scenario_method(
            scenario_config, m_slice, d_slice, rets_daily=r_slice, verbose=False
        )

        if window_returns is None or window_returns.empty:
            continue

        all_window_returns.append(window_returns)

    if not all_window_returns:
        if is_multi_objective:
            return tuple([float("nan")] * len(metrics_to_optimize))
        else:
            return float("nan")

    full_pnl_returns = pd.concat(all_window_returns).sort_index()
    full_pnl_returns = full_pnl_returns[~full_pnl_returns.index.duplicated(keep="first")]

    bench_ser = daily_data[global_config["benchmark"]].loc[full_pnl_returns.index]
    bench_period_rets = bench_ser.pct_change(fill_method=None).fillna(0)

    final_metrics = calculate_metrics(
        full_pnl_returns, bench_period_rets, global_config["benchmark"]
    )
    metric_avgs = [final_metrics.get(m, np.nan) for m in metrics_to_optimize]

    if is_multi_objective:
        return tuple(float(v) for v in metric_avgs)
    else:
        return float(metric_avgs[0])
