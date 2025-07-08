import pandas as pd
import numpy as np
import optuna
import logging
from typing import Any

from .reporting.performance_metrics import calculate_metrics

logger = logging.getLogger(__name__)

def _evaluate_params_walk_forward(self, trial: Any, scenario_config: dict, windows: list,
                                  monthly_data, daily_data, rets_full,
                                  metrics_to_optimize: list, is_multi_objective: bool) -> float | tuple[float, ...]:
    metric_values_per_objective = [[] for _ in metrics_to_optimize]
    processed_steps_for_pruning = 0

    pruning_enabled = getattr(self.args, "pruning_enabled", False)
    pruning_interval_steps = getattr(self.args, "pruning_interval_steps", 1)

    for window_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        m_slice = monthly_data.loc[tr_start:te_end]
        d_slice = daily_data.loc[tr_start:te_end]
        r_slice = rets_full.loc[tr_start:te_end]

        window_returns = self.run_scenario(scenario_config, m_slice, d_slice, r_slice, verbose=False)

        if window_returns is None or window_returns.empty:
            self.logger.warning(f"No returns generated for window {tr_start}-{te_end}. Skipping.")
            for i in range(len(metrics_to_optimize)):
                metric_values_per_objective[i].append(np.nan)
            continue

        test_rets = window_returns.loc[te_start:te_end]
        if test_rets.empty:
            self.logger.debug(f"Test returns empty for window {tr_start}-{te_end} with params {scenario_config['strategy_params']}.")
            if is_multi_objective:
                return tuple([float("nan")] * len(metrics_to_optimize))
            return float("nan")

        if abs(test_rets.mean()) < 1e-9 and abs(test_rets.std()) < 1e-9:
            if trial and hasattr(trial, "set_user_attr"):
                trial.set_user_attr("zero_returns", True)
                if hasattr(trial, "number"):
                    self.logger.debug(f"Trial {trial.number}, window {window_idx+1}: Marked with zero_returns.")

        bench_ser = d_slice[self.global_config["benchmark"]].loc[te_start:te_end]
        bench_period_rets = bench_ser.pct_change(fill_method=None).fillna(0)
        metrics = calculate_metrics(test_rets, bench_period_rets, self.global_config["benchmark"])
        current_metrics = np.array([metrics.get(m, np.nan) for m in metrics_to_optimize], dtype=float)

        if np.isnan(current_metrics).any():
            nan_metrics = [metrics_to_optimize[i] for i, is_nan in enumerate(np.isnan(current_metrics)) if is_nan]
            self.logger.info(f"NaN metric(s) found for window {tr_start}-{te_end}: {', '.join(nan_metrics)}. Params: {scenario_config['strategy_params']}")

        for i, metric_val in enumerate(current_metrics):
            metric_values_per_objective[i].append(float(metric_val) if np.isfinite(metric_val) else np.nan)

        if not np.isnan(current_metrics).all():
            processed_steps_for_pruning += 1

        if pruning_enabled and not is_multi_objective and processed_steps_for_pruning > 0 and processed_steps_for_pruning % pruning_interval_steps == 0:
            if trial and hasattr(trial, 'report') and hasattr(trial, 'should_prune') and hasattr(trial, 'study'):
                intermediate_values = metric_values_per_objective[0]
                intermediate_value = np.nanmean(np.asarray(intermediate_values, dtype=float))

                if not np.isfinite(intermediate_value):
                    first_metric_direction = trial.study.directions[0]
                    intermediate_value = -1e12 if first_metric_direction == optuna.study.StudyDirection.MAXIMIZE else 1e12
                    self.logger.debug(f"Trial {trial.number}, window {window_idx+1}: intermediate metric {metrics_to_optimize[0]} was non-finite. Reporting {intermediate_value}")

                trial.report(float(intermediate_value), window_idx + 1)
                if trial.should_prune():
                    self.logger.info(f"Trial {trial.number} pruned at window {window_idx + 1} with intermediate value for '{metrics_to_optimize[0]}': {intermediate_value:.4f}")
                    raise optuna.exceptions.TrialPruned()

    metric_avgs = [np.nanmean(values) if not all(np.isnan(values)) else np.nan for values in metric_values_per_objective]

    if all(np.isnan(np.array(metric_avgs))):
        self.logger.warning(f"No valid windows produced results for params: {scenario_config['strategy_params']}. Returning NaN.")
        return tuple([float("nan")] * len(metrics_to_optimize)) if is_multi_objective else float("nan")

    if is_multi_objective:
        return tuple(float(v) for v in metric_avgs)
    else:
        return float(metric_avgs[0])
