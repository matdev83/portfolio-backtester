import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

class TrialEvaluator:
    """Shared evaluation engine used by both Optuna and Genetic optimizers.

    It isolates walk-forward evaluation logic so that parameter-search strategies
    (Optuna, GA, etc.) only need to focus on suggesting parameters.
    """

    def __init__(
        self,
        backtester_instance,
        scenario_config: dict,
        monthly_data,
        daily_data,
        rets_full,
        metrics_to_optimize: list[str],
        is_multi_objective: bool,
        windows: list,
    ) -> None:
        self.backtester = backtester_instance
        self.scenario_config = scenario_config
        self.monthly_data = monthly_data
        self.daily_data = daily_data
        self.rets_full = rets_full
        self.metrics_to_optimize = metrics_to_optimize
        self.is_multi_objective = is_multi_objective
        self.windows = windows
        self.logger = logging.getLogger(__name__)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _suggest_optuna_params(self, trial, base_params: dict, opt_specs: list[dict]):
        """Ask an Optuna trial to suggest values for *opt_specs*.

        Returns a new parameter dict that can be used in the scenario config.
        """
        params = base_params.copy()
        for spec in opt_specs:
            pname = spec["parameter"]
            opt_defaults_source = {}
            try:
                opt_defaults_source = getattr(self.backtester, "global_config", {}).get(
                    "optimizer_parameter_defaults", {}
                )
            except Exception:
                # When the backtester is a mock object in unit-tests it may not have global_config
                pass

            opt_def = opt_defaults_source.get(pname, {})
            ptype = spec.get("type", opt_def.get("type"))

            low = spec.get("min_value", opt_def.get("low"))
            high = spec.get("max_value", opt_def.get("high"))
            step = spec.get("step", opt_def.get("step", 1 if ptype == "int" else None))
            log = spec.get("log", opt_def.get("log", False))

            try:
                if ptype == "int":
                    params[pname] = trial.suggest_int(pname, int(low), int(high), step=int(step) if step else 1)
                elif ptype == "float":
                    params[pname] = trial.suggest_float(
                        pname, float(low), float(high), step=float(step) if step else None, log=log
                    )
                elif ptype == "categorical":
                    choices = spec.get("values", opt_def.get("values"))
                    if not choices:
                        raise ValueError("No choices supplied for categorical parameter.")
                    params[pname] = trial.suggest_categorical(pname, choices)
                else:
                    self.logger.warning("Unsupported parameter type '%s' for %s. Skipping.", ptype, pname)
            except Exception as exc:
                self.logger.warning("Failed suggesting parameter %s via Optuna – %s", pname, exc)
        return params

    # ---------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------
    def evaluate(self, trial):
        """Evaluate *trial* across all WFO windows and aggregate the objective.

        1. Ask the optimizer to suggest parameter values (when trial is provided).
        2. Run the backtester for each walk-forward window in parallel.
        3. Aggregate objective values (mean) and full-PnL returns (concatenate).
        4. Attach the full returns to the Optuna trial for later analysis.
        """
        # ------------------------------------------------------------------
        # 1. Resolve the concrete parameter set for this trial
        # ------------------------------------------------------------------
        current_params = self._suggest_optuna_params(
            trial, self.scenario_config["strategy_params"], self.scenario_config.get("optimize", [])
        )
        trial_scenario_cfg = self.scenario_config.copy()
        trial_scenario_cfg["strategy_params"] = current_params

        # ------------------------------------------------------------------
        # 2. Evaluate each WFO window in a separate process
        # ------------------------------------------------------------------
        # Decide whether to use multiprocessing or fall back to serial execution.
        use_multiprocessing = True

        # Avoid multiprocessing when the backtester is a unittest.mock object (tests) – it cannot be pickled.
        try:
            import unittest.mock as umock

            if isinstance(self.backtester, umock.Mock):
                use_multiprocessing = False
        except Exception:
            # unittest may not be available in runtime env but that's fine.
            pass

        max_workers = max(1, getattr(self.backtester, "n_jobs", 1))
        objective_values: list[np.ndarray | float] = []
        full_pnl_returns = []

        if use_multiprocessing and max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.backtester.evaluate_fast,
                        trial,
                        trial_scenario_cfg,
                        [window],  # evaluate_fast expects a list of windows
                        self.monthly_data,
                        self.daily_data,
                        self.rets_full,
                        self.metrics_to_optimize,
                        self.is_multi_objective,
                    )
                    for window in self.windows
                ]

                for fut in as_completed(futures):
                    obj_val, pnl_ret = fut.result()
                    objective_values.append(obj_val)
                    if pnl_ret is not None and not pnl_ret.empty:
                        full_pnl_returns.append(pnl_ret)
        else:
            # Serial evaluation (safer for mocked objects and debug runs)
            for window in self.windows:
                obj_val, pnl_ret = self.backtester.evaluate_fast(
                    trial,
                    trial_scenario_cfg,
                    [window],
                    self.monthly_data,
                    self.daily_data,
                    self.rets_full,
                    self.metrics_to_optimize,
                    self.is_multi_objective,
                )
                objective_values.append(obj_val)
                if pnl_ret is not None and not pnl_ret.empty:
                    full_pnl_returns.append(pnl_ret)

        # ------------------------------------------------------------------
        # 3. Aggregate results across windows
        # ------------------------------------------------------------------
        aggregate_objective = np.mean(objective_values, axis=0)
        aggregate_returns = (
            np.concatenate(full_pnl_returns) if full_pnl_returns else np.array([], dtype=float)
        )

        # ------------------------------------------------------------------
        # 4. Attach aggregated returns to the trial for downstream diagnostics
        # ------------------------------------------------------------------
        if trial is not None:
            # Convert numpy array to list for JSON serialisation in Optuna storage
            try:
                trial.set_user_attr("full_pnl_returns", aggregate_returns.tolist())
            except Exception:
                # Optuna might throw if trial is a mock object without set_user_attr
                pass

        return aggregate_objective

