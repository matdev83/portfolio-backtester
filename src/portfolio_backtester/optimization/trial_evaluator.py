"""Trial evaluator module.

This module contains the :class:`TrialEvaluator` which is responsible for running
walk-forward evaluations during parameter search.  A **key performance concern**
identified during profiling was the repeated pickling of *large* immutable data
frames (daily/monthly price data, full returns) every time an individual walk
forward window is submitted to a worker process.  To mitigate this the module
now uses a *process-level* initializer so that each worker only receives these
objects **once** at start-up.  Subsequent task submissions pass only the
*window* slice, dramatically reducing inter-process serialisation overhead on
platforms that use the *spawn* start method (e.g. Windows).
"""

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd  # type: ignore
from typing import Any, Tuple, List

# ------------------------------------------------------------------
# Helper to ensure a WFO window has at least two trading rows in daily_data.
# Simply checking calendar days is unreliable once missing-asset rows are
# dropped.  We validate using the actual DataFrame index.
# ------------------------------------------------------------------


def _is_valid_window(daily_index: pd.Index, window: tuple) -> bool:
    """Return True when the slice between window[2] and window[3] spans ≥2 rows."""

    start_idx = daily_index.searchsorted(pd.Timestamp(window[2]))
    end_idx = daily_index.searchsorted(pd.Timestamp(window[3]))
    return (end_idx - start_idx) >= 2

# ---------------------------------------------------------------------------
# Module-level globals for worker initialisation.  They will be populated by
# the ProcessPoolExecutor *initializer* so that large read-only objects only
# need to be serialised **once** per worker process instead of once **per task**.
# ---------------------------------------------------------------------------

_BT: "Any" = None                 # Backtester instance (picklable)
_TRIAL: "Any" = None              # Optuna/Genetic trial object (picklable)
_SCENARIO_CFG: dict | None = None
_MONTHLY_DATA = None
_DAILY_DATA = None
_RETS_FULL = None
_METRICS: List[str] | None = None
_IS_MULTI: bool | None = None

# Heavy read-only NumPy caches (initialised once per process)
_PRICES_NP: np.ndarray | None = None
_SIGNALS_NP: np.ndarray | None = None
_DATE_INDEX: np.ndarray | None = None


def _init_worker(backtester: Any,
                 trial: Any,
                 scenario_cfg: dict,
                 monthly_data,
                 daily_data,
                 rets_full,
                 metrics_to_optimize: List[str],
                 is_multi_objective: bool) -> None:
    """Initializer for worker processes.

    Heavy, read-only inputs are stored in module-level globals so that individual
    *task* submissions only have to serialise a lightweight *window* object.
    """
    global _BT, _TRIAL, _SCENARIO_CFG, _MONTHLY_DATA, _DAILY_DATA, _RETS_FULL, _METRICS, _IS_MULTI

    _BT = backtester
    _TRIAL = trial
    _SCENARIO_CFG = scenario_cfg
    _MONTHLY_DATA = monthly_data
    _DAILY_DATA = daily_data
    _RETS_FULL = rets_full
    _METRICS = metrics_to_optimize
    _IS_MULTI = is_multi_objective

    # ------------------------------------------------------------------
    # Prepare heavy NumPy matrices *once* per worker.
    # ------------------------------------------------------------------
    from src.portfolio_backtester.utils import _df_to_float32_array  # local import to avoid overhead in main proc

    global _PRICES_NP, _SIGNALS_NP, _DATE_INDEX

    # Price matrix (float32)
    if isinstance(daily_data.columns, pd.MultiIndex):
        _PRICES_NP, _ = _df_to_float32_array(daily_data, field="Close")  # type: ignore[arg-type]
    else:
        _PRICES_NP, _ = _df_to_float32_array(daily_data)  # type: ignore[arg-type]

    # Date index for fast lookup
    _DATE_INDEX = daily_data.index.values.astype('datetime64[ns]')

    # Compute signals once for this parameter set
    strategy_cls = backtester.strategy_map.get(scenario_cfg["strategy"])
    strategy_instance = strategy_cls(scenario_cfg["strategy_params"])
    signals_df = strategy_instance.generate_signals(
        monthly_data,
        daily_data,
        rets_full,
        None,
        None,
        None,
    )
    _SIGNALS_NP, _ = _df_to_float32_array(signals_df)  # type: ignore[arg-type]


def _evaluate_window(window) -> Tuple[Any, Any]:
    """Evaluate a single walk-forward *window* using global caches.

    Returns
    -------
    tuple
        (objective_value, numpy_returns)
    """
    from src.portfolio_backtester.numba_kernels import run_backtest_numba  # local to keep worker import cost low

    if _PRICES_NP is None or _SIGNALS_NP is None or _DATE_INDEX is None:
        raise RuntimeError("Worker caches not initialised")

    # Translate window (tuple) start/end dates to indices
    start_idx = np.searchsorted(_DATE_INDEX, np.datetime64(window[2]))
    end_idx = np.searchsorted(_DATE_INDEX, np.datetime64(window[3]))

    # Skip invalid slices with <2 rows
    if end_idx - start_idx < 2:
        return -1e9, np.asarray([np.nan], dtype=np.float64)

    start_indices = np.asarray([start_idx], dtype=np.int64)
    end_indices = np.asarray([end_idx], dtype=np.int64)

    try:
        ret_arr = run_backtest_numba(_PRICES_NP, _SIGNALS_NP, start_indices, end_indices)

        finite_mask = np.isfinite(ret_arr)
        if finite_mask.any():
            objective_value = float(ret_arr[finite_mask].mean())
        else:
            # No finite returns – treat as invalid window
            objective_value = -1e9
    except Exception as exc:
        # Catch any unexpected crashes inside the Numba kernel so the worker
        # does not terminate the whole ProcessPool.  We log the error and
        # return NaNs so the trial is treated as a poor performer.
        import traceback, logging
        logging.getLogger(__name__).error("run_backtest_numba failed: %s", exc)
        logging.getLogger(__name__).debug("Traceback:\n%s", traceback.format_exc())
        ret_arr = np.asarray([np.nan], dtype=np.float64)
        objective_value = -1e9

    return objective_value, ret_arr


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

        max_workers_cfg = max(1, getattr(self.backtester, "n_jobs", 1))
        # Do not spawn more workers than windows – that only wastes resources.
        worker_count = min(len(self.windows), max_workers_cfg)

        objective_values: list[float | np.ndarray] = []
        window_lengths: list[int] = []
        full_pnl_returns: list[np.ndarray] = []

        from sys import platform

        # Decide executor class: use processes by default except on Windows
        # where spawning heavy Numba workers is fragile; threads are safer and
        # still run fast because Numba releases the GIL inside nopython code.
        executor_cls = ProcessPoolExecutor
        if platform.startswith("win"):
            from concurrent.futures import ThreadPoolExecutor
            executor_cls = ThreadPoolExecutor
            # Thread pools share memory with parent – initialise caches here.
            try:
                _init_worker(
                    self.backtester,
                    trial,
                    trial_scenario_cfg,
                    self.monthly_data,
                    self.daily_data,
                    self.rets_full,
                    self.metrics_to_optimize,
                    self.is_multi_objective,
                )
            except Exception as init_exc:
                self.logger.error("ThreadPool cache initialisation failed: %s", init_exc)
                # Fallback to single-thread serial path
                use_multiprocessing = False

        if use_multiprocessing and worker_count > 1:
            # ------------------------------------------------------------------
            # By passing the heavy, read-only inputs only once via *initializer*
            # we avoid re-serialising them for every individual task.
            # ------------------------------------------------------------------
            exec_kwargs = {"max_workers": worker_count}
            if executor_cls is ProcessPoolExecutor:
                exec_kwargs.update(
                    initializer=_init_worker,
                    initargs=(
                        self.backtester,
                        trial,
                        trial_scenario_cfg,
                        self.monthly_data,
                        self.daily_data,
                        self.rets_full,
                        self.metrics_to_optimize,
                        self.is_multi_objective,
                    ),
                )

            with executor_cls(**exec_kwargs) as executor:
                futures = []
                for window in self.windows:
                    # Safeguard against windows with fewer than two trading days –
                    # they cannot yield a valid return series.
                    if not _is_valid_window(self.daily_data.index, window):  # type: ignore[arg-type]
                        objective_values.append(np.nan)
                        full_pnl_returns.append(np.asarray([np.nan], dtype=float))
                        window_lengths.append(0)
                        continue
                    futures.append(executor.submit(_evaluate_window, window))

                for fut in as_completed(futures):
                    obj_val, pnl_ret = fut.result()
                    objective_values.append(obj_val)
                    full_pnl_returns.append(pnl_ret)
                    window_lengths.append(pnl_ret.shape[0] if isinstance(pnl_ret, np.ndarray) else len(pnl_ret))
        else:
            # Serial evaluation (safer for mocked objects and debug runs)
            for window in self.windows:
                if not _is_valid_window(self.daily_data.index, window):  # type: ignore[arg-type]
                    objective_values.append(-1e9)
                    full_pnl_returns.append(np.asarray([np.nan]))
                    window_lengths.append(0)
                    continue

                obj_val, pnl_ret = self.backtester.evaluate_fast_numba(
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
                if pnl_ret is not None:
                    if isinstance(pnl_ret, np.ndarray):
                        full_pnl_returns.append(pnl_ret)
                        window_lengths.append(pnl_ret.shape[0])
                    elif not pnl_ret.empty:
                        full_pnl_returns.append(pnl_ret.to_numpy())
                        window_lengths.append(len(pnl_ret))

        # ------------------------------------------------------------------
        # 3. Aggregate results across windows
        # ------------------------------------------------------------------
        if self.is_multi_objective:
            # Values are arrays/tuples; compute vector average element-wise.
            obj_matrix = np.asarray(objective_values, dtype=float)
            if self.scenario_config.get("aggregate_length_weighted", False) and len(window_lengths) == len(objective_values):
                weights = np.asarray(window_lengths, dtype=float)
                weights /= weights.sum()
                aggregate_objective = tuple(np.average(obj_matrix, axis=0, weights=weights))
            else:
                aggregate_objective = tuple(np.mean(obj_matrix, axis=0))
        else:
            if self.scenario_config.get("aggregate_length_weighted", False) and len(window_lengths) == len(objective_values):
                weights = np.asarray(window_lengths, dtype=float)
                weights /= weights.sum()
                aggregate_objective = float(np.average(np.asarray(objective_values, dtype=float), weights=weights))
            else:
                aggregate_objective = float(np.mean(objective_values, axis=0))

        aggregate_returns = np.concatenate(full_pnl_returns) if full_pnl_returns else np.array([], dtype=float)

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
