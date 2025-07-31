"""
High-level runner that keeps the existing orchestrator/evaluator/backtester
intact while adding true multi-process trial evaluation via Optuna.

Why multi-process?
Optuna's built-in `n_jobs>1` uses a ThreadPoolExecutor which suffers from the
GIL and becomes a bottleneck for CPU-heavy vectorised/Numba code.  By launching
_independent Python processes_ that share the same study in a SQLite storage we
get true parallelism on Windows without hitting the GIL.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
from typing import Any, Dict, List

import optuna

from .optuna_objective_adapter import OptunaObjectiveAdapter
from .results import OptimizationData, OptimizationResult
from .utils import discrete_space_size
from .trial_deduplication import create_deduplicating_objective

logger = logging.getLogger(__name__)

###############################################################################
# Helper – worker entry-point                                                 #
###############################################################################

def _optuna_worker(
    scenario_config: Dict[str, Any],
    optimization_config: Dict[str, Any],
    data: OptimizationData,
    storage_url: str,
    study_name: str,
    n_trials: int,
    enable_deduplication: bool = True,
) -> None:
    """Run ``n_trials`` optimisation steps in *this* process.

    Each worker uses ``n_jobs=1`` (no threads) so multiple workers can execute
    in parallel without the GIL contention seen with ThreadPoolExecutor.
    """
    # Re-create / load the study from the shared storage.
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
    )

    base_objective = OptunaObjectiveAdapter(
        scenario_config=scenario_config,
        data=data,
        n_jobs=1,  # single-threaded inside the worker
    )
    
    # Wrap with deduplication if enabled
    if enable_deduplication:
        objective = create_deduplicating_objective(base_objective, enable_deduplication=True)
    else:
        objective = base_objective

    logger.info("Worker %d starting %d trials", os.getpid(), n_trials)
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,  # keep worker single-threaded
        show_progress_bar=False,
    )
    logger.info("Worker %d finished", os.getpid())


###############################################################################
# Main runner                                                                 #
###############################################################################

class ParallelOptimizationRunner:
    """Drop-in replacement for the old serial orchestrator.

    It coordinates a *process pool* (one process per CPU core by default) so
    that each process runs its slice of trials with ``n_jobs=1``.  The SQLite
    storage mediates synchronisation of the global study state between workers.
    """

    def __init__(
        self,
        scenario_config: Dict[str, Any],
        optimization_config: Dict[str, Any],
        data: OptimizationData,
        n_jobs: int = -1,
        storage_url: str = "sqlite:///optuna_studies.db",
        enable_deduplication: bool = True,
    ) -> None:
        self.scenario_config = scenario_config
        self.optimization_config = optimization_config
        self.data = data
        self.n_jobs = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        self.storage_url = storage_url or "sqlite:///optuna_studies.db"
        self.enable_deduplication = enable_deduplication

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run(self) -> OptimizationResult:
        study_name = f"{self.scenario_config['name']}_optuna"

        # Ensure study exists (idempotent).
        optuna.create_study(
            study_name=study_name,
            storage=self.storage_url,
            direction="maximize",
            load_if_exists=True,
        )

        # Determine number of trials to run in total.
        requested_trials: int = self.optimization_config.get("optuna_trials", 100)
        space_size = discrete_space_size(self.optimization_config.get("parameter_space", {}))
        if space_size is not None and requested_trials > space_size:
            logger.warning(
                "Parameter space has only %s unique combinations but %s trials were requested. "
                "Capping to %s.",
                space_size,
                requested_trials,
                space_size,
            )
            requested_trials = space_size

        # If only one job requested fall back to simple optimise call.
        if self.n_jobs == 1:
            logger.info("Running optimisation in a single process (%d trials)", requested_trials)
            _optuna_worker(
                self.scenario_config,
                self.optimization_config,
                self.data,
                self.storage_url,
                study_name,
                requested_trials,
                self.enable_deduplication,
            )
        else:
            logger.info("Launching %d worker processes for %d trials", self.n_jobs, requested_trials)
            trials_per_worker = math.ceil(requested_trials / self.n_jobs)

            ctx = mp.get_context("spawn")  # Safe on Windows
            processes: List[mp.Process] = []
            remaining = requested_trials
            for _ in range(self.n_jobs):
                if remaining <= 0:
                    break
                n_this = min(trials_per_worker, remaining)
                p = ctx.Process(
                    target=_optuna_worker,
                    args=(
                        self.scenario_config,
                        self.optimization_config,
                        self.data,
                        self.storage_url,
                        study_name,
                        n_this,
                        self.enable_deduplication,
                    ),
                )
                p.start()
                processes.append(p)
                remaining -= n_this

            # Wait for all workers to finish.
            for p in processes:
                p.join()

        # Load final study to extract best result.
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage_url,
            direction="maximize",
            load_if_exists=True,
        )
        best_params = study.best_params
        best_value = study.best_value
        return OptimizationResult(
            best_parameters=best_params,
            best_value=best_value,
            n_evaluations=len(study.trials),
            optimization_history=[],  # future enhancement
        )
