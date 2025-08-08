import logging
from typing import Any, Optional, Protocol, Sequence

# Optimizer imports (optuna is optional; guard at import and call sites)
try:
    from ..optimization.optuna_optimizer import OptunaOptimizer  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency path
    OptunaOptimizer = None

from ..optimization.genetic_optimizer import GeneticOptimizer


# Define a minimal runtime Protocol-like base to help mypy without importing heavy deps
class _BaseOptimizerLike(Protocol):
    def optimize(self) -> Sequence[Any]: ...


class _OptimizerProto(_BaseOptimizerLike, Protocol):
    # Avoid PEP 646 unpack for Python < 3.11; use a Sequence and index later
    def optimize(self) -> Sequence[Any]: ...


# TESTING NOTE: When testing optimization functions, be aware that Mock objects
# may be passed as timeout values or other numeric parameters. The TimeoutManager
# class in core.py handles this with defensive programming using try-catch blocks
# to prevent TypeError exceptions when Mock objects are used in numeric operations.

# Global progress tracker for optimization
_global_progress_tracker = None


def get_optimizer(
    optimizer_type: str,
    scenario_config: dict[str, Any],
    backtester_instance: Any,
    global_config: dict[str, Any],
    monthly_data: Any,
    daily_data: Any,
    rets_full: Any,
    random_state: Optional[int],
) -> _OptimizerProto:
    """Factory function to create optimizer instances."""
    if optimizer_type == "genetic":
        return GeneticOptimizer(
            scenario_config=scenario_config,
            backtester_instance=backtester_instance,
            global_config=global_config,
            monthly_data=monthly_data,
            daily_data=daily_data,
            rets_full=rets_full,
            random_state=random_state,
        )
    elif optimizer_type == "optuna":
        if OptunaOptimizer is None:
            raise ImportError(
                "Optuna optimizer is not available in this build. Ensure optuna optimizer module is present."
            )
        # Construct and assert structural compatibility with _OptimizerProto
        opt = OptunaOptimizer(
            scenario_config=scenario_config,
            backtester_instance=backtester_instance,
            global_config=global_config,
            monthly_data=monthly_data,
            daily_data=daily_data,
            rets_full=rets_full,
            random_state=random_state,
        )
        # Runtime/typing guard to satisfy mypy without ignores
        assert hasattr(opt, "optimize"), "OptunaOptimizer must define optimize()"
        # Narrow the type using cast to the protocol after the runtime check
        from typing import cast

        return cast(_OptimizerProto, opt)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def run_optimization(
    self: Any,
    scenario_config: dict[str, Any],
    monthly_data: Any,
    daily_data: Any,
    rets_full: Any,
) -> tuple[Any, Any]:
    global _global_progress_tracker

    optimizer_type = self.global_config.get("optimizer_config", {}).get("optimizer_type", "optuna")
    if self.logger.isEnabledFor(logging.DEBUG):
        self.logger.debug(
            f"Running {optimizer_type} optimization for scenario: {scenario_config['name']} with walk-forward splits."
        )

    optimizer = get_optimizer(
        optimizer_type,
        scenario_config,
        self,
        self.global_config,
        monthly_data,
        daily_data,
        rets_full,
        self.random_state,
    )

    # Call the optimize method and handle the return values
    result = optimizer.optimize()

    # Normalize to first two elements; accept 2+ length
    try:
        first, second = result[0], result[1]
    except Exception as exc:
        raise ValueError("Optimizer.optimize() returned unexpected number of elements") from exc
    return first, second
