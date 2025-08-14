import logging
from .constraint_handler import ConstraintHandler

logger = logging.getLogger(__name__)


def handle_constraints(
    backtester,
    scenario_config: dict,
    optimal_params: dict,
    full_rets,
    monthly_data,
    daily_data,
    rets_full,
):
    """Compatibility shim expected by `execution.py`.

    Attempts to use the new `ConstraintHandler` to verify/adjust parameters.
    Falls back to returning the original inputs unchanged when no constraints
    are defined or an error occurs.  The return signature mirrors the legacy
    implementation:

        (optimized_name, full_rets, optimal_params, constraint_status,
         constraint_message, constraint_violations, constraints_config)
    """

    logger = logging.getLogger(__name__)

    constraints_config = scenario_config.get("optimization_constraints", [])
    benchmark_ticker = backtester.global_config.get("benchmark", "SPY")

    # Prepare default return values
    optimized_name = scenario_config["name"]
    constraint_status = "OK"
    constraint_message: str = ""
    constraint_violations: list[str] = []

    if not constraints_config:
        return (
            optimized_name,
            full_rets,
            optimal_params,
            constraint_status,
            constraint_message,
            constraint_violations,
            constraints_config,
        )

    try:
        # Build benchmark returns for metrics evaluation
        bench_series = (
            backtester.rets_full[benchmark_ticker]
            if benchmark_ticker in backtester.rets_full
            else full_rets
        )
        handler = ConstraintHandler(backtester.global_config)
        adjusted_params, adjusted_rets, success = handler.find_constraint_satisfying_params(
            scenario_config,
            optimal_params,
            [],  # assume caller passes violations list later
            monthly_data,
            daily_data,
            rets_full,
            backtester.run_scenario,
            bench_series,
            benchmark_ticker,
        )
        if success and adjusted_params is not None and adjusted_rets is not None:
            optimal_params = adjusted_params
            full_rets = adjusted_rets
            constraint_status = "ADJUSTED"
            optimized_name += " (Constraints Adjusted)"
        else:
            constraint_status = "VIOLATED"
            constraint_message = "Unable to satisfy constraints with reasonable adjustments"
    except Exception as exc:  # noqa: BLE001
        logger.error("Constraint handling failed: %s", exc)
        constraint_status = "ERROR"
        constraint_message = str(exc)

    return (
        optimized_name,
        full_rets,
        optimal_params,
        constraint_status,
        constraint_message,
        constraint_violations,
        constraints_config,
    )
