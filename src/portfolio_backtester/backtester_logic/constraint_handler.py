"""
Constraint Handler for Portfolio Backtester

This module provides functionality to handle constraint violations
by adjusting parameters to meet specified constraints.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from ..reporting.performance_metrics import calculate_metrics

logger = logging.getLogger(__name__)


class ConstraintHandler:
    """Handles constraint violations by adjusting parameters."""

    def __init__(self, global_config: Dict) -> None:
        self.global_config = global_config

    def find_constraint_satisfying_params(
        self,
        scenario_config: Dict,
        optimal_params: Dict,
        constraint_violations: List[str],
        monthly_data,
        daily_data,
        rets_full,
        run_scenario_func,
        benchmark_returns: pd.Series,
        benchmark_ticker: str,
        max_attempts: int = 10,
    ) -> Tuple[Optional[Dict], Optional[pd.Series], bool]:
        """
        Try to find parameters that satisfy constraints by adjusting the violating parameters.

        Returns:
            (adjusted_params, adjusted_returns, success)
        """
        constraints_config = scenario_config.get("optimization_constraints", [])
        if not constraints_config:
            return optimal_params, None, True

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Attempting to find constraint-satisfying parameters (max {max_attempts} attempts)"
            )

        # Start with optimal params and iteratively adjust
        current_params = optimal_params.copy()

        for attempt in range(max_attempts):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Constraint adjustment attempt {attempt + 1}/{max_attempts}")

            # Adjust parameters based on constraint violations
            adjusted_params = self._adjust_parameters_for_constraints(
                current_params, constraint_violations, scenario_config, attempt
            )

            if adjusted_params is None:
                logger.warning(f"Could not adjust parameters for attempt {attempt + 1}")
                continue

            # Test the adjusted parameters
            test_scenario = scenario_config.copy()
            test_scenario["strategy_params"] = adjusted_params

            test_rets = run_scenario_func(test_scenario, monthly_data, daily_data, rets_full)

            if test_rets is None or test_rets.empty:
                logger.warning(f"Attempt {attempt + 1}: Backtest failed with adjusted parameters")
                current_params = adjusted_params  # Try further adjustments
                continue

            # Check if constraints are satisfied
            test_metrics = calculate_metrics(test_rets, benchmark_returns, benchmark_ticker)
            test_violations = self._check_constraint_violations(test_metrics, constraints_config)

            if not test_violations:
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        f"✅ Found constraint-satisfying parameters on attempt {attempt + 1}"
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Adjusted parameters: {adjusted_params}")
                return adjusted_params, test_rets, True
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Attempt {attempt + 1}: Still have violations: {test_violations}")
                current_params = adjusted_params  # Use as starting point for next iteration

        logger.error(
            f"Failed to find constraint-satisfying parameters after {max_attempts} attempts"
        )
        return None, None, False

    def _adjust_parameters_for_constraints(
        self, params: Dict, violations: List[str], scenario_config: Dict, attempt: int
    ) -> Optional[Dict]:
        """
        Adjust parameters to try to satisfy constraints.

        Current strategy:
        - If volatility is too high, reduce leverage
        - If returns are too low, try to increase other parameters
        """
        adjusted = params.copy()

        # Parse violations to understand what needs adjustment
        volatility_too_high = any("Ann. Vol" in v and ">" in v for v in violations)

        if volatility_too_high:
            # Reduce leverage to lower volatility; allow dropping below 1.0 if necessary
            current_leverage = adjusted.get("leverage", 1.0)

            # Determine the minimum leverage allowed (default 0.1, but can be overridden in scenario_config)
            min_leverage_allowed = scenario_config.get("min_leverage_allowed", 0.1)

            # Apply progressive reduction factor each attempt
            reduction_factor = 0.7 ** (attempt + 1)  # 0.7, 0.49, 0.343, etc.
            new_leverage = max(min_leverage_allowed, current_leverage * reduction_factor)

            # Only update if leverage actually changes to avoid infinite loops
            if abs(new_leverage - current_leverage) > 1e-6:
                adjusted["leverage"] = new_leverage
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Reducing leverage from {current_leverage:.3f} to {new_leverage:.3f}"
                    )
            else:
                # If leverage cannot be reduced further, fall back to other parameter tweaks
                return self._try_other_adjustments(adjusted, scenario_config, attempt)

        return adjusted

    def _try_other_adjustments(
        self, params: Dict, scenario_config: Dict, attempt: int
    ) -> Optional[Dict]:
        """Try other parameter adjustments when leverage can't be reduced further."""
        adjusted = params.copy()

        # For EMA strategy, try to make it less aggressive
        if "fast_ema_days" in adjusted and "slow_ema_days" in adjusted:
            # Increase fast EMA (make it slower) to reduce trading frequency
            current_fast = adjusted["fast_ema_days"]
            current_slow = adjusted["slow_ema_days"]

            # Increase fast EMA but keep it below slow EMA
            max_fast = min(current_slow - 5, current_fast + 5 + attempt * 2)
            if max_fast > current_fast:
                adjusted["fast_ema_days"] = max_fast
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Increasing fast EMA from {current_fast} to {max_fast} to reduce volatility"
                    )
                return adjusted

        # If no other adjustments possible, return None
        logger.warning("No further parameter adjustments possible")
        return None

    def _check_constraint_violations(
        self, metrics: Dict, constraints_config: List[Dict]
    ) -> List[str]:
        """Check if metrics violate any constraints."""
        violations = []

        for constraint in constraints_config:
            metric_name = constraint.get("metric")
            min_value = constraint.get("min_value")
            max_value = constraint.get("max_value")

            if metric_name and (min_value is not None or max_value is not None):
                metric_val = metrics.get(metric_name)

                if metric_val is not None and not pd.isna(metric_val):
                    if min_value is not None and metric_val < min_value:
                        violations.append(f"{metric_name} = {metric_val:.4f} < {min_value} (min)")
                    if max_value is not None and metric_val > max_value:
                        violations.append(f"{metric_name} = {metric_val:.4f} > {max_value} (max)")

        return violations
