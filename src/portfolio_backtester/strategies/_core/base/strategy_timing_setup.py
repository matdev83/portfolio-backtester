"""Timing controller construction for :class:`BaseStrategy`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

if TYPE_CHECKING:
    from ....canonical_config import CanonicalScenarioConfig
    from ....timing.timing_controller import TimingController

logger = logging.getLogger(__name__)


def create_timing_controller_from_strategy_config(
    *,
    strategy_class_name: str,
    canonical_config: Optional["CanonicalScenarioConfig"],
    strategy_params: Dict[str, Any],
) -> "TimingController":
    """Build a timing controller, mutating ``strategy_params['timing_config']`` when defaulting.

    Args:
        strategy_class_name: Used for Signal-vs-portfolio style defaults when config is absent.
        canonical_config: Optional canonical scenario (preferred source for ``timing_config``).
        strategy_params: Mutable strategy parameters (legacy path and defaults).

    Returns:
        A concrete :class:`~portfolio_backtester.timing.timing_controller.TimingController`.
    """
    from ....timing.custom_timing_registry import TimingControllerFactory

    try:
        timing_config: Optional[Dict[str, Any]] = None
        if canonical_config and canonical_config.timing_config:
            timing_config = dict(canonical_config.timing_config)
        else:
            timing_config_val = strategy_params.get("timing_config")
            timing_config = (
                cast(Dict[str, Any], timing_config_val) if timing_config_val is not None else None
            )

        if timing_config is None:
            if "Signal" in strategy_class_name:
                timing_config = {
                    "mode": "signal_based",
                    "scan_frequency": "D",
                    "min_holding_period": 1,
                }
            else:
                timing_config = {
                    "mode": "time_based",
                    "rebalance_frequency": strategy_params.get("rebalance_frequency", "ME"),
                }
            strategy_params["timing_config"] = timing_config

        return TimingControllerFactory.create_controller(timing_config)

    except Exception as e:
        logger.error("Failed to initialize timing controller: %s", e)
        logger.info("Falling back to time-based timing with monthly frequency")
        fallback_config = {"mode": "time_based", "rebalance_frequency": "M"}
        controller = TimingControllerFactory.create_controller(fallback_config)
        strategy_params["timing_config"] = fallback_config
        return controller
