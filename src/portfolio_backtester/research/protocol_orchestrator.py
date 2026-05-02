"""Pluggable orchestrator entry point for research validation protocols."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Mapping

import pandas as pd

from ..canonical_config import CanonicalScenarioConfig
from .double_oos_wfo import DoubleOOSWFOProtocol
from .protocol_config import ResearchProtocolConfigError, parse_double_oos_wfo_protocol
from .results import ResearchProtocolResult


class ResearchProtocolOrchestrator:
    """Coordinates research validation protocol execution."""

    def __init__(
        self,
        optimization_orchestrator: Any,
        backtest_runner: Any,
        artifact_writer: Any | None = None,
        *,
        optimization_orchestrator_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize with optimization/backtest primitives used by protocols."""

        self._optimization_orchestrator = optimization_orchestrator
        self._backtest_runner = backtest_runner
        self._artifact_writer = artifact_writer
        self._optimization_orchestrator_factory = optimization_orchestrator_factory

    def run(
        self,
        *,
        scenario_config: CanonicalScenarioConfig,
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
        args: argparse.Namespace,
        global_config: Mapping[str, Any],
    ) -> ResearchProtocolResult:
        """Validate configuration then dispatch ``double_oos_wfo`` when enabled."""

        if "research_protocol" not in scenario_config.extras:
            msg = "missing research_protocol section"
            raise ResearchProtocolConfigError(msg)

        cli_proto = (
            str(getattr(args, "protocol", "double_oos_wfo")).strip().lower().replace("-", "_")
        )
        if cli_proto != "double_oos_wfo":
            msg = f"unsupported CLI research protocol: {cli_proto!r}"
            raise ResearchProtocolConfigError(msg)

        protocol_raw_any = scenario_config.extras["research_protocol"]
        if not isinstance(protocol_raw_any, Mapping):
            msg = "research_protocol must be a mapping"
            raise ResearchProtocolConfigError(msg)

        parsed = parse_double_oos_wfo_protocol({"research_protocol": protocol_raw_any})
        parsed.validate_for_mode("research_validate")

        executor = DoubleOOSWFOProtocol(
            self._optimization_orchestrator,
            self._backtest_runner,
            self._artifact_writer,
            optimization_orchestrator_factory=self._optimization_orchestrator_factory,
        )
        return executor.run(
            scenario_config=scenario_config,
            protocol_config=parsed,
            monthly_data=monthly_data,
            daily_data=daily_data,
            rets_full=rets_full,
            args=args,
            global_config=dict(global_config),
        )
