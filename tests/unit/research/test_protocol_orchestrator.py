"""Tests for ``ResearchProtocolOrchestrator`` research validation dispatch."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from portfolio_backtester.canonical_config import CanonicalScenarioConfig
from portfolio_backtester.research.protocol_config import (
    FinalUnseenMode,
    ResearchProtocolConfigError,
)
from portfolio_backtester.research.protocol_orchestrator import ResearchProtocolOrchestrator
from portfolio_backtester.research.results import ResearchProtocolResult

from tests.unit.research.test_protocol_config import _minimal_primary_inner


def _scenario_with_research(enabled: bool = True, **inner_overrides) -> CanonicalScenarioConfig:
    inner = _minimal_primary_inner()
    inner["enabled"] = enabled
    inner.update(inner_overrides)
    raw = {
        "name": "orch_scen",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {},
        "extras": {"research_protocol": inner},
    }
    return CanonicalScenarioConfig.from_dict(raw)


def test_orchestrator_missing_research_protocol_in_extras_raises() -> None:
    scen = CanonicalScenarioConfig.from_dict(
        {
            "name": "n",
            "strategy": "DummyStrategyForTestingSignalStrategy",
            "strategy_params": {},
            "extras": {},
        }
    )
    orch = ResearchProtocolOrchestrator(MagicMock(), MagicMock())
    args = argparse.Namespace(
        protocol="double_oos_wfo", force_new_research_run=False, research_skip_unseen=False
    )
    with pytest.raises(ResearchProtocolConfigError, match="research_protocol"):
        orch.run(
            scenario_config=scen,
            monthly_data=pd.DataFrame(),
            daily_data=pd.DataFrame(),
            rets_full=pd.DataFrame(),
            args=args,
            global_config={},
        )


def test_orchestrator_disabled_protocol_raises() -> None:
    scen = _scenario_with_research(enabled=False)
    orch = ResearchProtocolOrchestrator(MagicMock(), MagicMock())
    args = argparse.Namespace(
        protocol="double_oos_wfo", force_new_research_run=False, research_skip_unseen=False
    )
    with pytest.raises(ResearchProtocolConfigError, match="disabled"):
        orch.run(
            scenario_config=scen,
            monthly_data=pd.DataFrame(),
            daily_data=pd.DataFrame(),
            rets_full=pd.DataFrame(),
            args=args,
            global_config={},
        )


def test_orchestrator_cli_protocol_mismatch_raises() -> None:
    scen = _scenario_with_research()
    orch = ResearchProtocolOrchestrator(MagicMock(), MagicMock())
    args = argparse.Namespace(
        protocol="other_proto", force_new_research_run=False, research_skip_unseen=False
    )
    with pytest.raises(ResearchProtocolConfigError, match="protocol"):
        orch.run(
            scenario_config=scen,
            monthly_data=pd.DataFrame(),
            daily_data=pd.DataFrame(),
            rets_full=pd.DataFrame(),
            args=args,
            global_config={},
        )


def test_valid_protocol_dispatches_double_oos_run_once(tmp_path: Path) -> None:
    scen = _scenario_with_research()
    dummy_result = ResearchProtocolResult(
        scenario_name=scen.name,
        grid_results=(),
        selected_protocols=(),
        unseen_result=None,
        artifact_dir=tmp_path,
    )
    mock_impl = MagicMock()
    mock_impl.run.return_value = dummy_result

    opt = MagicMock()
    bt = MagicMock()

    with patch(
        "portfolio_backtester.research.protocol_orchestrator.DoubleOOSWFOProtocol",
        return_value=mock_impl,
    ) as ctor:
        orch = ResearchProtocolOrchestrator(opt, bt, artifact_writer=None)
        args = argparse.Namespace(
            protocol="double_oos_wfo",
            force_new_research_run=False,
            research_skip_unseen=False,
            optimizer="optuna",
        )
        gcfg = {"benchmark": "SPY"}
        out = orch.run(
            scenario_config=scen,
            monthly_data=pd.DataFrame(),
            daily_data=pd.DataFrame(),
            rets_full=pd.DataFrame(),
            args=args,
            global_config=gcfg,
        )

    ctor.assert_called_once_with(opt, bt, None)
    mock_impl.run.assert_called_once()
    (_, kwargs) = mock_impl.run.call_args
    assert kwargs["scenario_config"] is scen
    assert kwargs["monthly_data"] is not None
    assert kwargs["args"] is args
    assert kwargs["global_config"] == gcfg
    assert kwargs["protocol_config"].final_unseen_mode == FinalUnseenMode.FIXED_SELECTED_PARAMS
    assert out is dummy_result
