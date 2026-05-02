import argparse
import unittest
from unittest import mock
import sys

import pytest

from portfolio_backtester.backtester import _create_parser


def test_cli_help_text_encodes_on_cp1250() -> None:
    """Regression: Windows consoles often use cp1250; help must not use unmappable Unicode."""
    parser = _create_parser()
    help_text = parser.format_help()
    help_text.encode("cp1250")


class TestBacktesterCLI(unittest.TestCase):
    """
    Test suite for the backtester's command-line interface.
    """

    @mock.patch("portfolio_backtester.backtester.Backtester")
    @mock.patch("portfolio_backtester.config_loader.load_config")
    @mock.patch("argparse.ArgumentParser.parse_args")
    @mock.patch(
        "portfolio_backtester.config_loader.BACKTEST_SCENARIOS", new=[{"name": "test_scenario"}]
    )
    def test_cli_backtest_mode(self, mock_parse_args, mock_load_config, mock_backtester):
        """
        Test that the CLI correctly handles the 'backtest' mode.
        """
        # Arrange
        mock_args = mock.MagicMock()
        mock_args.mode = "backtest"
        mock_args.scenario_name = "test_scenario"
        mock_args.scenario_filename = None  # Add missing attribute
        mock_args.log_level = "INFO"
        mock_args.random_seed = 42
        mock_parse_args.return_value = mock_args

        # Act
        with mock.patch.object(
            sys,
            "argv",
            [
                "-m",
                "portfolio_backtester.backtester",
                "--mode",
                "backtest",
                "--scenario-name",
                "test_scenario",
            ],
        ):
            from portfolio_backtester import backtester as backtester_module

            backtester_module.main()

        # Assert
        mock_load_config.assert_called_once()
        mock_backtester.assert_called_once()
        _args, kwargs = mock_backtester.call_args
        self.assertEqual(kwargs.get("random_state"), 42)


def test_parser_accepts_research_validate_and_research_flags() -> None:
    parser = _create_parser()
    args = parser.parse_args(
        [
            "--mode",
            "research_validate",
            "--scenario-name",
            "scen",
            "--protocol",
            "double_oos_wfo",
            "--force-new-research-run",
            "--research-skip-unseen",
        ]
    )
    assert args.mode == "research_validate"
    assert args.protocol == "double_oos_wfo"
    assert args.force_new_research_run is True
    assert args.research_skip_unseen is True


def test_parser_research_validate_protocol_default() -> None:
    parser = _create_parser()
    args = parser.parse_args(["--mode", "research_validate", "--scenario-name", "scen"])
    assert args.protocol == "double_oos_wfo"
    assert args.force_new_research_run is False
    assert args.research_skip_unseen is False
    assert args.research_artifact_base_dir is None


def test_parser_research_artifact_base_dir() -> None:
    parser = _create_parser()
    args = parser.parse_args(
        [
            "--mode",
            "research_validate",
            "--scenario-name",
            "scen",
            "--research-artifact-base-dir",
            "D:/runs/research",
        ]
    )
    assert args.research_artifact_base_dir == "D:/runs/research"


def test_main_research_validate_requires_scenario_name_or_file() -> None:
    def _raising_error(self: argparse.ArgumentParser, message: str) -> None:
        raise RuntimeError(message)

    with mock.patch.object(argparse.ArgumentParser, "error", _raising_error):
        with mock.patch("portfolio_backtester.backtester.config_loader.load_config"):
            from portfolio_backtester import backtester as backtester_module

            with pytest.raises(
                RuntimeError,
                match="--scenario-name or --scenario-filename is required for "
                "'optimize' or 'research_validate' mode",
            ):
                backtester_module.main(["--mode", "research_validate", "--log-level", "ERROR"])


if __name__ == "__main__":
    unittest.main()
