import unittest
from unittest import mock
import sys

class TestBacktesterCLI(unittest.TestCase):
    """
    Test suite for the backtester's command-line interface.
    """

    @mock.patch('src.portfolio_backtester.backtester.Backtester')
    @mock.patch('src.portfolio_backtester.config_loader.load_config')
    @mock.patch('argparse.ArgumentParser.parse_args')
    @mock.patch('src.portfolio_backtester.config_loader.BACKTEST_SCENARIOS', new=[{"name": "test_scenario"}])
    def test_cli_backtest_mode(self, mock_parse_args, mock_load_config, mock_backtester):
        """
        Test that the CLI correctly handles the 'backtest' mode.
        """
        # Arrange
        mock_args = mock.MagicMock()
        mock_args.mode = 'backtest'
        mock_args.scenario_name = 'test_scenario'
        mock_args.scenario_filename = None  # Add missing attribute
        mock_args.log_level = 'INFO'
        mock_args.random_seed = 42
        mock_parse_args.return_value = mock_args

        # Act
        with mock.patch.object(sys, 'argv', ['-m', 'src.portfolio_backtester.backtester', '--mode', 'backtest', '--scenario-name', 'test_scenario']):
            from src.portfolio_backtester import backtester as backtester_module
            backtester_module.main()

        # Assert
        mock_load_config.assert_called_once()
        mock_backtester.assert_called_once()
        # More specific assertions can be added here

if __name__ == '__main__':
    unittest.main()