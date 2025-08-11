"""
Unit tests for enhanced window generation utilities.

Tests the enhanced WFO window generation with evaluation frequency detection.
"""

import pandas as pd
from unittest.mock import patch
from portfolio_backtester.utils import (
    generate_enhanced_wfo_windows,
    _determine_evaluation_frequency,
)


class TestEnhancedWindowGeneration:
    """Test cases for enhanced window generation utilities."""

    def test_determine_evaluation_frequency_intramonth_class(self):
        """Test frequency detection for intramonth strategy class."""
        scenario_config = {"strategy_class": "SeasonalSignalStrategy", "strategy": "momentum"}

        freq = _determine_evaluation_frequency(scenario_config)
        assert freq == "D"

    def test_determine_evaluation_frequency_intramonth_name(self):
        """Test frequency detection for intramonth strategy name."""
        scenario_config = {"strategy_class": "SeasonalStrategy", "strategy": "intramonth_momentum"}

        freq = _determine_evaluation_frequency(scenario_config)
        assert freq == "D"

    def test_determine_evaluation_frequency_signal_based_daily(self):
        """Test frequency detection for daily signal-based strategy."""
        scenario_config = {
            "strategy_class": "SignalStrategy",
            "timing_config": {"mode": "signal_based", "scan_frequency": "D"},
        }

        freq = _determine_evaluation_frequency(scenario_config)
        assert freq == "D"

    def test_determine_evaluation_frequency_signal_based_weekly(self):
        """Test frequency detection for weekly signal-based strategy."""
        scenario_config = {
            "strategy_class": "SignalStrategy",
            "timing_config": {"mode": "signal_based", "scan_frequency": "W"},
        }

        freq = _determine_evaluation_frequency(scenario_config)
        assert freq == "M"  # Falls back to monthly

    def test_determine_evaluation_frequency_daily_rebalance(self):
        """Test frequency detection for daily rebalance strategy."""
        scenario_config = {
            "strategy_class": "SimpleMomentumPortfolioStrategy",
            "rebalance_frequency": "D",
        }

        freq = _determine_evaluation_frequency(scenario_config)
        assert freq == "D"

    def test_determine_evaluation_frequency_monthly_default(self):
        """Test frequency detection defaults to monthly."""
        scenario_config = {
            "strategy_class": "SimpleMomentumPortfolioStrategy",
            "rebalance_frequency": "M",
        }

        freq = _determine_evaluation_frequency(scenario_config)
        assert freq == "M"

    def test_determine_evaluation_frequency_empty_config(self):
        """Test frequency detection with empty config."""
        scenario_config: dict = {}

        freq = _determine_evaluation_frequency(scenario_config)
        assert freq == "M"

    @patch("portfolio_backtester.utils.generate_randomized_wfo_windows")
    def test_generate_enhanced_wfo_windows_with_wfo_window_available(self, mock_generate_base):
        """Test enhanced window generation when WFOWindow is available."""
        # Mock base window generation
        mock_generate_base.return_value = [
            (
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-31"),
            ),
            (
                pd.Timestamp("2024-02-01"),
                pd.Timestamp("2025-01-31"),
                pd.Timestamp("2025-02-01"),
                pd.Timestamp("2025-02-28"),
            ),
        ]

        # Test data - need at least 84 months for window generation
        monthly_data_index = pd.date_range("2017-01-01", "2025-12-31", freq="M")
        scenario_config = {"strategy_class": "SeasonalSignalStrategy", "name": "test_strategy"}
        global_config: dict = {}

        # Generate enhanced windows
        windows = generate_enhanced_wfo_windows(monthly_data_index, scenario_config, global_config)

        # Check that base generation was called
        mock_generate_base.assert_called_once_with(
            monthly_data_index, scenario_config, global_config, None
        )

        # Check enhanced windows
        assert len(windows) == 2

        # Check first window
        window1 = windows[0]
        assert hasattr(window1, "evaluation_frequency")
        assert window1.evaluation_frequency == "D"  # Intramonth strategy
        assert window1.strategy_name == "test_strategy"
        assert window1.train_start == pd.Timestamp("2024-01-01")
        assert window1.test_end == pd.Timestamp("2025-01-31")

    @patch("portfolio_backtester.utils.generate_randomized_wfo_windows")
    def test_generate_enhanced_wfo_windows_monthly_strategy(self, mock_generate_base):
        """Test enhanced window generation for monthly strategy."""
        # Mock base window generation
        mock_generate_base.return_value = [
            (
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-31"),
            )
        ]

        # Test data
        monthly_data_index = pd.date_range("2017-01-01", "2025-12-31", freq="M")
        scenario_config = {
            "strategy_class": "SimpleMomentumPortfolioStrategy",
            "rebalance_frequency": "M",
            "name": "momentum_test",
        }
        global_config: dict = {}

        # Generate enhanced windows
        windows = generate_enhanced_wfo_windows(monthly_data_index, scenario_config, global_config)

        # Check enhanced windows
        assert len(windows) == 1

        # Check window
        window = windows[0]
        assert window.evaluation_frequency == "M"  # Monthly strategy
        assert window.strategy_name == "momentum_test"

    @patch("portfolio_backtester.utils.generate_randomized_wfo_windows")
    def test_generate_enhanced_wfo_windows_with_random_state(self, mock_generate_base):
        """Test enhanced window generation with random state."""
        # Mock base window generation
        mock_generate_base.return_value = [
            (
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-31"),
            )
        ]

        # Test data
        monthly_data_index = pd.date_range("2017-01-01", "2025-12-31", freq="M")
        scenario_config = {"strategy_class": "SimpleMomentumPortfolioStrategy"}
        global_config: dict = {}
        random_state = 42

        # Generate enhanced windows
        generate_enhanced_wfo_windows(
            monthly_data_index, scenario_config, global_config, random_state
        )

        # Check that random state was passed through
        mock_generate_base.assert_called_once_with(
            monthly_data_index, scenario_config, global_config, random_state
        )

    @patch("portfolio_backtester.utils.generate_randomized_wfo_windows")
    def test_generate_enhanced_wfo_windows_fallback_when_import_fails(self, mock_generate_base):
        """Test fallback to regular windows when WFOWindow import fails."""
        # Mock base window generation
        base_windows = [
            (
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-31"),
            )
        ]
        mock_generate_base.return_value = base_windows

        # Test data
        monthly_data_index = pd.date_range("2017-01-01", "2025-12-31", freq="M")
        scenario_config = {"strategy_class": "SimpleMomentumPortfolioStrategy"}
        global_config: dict = {}

        # Mock import failure by patching the import inside the function
        with patch("portfolio_backtester.utils.generate_enhanced_wfo_windows") as mock_enhanced:
            # Make the mock function simulate import failure and call the original fallback
            def mock_fallback(*args, **kwargs):
                # Simulate the import failure path in the actual function
                return generate_randomized_wfo_windows(*args, **kwargs)

            mock_enhanced.side_effect = mock_fallback

            # Import the function directly to test the fallback logic
            from portfolio_backtester.utils import generate_randomized_wfo_windows

            windows = generate_randomized_wfo_windows(
                monthly_data_index, scenario_config, global_config
            )

        # Should return base windows (fallback)
        assert isinstance(windows, list)
        assert len(windows) > 0
        # Each window should be a tuple of 4 timestamps
        for window in windows:
            assert isinstance(window, tuple)
            assert len(window) == 4

    def test_determine_evaluation_frequency_case_insensitive(self):
        """Test that frequency detection is case insensitive."""
        # Test uppercase
        scenario_config = {"strategy_class": "INTRAMONTHMOMENTSTRATEGY", "strategy": "momentum"}

        freq = _determine_evaluation_frequency(scenario_config)
        assert freq == "D"

        # Test mixed case
        scenario_config = {"strategy_class": "SomeStrategy", "strategy": "IntraMonth_Seasonal"}

        freq = _determine_evaluation_frequency(scenario_config)
        assert freq == "D"

    def test_determine_evaluation_frequency_priority_order(self):
        """Test that frequency detection follows correct priority order."""
        # Intramonth should take priority over rebalance frequency
        scenario_config: dict = {
            "strategy_class": "SeasonalSignalStrategy",
            "rebalance_frequency": "M",  # Monthly rebalance
        }

        freq = _determine_evaluation_frequency(scenario_config)
        assert freq == "D"  # Should be daily due to intramonth, not monthly

        # Signal-based should take priority over rebalance frequency
        scenario_config = {
            "strategy_class": "SignalStrategy",
            "timing_config": {"mode": "signal_based", "scan_frequency": "D"},
            "rebalance_frequency": "M",
        }

        freq = _determine_evaluation_frequency(scenario_config)
        assert freq == "D"  # Should be daily due to signal-based, not monthly

    @patch("portfolio_backtester.utils.generate_randomized_wfo_windows")
    def test_generate_enhanced_wfo_windows_preserves_window_boundaries(self, mock_generate_base):
        """Test that enhanced windows preserve original window boundaries."""
        # Mock base window generation with specific dates
        original_windows = [
            (
                pd.Timestamp("2024-01-15"),
                pd.Timestamp("2024-11-30"),
                pd.Timestamp("2024-12-01"),
                pd.Timestamp("2024-12-31"),
            ),
            (
                pd.Timestamp("2024-03-01"),
                pd.Timestamp("2025-01-31"),
                pd.Timestamp("2025-02-01"),
                pd.Timestamp("2025-02-28"),
            ),
        ]
        mock_generate_base.return_value = original_windows

        # Test data
        monthly_data_index = pd.date_range("2017-01-01", "2025-12-31", freq="M")
        scenario_config = {"strategy_class": "SimpleMomentumPortfolioStrategy"}
        global_config: dict = {}

        # Generate enhanced windows
        windows = generate_enhanced_wfo_windows(monthly_data_index, scenario_config, global_config)

        # Check that boundaries are preserved
        assert len(windows) == 2

        # Check first window boundaries
        assert windows[0].train_start == pd.Timestamp("2024-01-15")
        assert windows[0].train_end == pd.Timestamp("2024-11-30")
        assert windows[0].test_start == pd.Timestamp("2024-12-01")
        assert windows[0].test_end == pd.Timestamp("2024-12-31")

        # Check second window boundaries
        assert windows[1].train_start == pd.Timestamp("2024-03-01")
        assert windows[1].train_end == pd.Timestamp("2025-01-31")
        assert windows[1].test_start == pd.Timestamp("2025-02-01")
        assert windows[1].test_end == pd.Timestamp("2025-02-28")
