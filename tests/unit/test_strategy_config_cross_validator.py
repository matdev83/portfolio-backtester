"""Unit tests for the strategy configuration cross-validator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from portfolio_backtester.strategy_config_cross_validator import (
    StrategyConfigCrossValidator,
    validate_strategy_config_cross_references,
)


class TestStrategyConfigCrossValidator(unittest.TestCase):
    """Test cases for the strategy configuration cross-validator."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories
        self.temp_dir = Path(tempfile.mkdtemp())
        self.src_strategies_dir = self.temp_dir / "src" / "portfolio_backtester" / "strategies"
        self.config_scenarios_dir = self.temp_dir / "config" / "scenarios"

        # Create directory structure
        for category in ["portfolio", "signal", "meta"]:
            (self.src_strategies_dir / category).mkdir(parents=True, exist_ok=True)
            (self.config_scenarios_dir / category).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_strategy_config(self, category: str, strategy_name: str, config_data: dict):
        """Helper to create a strategy config file."""
        config_dir = self.config_scenarios_dir / category / strategy_name
        config_dir.mkdir(parents=True, exist_ok=True)

        config_file = config_dir / "default.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

    def _create_strategy_file(self, category: str, filename: str):
        """Helper to create a strategy Python file."""
        strategy_file = self.src_strategies_dir / category / filename
        strategy_file.parent.mkdir(parents=True, exist_ok=True)
        strategy_file.touch()

    @patch('portfolio_backtester.strategies.registry.get_strategy_registry')
    @patch('portfolio_backtester.strategy_config_validator.validate_strategy_configs')
    def test_validate_cross_references_with_valid_strategies(self, mock_validate_configs, mock_get_registry):
        """Test cross-reference validation with all valid strategies."""
        # Mock basic validator success
        mock_validate_configs.return_value = (True, [])

        # Mock strategy registry
        mock_registry = Mock()
        mock_registry.get_all_strategies.return_value = {
            "MomentumPortfolioStrategy": Mock(),
            "EmaSignalStrategy": Mock(),
        }
        mock_get_registry.return_value = mock_registry

        # Create matching config files
        self._create_strategy_config(
            "portfolio",
            "momentum_portfolio_strategy",
            {
                "name": "momentum_test",
                "strategy": "momentum",
                "strategy_class": "MomentumPortfolioStrategy",
            },
        )

        validator = StrategyConfigCrossValidator(
            str(self.src_strategies_dir), str(self.config_scenarios_dir)
        )

        with patch.object(validator.strategy_resolver, "resolve_strategy") as mock_resolve:
            mock_resolve.return_value = Mock()
            is_valid, errors = validator.validate_cross_references()

        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    @patch('portfolio_backtester.strategies.registry.get_strategy_registry')
    @patch('portfolio_backtester.strategy_config_validator.validate_strategy_configs')
    def test_detect_stale_config_folders(self, mock_validate_configs, mock_get_registry):
        """Test detection of stale config folders."""
        # Mock basic validator success
        mock_validate_configs.return_value = (True, [])

        # Mock strategy registry with limited strategies
        mock_registry = Mock()
        mock_registry.get_all_strategies.return_value = {
            "MomentumPortfolioStrategy": Mock(),
        }
        mock_get_registry.return_value = mock_registry

        # Create a config for a non-existent strategy
        self._create_strategy_config(
            "portfolio",
            "nonexistent_strategy",
            {"name": "nonexistent_test", "strategy": "nonexistent_strategy"},
        )

        validator = StrategyConfigCrossValidator(
            str(self.src_strategies_dir), str(self.config_scenarios_dir)
        )

        with patch.object(validator.strategy_resolver, "resolve_strategy") as mock_resolve:
            mock_resolve.return_value = None  # Strategy doesn't exist
            is_valid, errors = validator.validate_cross_references()

        self.assertFalse(is_valid)
        self.assertTrue(any("Stale config folder detected" in error for error in errors))
        self.assertTrue(any("nonexistent_strategy" in error for error in errors))

    @patch('portfolio_backtester.strategies.registry.get_strategy_registry')
    @patch('portfolio_backtester.strategy_config_validator.validate_strategy_configs')
    def test_detect_invalid_strategy_references_in_config(self, mock_validate_configs, mock_get_registry):
        """Test detection of invalid strategy references in YAML configs."""
        # Mock basic validator success
        mock_validate_configs.return_value = (True, [])

        # Mock strategy registry
        mock_registry = Mock()
        mock_registry.get_all_strategies.return_value = {
            "MomentumPortfolioStrategy": Mock(),
        }
        mock_get_registry.return_value = mock_registry

        # Create config with invalid strategy reference
        self._create_strategy_config(
            "portfolio",
            "test_strategy",
            {
                "name": "test",
                "strategy": "invalid_strategy_name",
                "strategy_class": "InvalidStrategyClass",
            },
        )

        validator = StrategyConfigCrossValidator(
            str(self.src_strategies_dir), str(self.config_scenarios_dir)
        )

        with patch.object(validator.strategy_resolver, "resolve_strategy") as mock_resolve:
            mock_resolve.return_value = None  # Strategy doesn't exist
            is_valid, errors = validator.validate_cross_references()

        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid strategy reference" in error for error in errors))
        self.assertTrue(any("invalid_strategy_name" in error for error in errors))

    @patch('portfolio_backtester.strategies.registry.get_strategy_registry')
    @patch('portfolio_backtester.strategy_config_validator.validate_strategy_configs')
    def test_detect_invalid_meta_strategy_allocations(self, mock_validate_configs, mock_get_registry):
        """Test detection of invalid strategy references in meta-strategy allocations."""
        # Mock basic validator success
        mock_validate_configs.return_value = (True, [])

        # Mock strategy registry
        mock_registry = Mock()
        mock_registry.get_all_strategies.return_value = {
            "SimpleMetaStrategy": Mock(),
            "MomentumPortfolioStrategy": Mock(),
        }
        mock_get_registry.return_value = mock_registry

        # Create meta-strategy config with invalid allocation
        self._create_strategy_config(
            "meta",
            "test_meta_strategy",
            {
                "name": "test_meta",
                "strategy": "SimpleMetaStrategy",
                "strategy_params": {
                    "allocations": [
                        {"strategy_id": "momentum", "weight": 0.6},
                        {"strategy_id": "nonexistent_strategy", "weight": 0.4},  # Invalid
                    ]
                },
            },
        )

        validator = StrategyConfigCrossValidator(
            str(self.src_strategies_dir), str(self.config_scenarios_dir)
        )

        def mock_resolve_side_effect(strategy_name):
            if strategy_name in ["SimpleMetaStrategy", "momentum"]:
                return Mock()
            return None

        def mock_is_meta_side_effect(strategy_class):
            return strategy_class is not None

        with patch.object(
            validator.strategy_resolver, "resolve_strategy", side_effect=mock_resolve_side_effect
        ):
            with patch.object(
                validator.strategy_resolver,
                "is_meta_strategy",
                side_effect=mock_is_meta_side_effect,
            ):
                is_valid, errors = validator.validate_cross_references()

        self.assertFalse(is_valid)
        self.assertTrue(
            any(
                "Invalid strategy reference in meta-strategy allocation" in error
                for error in errors
            )
        )
        self.assertTrue(any("nonexistent_strategy" in error for error in errors))

    def test_class_name_to_snake_case(self):
        """Test conversion of class names to snake_case."""
        validator = StrategyConfigCrossValidator(
            str(self.src_strategies_dir), str(self.config_scenarios_dir)
        )

        test_cases = [
            ("MomentumPortfolioStrategy", "momentum_portfolio_strategy"),
            ("EMASignalStrategy", "e_m_a_signal_strategy"),
            ("SimpleStrategy", "simple_strategy"),
            ("TestABCStrategy", "test_a_b_c_strategy"),
        ]

        for class_name, expected in test_cases:
            with self.subTest(class_name=class_name):
                result = validator._class_name_to_snake_case(class_name)
                self.assertEqual(result, expected)

    def test_extract_base_strategy_name(self):
        """Test extraction of base strategy names from class names."""
        validator = StrategyConfigCrossValidator(
            str(self.src_strategies_dir), str(self.config_scenarios_dir)
        )

        test_cases = [
            ("MomentumPortfolioStrategy", "momentum"),
            ("EMASignalStrategy", "e_m_a"),
            ("SimpleMetaStrategy", "simple"),
            ("TestStrategy", "test"),
            ("NoSuffixClass", None),  # No recognized suffix
        ]

        for class_name, expected in test_cases:
            with self.subTest(class_name=class_name):
                result = validator._extract_base_strategy_name(class_name)
                self.assertEqual(result, expected)

    def test_check_strategy_code_references(self):
        """Test detection of strategy references in Python code."""
        # Create a Python file with strategy references
        test_file = self.src_strategies_dir / "meta" / "test_strategy.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)

        with open(test_file, "w") as f:
            f.write(
                """
# Test file with strategy references
strategy_name = "momentum_strategy"
other_ref = _resolve_strategy("nonexistent_strategy")
config = {"strategy": "another_nonexistent"}
"""
            )

        validator = StrategyConfigCrossValidator(
            str(self.src_strategies_dir), str(self.config_scenarios_dir)
        )

        with patch.object(
            validator, "_get_valid_strategy_names", return_value={"momentum_strategy"}
        ):
            with patch.object(validator.strategy_resolver, "resolve_strategy") as mock_resolve:

                def resolve_side_effect(name):
                    return Mock() if name == "momentum_strategy" else None

                mock_resolve.side_effect = resolve_side_effect
                errors = validator._check_strategy_code_references()

        # Should detect the invalid references
        self.assertTrue(len(errors) > 0)
        error_messages = " ".join(errors)
        self.assertIn("nonexistent_strategy", error_messages)

    def test_convenience_function(self):
        """Test the convenience function works correctly."""
        with patch(
            "portfolio_backtester.strategy_config_cross_validator.StrategyConfigCrossValidator"
        ) as mock_validator_class:
            mock_validator = Mock()
            mock_validator.validate_cross_references.return_value = (True, [])
            mock_validator_class.return_value = mock_validator

            is_valid, errors = validate_strategy_config_cross_references(
                str(self.src_strategies_dir), str(self.config_scenarios_dir)
            )

            self.assertTrue(is_valid)
            self.assertEqual(errors, [])
            mock_validator_class.assert_called_once_with(
                str(self.src_strategies_dir), str(self.config_scenarios_dir)
            )


if __name__ == "__main__":
    unittest.main()
