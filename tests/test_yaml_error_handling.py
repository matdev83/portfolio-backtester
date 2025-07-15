"""
Test cases for YAML error handling and validation.
"""

import unittest
import tempfile
import os
from pathlib import Path

from src.portfolio_backtester.yaml_validator import (
    YamlValidator, 
    YamlErrorType, 
    validate_yaml_file,
    lint_yaml_file
)
from src.portfolio_backtester.config_loader import ConfigurationError, load_config
from src.portfolio_backtester import config_loader


class TestYamlErrorHandling(unittest.TestCase):
    """Test YAML error handling and validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = YamlValidator()

    def test_valid_yaml_validation(self):
        """Test that valid YAML passes validation."""
        valid_yaml = """
GLOBAL_CONFIG:
  data_source: "hybrid"
  benchmark: "SPY"
  universe:
    - "AAPL"
    - "MSFT"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(valid_yaml)
            temp_file = f.name

        try:
            is_valid, data, errors = self.validator.validate_file(temp_file)
            self.assertTrue(is_valid)
            self.assertIsNotNone(data)
            self.assertEqual(len(errors), 0)
            self.assertIn("GLOBAL_CONFIG", data)
        finally:
            os.unlink(temp_file)

    def test_missing_colon_error(self):
        """Test detection of missing colon syntax error."""
        invalid_yaml = """
GLOBAL_CONFIG:
  data_source "hybrid"
  benchmark: "SPY"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_file = f.name

        try:
            is_valid, data, errors = self.validator.validate_file(temp_file)
            self.assertFalse(is_valid)
            self.assertIsNone(data)
            self.assertGreater(len(errors), 0)
            
            error = errors[0]
            self.assertEqual(error.error_type, YamlErrorType.SYNTAX_ERROR)
            self.assertIsNotNone(error.line_number)
            self.assertIn("colon", error.suggestion.lower())
        finally:
            os.unlink(temp_file)

    def test_unmatched_quotes_error(self):
        """Test detection of unmatched quotes."""
        invalid_yaml = """
GLOBAL_CONFIG:
  data_source: "hybrid
  benchmark: "SPY"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_file = f.name

        try:
            is_valid, data, errors = self.validator.validate_file(temp_file)
            self.assertFalse(is_valid)
            self.assertIsNone(data)
            self.assertGreater(len(errors), 0)
            
            error = errors[0]
            self.assertEqual(error.error_type, YamlErrorType.SYNTAX_ERROR)
        finally:
            os.unlink(temp_file)

    def test_tab_character_error(self):
        """Test detection of tab characters in YAML."""
        invalid_yaml = """
GLOBAL_CONFIG:
\tdata_source: "hybrid"
\tbenchmark: "SPY"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_file = f.name

        try:
            is_valid, data, errors = self.validator.validate_file(temp_file)
            self.assertFalse(is_valid)
            self.assertIsNone(data)
            self.assertGreater(len(errors), 0)
            
            error = errors[0]
            self.assertEqual(error.error_type, YamlErrorType.SYNTAX_ERROR)
            self.assertIn("tab", error.suggestion.lower())
        finally:
            os.unlink(temp_file)

    def test_file_not_found_error(self):
        """Test handling of missing files."""
        non_existent_file = "/path/that/does/not/exist.yaml"
        
        is_valid, data, errors = self.validator.validate_file(non_existent_file)
        self.assertFalse(is_valid)
        self.assertIsNone(data)
        self.assertGreater(len(errors), 0)
        
        error = errors[0]
        self.assertEqual(error.error_type, YamlErrorType.FILE_NOT_FOUND)

    def test_empty_yaml_file(self):
        """Test handling of empty YAML files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            temp_file = f.name

        try:
            is_valid, data, errors = self.validator.validate_file(temp_file)
            self.assertFalse(is_valid)
            self.assertIsNone(data)
            self.assertGreater(len(errors), 0)
            
            error = errors[0]
            self.assertEqual(error.error_type, YamlErrorType.VALIDATION_ERROR)
            self.assertIn("empty", error.message.lower())
        finally:
            os.unlink(temp_file)

    def test_convenience_functions(self):
        """Test convenience functions for YAML validation."""
        valid_yaml = """
test_key: "test_value"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(valid_yaml)
            temp_file = f.name

        try:
            # Test validate_yaml_file function
            is_valid, data, error_message = validate_yaml_file(temp_file)
            self.assertTrue(is_valid)
            self.assertIsNotNone(data)
            self.assertEqual(error_message, "")
            
            # Test lint_yaml_file function
            lint_result = lint_yaml_file(temp_file)
            self.assertIn("[OK]", lint_result)
            self.assertIn("valid", lint_result)
        finally:
            os.unlink(temp_file)

    def test_config_loader_integration(self):
        """Test that config_loader properly handles corrupted YAML."""
        # Create corrupted YAML content
        corrupted_yaml = """
GLOBAL_CONFIG:
  data_source "hybrid"  # Missing colon
  benchmark: "SPY"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(corrupted_yaml)
            corrupted_file = f.name

        try:
            # Backup original config
            original_params = config_loader.PARAMETERS_FILE
            
            # Point to corrupted file
            config_loader.PARAMETERS_FILE = Path(corrupted_file)
            
            # Reset module state
            config_loader.GLOBAL_CONFIG = {}
            config_loader.OPTIMIZER_PARAMETER_DEFAULTS = {}
            config_loader.BACKTEST_SCENARIOS = []
            
            # Test that load_config raises ConfigurationError
            with self.assertRaises(ConfigurationError) as cm:
                load_config()
            
            self.assertIn("Invalid parameters.yaml file", str(cm.exception))
            
        finally:
            # Restore original config
            config_loader.PARAMETERS_FILE = original_params
            os.unlink(corrupted_file)

    def test_error_formatting(self):
        """Test that error messages are properly formatted."""
        invalid_yaml = """
GLOBAL_CONFIG:
  data_source "hybrid"  # Missing colon here
  benchmark: "SPY"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_file = f.name

        try:
            is_valid, data, errors = self.validator.validate_file(temp_file)
            self.assertFalse(is_valid)
            
            formatted_errors = self.validator.format_errors(errors)
            
            # Check that formatted output contains expected elements
            self.assertIn("YAML CONFIGURATION ERROR(S) DETECTED", formatted_errors)
            self.assertIn("Error #1:", formatted_errors)
            self.assertIn("Type: syntax_error", formatted_errors)
            self.assertIn("Location:", formatted_errors)
            self.assertIn("Suggestion:", formatted_errors)
            self.assertIn("Context:", formatted_errors)
            
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()