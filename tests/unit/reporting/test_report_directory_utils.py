"""
Unit tests for report_directory_utils module.
"""
import hashlib
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from portfolio_backtester.reporting.report_directory_utils import (
    create_report_directory,
    generate_content_hash,
    get_strategy_source_path,
)


class TestReportDirectoryUtils:
    """Test suite for report directory utilities."""

    def test_create_report_directory_without_hash(self):
        """Test directory creation without content hash."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / "reports"
            base_dir.mkdir()

            report_dir = create_report_directory(
                base_dir, "test_strategy", timestamp="20230101_120000"
            )

            assert report_dir.exists()
            assert report_dir.name == "test_strategy_20230101_120000"
            assert (report_dir / "plots").exists()
            assert (report_dir / "data").exists()

    def test_create_report_directory_with_hash(self):
        """Test directory creation with content hash."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / "reports"
            base_dir.mkdir()

            report_dir = create_report_directory(
                base_dir,
                "test_strategy",
                content_hash="abc123def456",
                timestamp="20230101_120000",
            )

            assert report_dir.exists()
            assert report_dir.name == "20230101_120000"
            assert report_dir.parent.name == "test_strategy_abc123def456"
            assert (report_dir / "plots").exists()
            assert (report_dir / "data").exists()

    def test_create_report_directory_default_timestamp(self):
        """Test directory creation with default timestamp."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / "reports"
            base_dir.mkdir()

            report_dir = create_report_directory(base_dir, "test_strategy")

            assert report_dir.exists()
            assert (report_dir / "plots").exists()
            assert (report_dir / "data").exists()

    def test_generate_content_hash_from_config_contents(self):
        """Test hash generation from config contents string."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "config.yaml"
            config_file.write_text("test: value\nkey: other")
            hash_value = generate_content_hash(
                config_file_path=config_file
            )

            assert isinstance(hash_value, str)
            assert len(hash_value) == 32  # MD5 produces 32 hex chars

    def test_generate_content_hash_from_config_file(self):
        """Test hash generation from config file path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "config.yaml"
            config_file.write_text("test: value\nkey: other")
            hash_value = generate_content_hash(
                config_file_path=config_file
            )

            assert isinstance(hash_value, str)
            assert len(hash_value) == 32

    def test_generate_content_hash_from_both(self):
        """Test hash generation from both strategy and config files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "config.yaml"
            config_file.write_text("test: value")
            strategy_file = Path(tmp_dir) / "strategy.py"
            strategy_file.write_text("class TestStrategy:\n    pass")

            hash_value = generate_content_hash(
                strategy_file_path=strategy_file, config_file_path=config_file
            )

            assert isinstance(hash_value, str)
            assert len(hash_value) == 32

    def test_generate_content_hash_missing_sources(self):
        """Test hash generation raises error when both sources are missing."""
        with pytest.raises(ValueError, match="Either strategy_file_path, strategy_class, config_file_path, or config_contents"):
            generate_content_hash()

    def test_different_content_different_hash(self):
        """Test that different content produces different hashes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file1 = Path(tmp_dir) / "config1.yaml"
            config_file1.write_text("test: value1")
            config_file2 = Path(tmp_dir) / "config2.yaml"
            config_file2.write_text("test: value2")

            hash1 = generate_content_hash(config_file_path=config_file1)
            hash2 = generate_content_hash(config_file_path=config_file2)

            assert hash1 != hash2

    def test_same_content_same_hash(self):
        """Test that same content produces same hash."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file1 = Path(tmp_dir) / "config1.yaml"
            config_file1.write_text("test: value")
            config_file2 = Path(tmp_dir) / "config2.yaml"
            config_file2.write_text("test: value")

            hash1 = generate_content_hash(config_file_path=config_file1)
            hash2 = generate_content_hash(config_file_path=config_file2)

            assert hash1 == hash2
