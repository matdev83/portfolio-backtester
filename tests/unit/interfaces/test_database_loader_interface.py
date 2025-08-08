"""
Tests for database loader interfaces implementing Dependency Inversion Principle.

This module tests the database loading abstractions that enable dependency inversion
for the CUSIP mapping system.
"""

import pytest
import tempfile
import csv
from pathlib import Path
from typing import Dict, Tuple

from portfolio_backtester.interfaces.database_loader_interface import (
    ISeedLoader,
    ILiveDBLoader,
    ILiveDBWriter,
    CsvSeedLoader,
    CsvLiveDBLoader,
    CsvLiveDBWriter,
    DatabaseLoaderFactory,
    create_seed_loader,
    create_live_db_loader,
    create_live_db_writer,
)


class TestDatabaseLoaderInterfaces:
    """Test database loader interfaces and implementations."""

    def test_seed_loader_interface_contract(self):
        """Test that ISeedLoader defines the correct interface."""
        # Verify abstract method exists
        assert hasattr(ISeedLoader, "load_seeds")
        assert hasattr(ISeedLoader, "__abstractmethods__")
        assert "load_seeds" in ISeedLoader.__abstractmethods__

    def test_live_db_loader_interface_contract(self):
        """Test that ILiveDBLoader defines the correct interface."""
        # Verify abstract method exists
        assert hasattr(ILiveDBLoader, "load_live_db")
        assert hasattr(ILiveDBLoader, "__abstractmethods__")
        assert "load_live_db" in ILiveDBLoader.__abstractmethods__

    def test_csv_seed_loader_implements_interface(self):
        """Test that CsvSeedLoader properly implements ISeedLoader."""
        loader = CsvSeedLoader()
        assert isinstance(loader, ISeedLoader)
        assert hasattr(loader, "load_seeds")

    def test_csv_live_db_loader_implements_interface(self):
        """Test that CsvLiveDBLoader properly implements ILiveDBLoader."""
        loader = CsvLiveDBLoader()
        assert isinstance(loader, ILiveDBLoader)
        assert hasattr(loader, "load_live_db")


class TestCsvSeedLoader:
    """Test CsvSeedLoader implementation."""

    def test_load_seeds_from_valid_csv(self):
        """Test loading seeds from valid CSV files."""
        # Create temporary seed files
        seed_data = [
            ["12345678", "AAPL", "Apple Inc."],
            ["87654321", "MSFT", "Microsoft Corporation"],
            ["11111111", "GOOGL", "Alphabet Inc."],
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            seed_file = Path(f.name)
            writer = csv.writer(f)
            writer.writerows(seed_data)

        try:
            cache: Dict[str, Tuple[str, str]] = {}
            loader = CsvSeedLoader()
            loader.load_seeds([seed_file], cache)

            # Verify cache was populated correctly
            assert len(cache) == 3
            assert cache["AAPL"] == ("12345678", "Apple Inc.")
            assert cache["MSFT"] == ("87654321", "Microsoft Corporation")
            assert cache["GOOGL"] == ("11111111", "Alphabet Inc.")

        finally:
            seed_file.unlink()

    def test_load_seeds_with_missing_file(self):
        """Test loading seeds when file doesn't exist."""
        cache: Dict[str, Tuple[str, str]] = {}
        loader = CsvSeedLoader()
        non_existent_file = Path("/tmp/non_existent_file.csv")

        # This should not raise an exception
        loader.load_seeds([non_existent_file], cache)

        # Cache should remain empty
        assert len(cache) == 0

    def test_load_seeds_with_invalid_cusip_length(self):
        """Test that seeds with invalid CUSIP length are ignored."""
        seed_data = [
            ["123", "AAPL", "Apple Inc."],  # Too short
            ["12345678", "MSFT", "Microsoft Corporation"],  # Valid
            ["1234567890", "GOOGL", "Alphabet Inc."],  # Too long
            ["123456789", "TSLA", "Tesla Inc."],  # Valid
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            seed_file = Path(f.name)
            writer = csv.writer(f)
            writer.writerows(seed_data)

        try:
            cache: Dict[str, Tuple[str, str]] = {}
            loader = CsvSeedLoader()
            loader.load_seeds([seed_file], cache)

            # Only valid CUSIPs should be loaded
            assert len(cache) == 2
            assert "MSFT" in cache
            assert "TSLA" in cache
            assert "AAPL" not in cache  # Too short CUSIP
            assert "GOOGL" not in cache  # Too long CUSIP

        finally:
            seed_file.unlink()

    def test_load_seeds_with_incomplete_rows(self):
        """Test that rows with insufficient data are ignored."""
        seed_data = [
            ["12345678"],  # Missing ticker
            ["12345678", "MSFT", "Microsoft Corporation"],  # Complete
            [],  # Empty row
            ["87654321", "GOOGL"],  # Missing name (should still work)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            seed_file = Path(f.name)
            writer = csv.writer(f)
            writer.writerows(seed_data)

        try:
            cache: Dict[str, Tuple[str, str]] = {}
            loader = CsvSeedLoader()
            loader.load_seeds([seed_file], cache)

            # Only valid rows should be loaded
            assert len(cache) == 2
            assert cache["MSFT"] == ("12345678", "Microsoft Corporation")
            assert cache["GOOGL"] == ("87654321", "")  # Empty name is OK

        finally:
            seed_file.unlink()

    def test_load_seeds_uses_setdefault(self):
        """Test that loader uses setdefault to not overwrite existing cache entries."""
        # Pre-populate cache
        cache: Dict[str, Tuple[str, str]] = {"AAPL": ("existing_cusip", "Existing Apple")}

        seed_data = [
            ["12345678", "AAPL", "New Apple Inc."],  # Should not overwrite
            ["87654321", "MSFT", "Microsoft Corporation"],  # Should add
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            seed_file = Path(f.name)
            writer = csv.writer(f)
            writer.writerows(seed_data)

        try:
            loader = CsvSeedLoader()
            loader.load_seeds([seed_file], cache)

            # Original AAPL entry should be preserved
            assert cache["AAPL"] == ("existing_cusip", "Existing Apple")
            # New MSFT entry should be added
            assert cache["MSFT"] == ("87654321", "Microsoft Corporation")

        finally:
            seed_file.unlink()


class TestCsvLiveDBLoader:
    """Test CsvLiveDBLoader implementation."""

    def test_load_live_db_from_valid_csv(self):
        """Test loading live database from valid CSV file."""
        live_db_data = [
            ["ticker", "cusip", "name", "source"],  # Header
            ["AAPL", "12345678", "Apple Inc.", "openfigi"],
            ["MSFT", "87654321", "Microsoft Corporation", "edgar"],
            ["GOOGL", "11111111", "Alphabet Inc.", "duckduckgo"],
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            live_db_file = Path(f.name)
            writer = csv.writer(f)
            writer.writerows(live_db_data)

        try:
            cache: Dict[str, Tuple[str, str]] = {}
            loader = CsvLiveDBLoader()
            loader.load_live_db(live_db_file, cache)

            # Verify cache was populated correctly
            assert len(cache) == 3
            assert cache["AAPL"] == ("12345678", "Apple Inc.")
            assert cache["MSFT"] == ("87654321", "Microsoft Corporation")
            assert cache["GOOGL"] == ("11111111", "Alphabet Inc.")

        finally:
            live_db_file.unlink()

    def test_load_live_db_with_missing_file(self):
        """Test loading live database when file doesn't exist."""
        cache: Dict[str, Tuple[str, str]] = {}
        loader = CsvLiveDBLoader()
        non_existent_file = Path("/tmp/non_existent_live_db.csv")

        # This should not raise an exception
        loader.load_live_db(non_existent_file, cache)

        # Cache should remain empty
        assert len(cache) == 0

    def test_load_live_db_with_missing_name_column(self):
        """Test loading live database when name column is missing."""
        live_db_data = [
            ["ticker", "cusip", "source"],  # Header without name
            ["AAPL", "12345678", "openfigi"],
            ["MSFT", "87654321", "edgar"],
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            live_db_file = Path(f.name)
            writer = csv.writer(f)
            writer.writerows(live_db_data)

        try:
            cache: Dict[str, Tuple[str, str]] = {}
            loader = CsvLiveDBLoader()
            loader.load_live_db(live_db_file, cache)

            # Entries should be loaded with empty names
            assert len(cache) == 2
            assert cache["AAPL"] == ("12345678", "")
            assert cache["MSFT"] == ("87654321", "")

        finally:
            live_db_file.unlink()

    def test_load_live_db_filters_invalid_entries(self):
        """Test that invalid entries (missing ticker or cusip) are ignored."""
        live_db_data = [
            ["ticker", "cusip", "name", "source"],  # Header
            ["AAPL", "12345678", "Apple Inc.", "openfigi"],  # Valid
            ["", "87654321", "No Ticker", "edgar"],  # Missing ticker
            ["MSFT", "", "No CUSIP", "duckduckgo"],  # Missing CUSIP
            ["GOOGL", "11111111", "Alphabet Inc.", "openfigi"],  # Valid
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            live_db_file = Path(f.name)
            writer = csv.writer(f)
            writer.writerows(live_db_data)

        try:
            cache: Dict[str, Tuple[str, str]] = {}
            loader = CsvLiveDBLoader()
            loader.load_live_db(live_db_file, cache)

            # Only valid entries should be loaded
            assert len(cache) == 2
            assert "AAPL" in cache
            assert "GOOGL" in cache
            assert "" not in cache  # Empty ticker should not be added

        finally:
            live_db_file.unlink()

    def test_load_live_db_overwrites_existing_cache(self):
        """Test that live database loader overwrites existing cache entries."""
        # Pre-populate cache
        cache: Dict[str, Tuple[str, str]] = {"AAPL": ("existing_cusip", "Existing Apple")}

        live_db_data = [
            ["ticker", "cusip", "name", "source"],  # Header
            ["AAPL", "12345678", "Apple Inc.", "openfigi"],  # Should overwrite
            ["MSFT", "87654321", "Microsoft Corporation", "edgar"],  # Should add
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            live_db_file = Path(f.name)
            writer = csv.writer(f)
            writer.writerows(live_db_data)

        try:
            loader = CsvLiveDBLoader()
            loader.load_live_db(live_db_file, cache)

            # AAPL entry should be updated
            assert cache["AAPL"] == ("12345678", "Apple Inc.")
            # New MSFT entry should be added
            assert cache["MSFT"] == ("87654321", "Microsoft Corporation")

        finally:
            live_db_file.unlink()


class TestDatabaseLoaderFactory:
    """Test DatabaseLoaderFactory."""

    def test_create_seed_loader(self):
        """Test factory creates seed loader."""
        loader = DatabaseLoaderFactory.create_seed_loader()
        assert isinstance(loader, CsvSeedLoader)
        assert isinstance(loader, ISeedLoader)

    def test_create_live_db_loader(self):
        """Test factory creates live database loader."""
        loader = DatabaseLoaderFactory.create_live_db_loader()
        assert isinstance(loader, CsvLiveDBLoader)
        assert isinstance(loader, ILiveDBLoader)

    def test_factory_functions(self):
        """Test module-level factory functions."""
        # Test seed loader factory function
        seed_loader = create_seed_loader()
        assert isinstance(seed_loader, ISeedLoader)

        # Test live db loader factory function
        live_db_loader = create_live_db_loader()
        assert isinstance(live_db_loader, ILiveDBLoader)

        # Test live db writer factory function
        live_db_writer = create_live_db_writer()
        assert isinstance(live_db_writer, ILiveDBWriter)

    def test_create_live_db_writer(self):
        """Test factory creates live database writer."""
        writer = DatabaseLoaderFactory.create_live_db_writer()
        assert isinstance(writer, CsvLiveDBWriter)
        assert isinstance(writer, ILiveDBWriter)


class TestDatabaseWriterInterfaces:
    """Test database writer interfaces and implementations."""

    def test_live_db_writer_interface_contract(self):
        """Test that ILiveDBWriter defines the correct interface."""
        # Verify abstract method exists
        assert hasattr(ILiveDBWriter, "append_to_db")
        assert hasattr(ILiveDBWriter, "__abstractmethods__")
        assert "append_to_db" in ILiveDBWriter.__abstractmethods__

    def test_csv_live_db_writer_implements_interface(self):
        """Test that CsvLiveDBWriter properly implements ILiveDBWriter."""
        writer = CsvLiveDBWriter()
        assert isinstance(writer, ILiveDBWriter)
        assert hasattr(writer, "append_to_db")


class TestCsvLiveDBWriter:
    """Test CsvLiveDBWriter implementation."""

    def test_append_to_db_creates_new_entry(self):
        """Test appending a new entry to the database."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            live_db_file = Path(f.name)
            # Create header
            writer_csv = csv.writer(f)
            writer_csv.writerow(["ticker", "cusip", "name", "source"])

        try:
            writer = CsvLiveDBWriter()
            writer.append_to_db(live_db_file, "AAPL", "037833100", "Apple Inc.", "manual")

            # Read the file and verify content
            with open(live_db_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                assert rows[0]["ticker"] == "AAPL"
                assert rows[0]["cusip"] == "037833100"
                assert rows[0]["name"] == "Apple Inc."
                assert rows[0]["source"] == "manual"

        finally:
            live_db_file.unlink()

    def test_append_to_db_multiple_entries(self):
        """Test appending multiple entries to the database."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            live_db_file = Path(f.name)
            # Create header
            writer_csv = csv.writer(f)
            writer_csv.writerow(["ticker", "cusip", "name", "source"])

        try:
            writer = CsvLiveDBWriter()
            writer.append_to_db(live_db_file, "AAPL", "037833100", "Apple Inc.", "manual")
            writer.append_to_db(live_db_file, "MSFT", "594918104", "Microsoft Corp.", "openfigi")
            writer.append_to_db(live_db_file, "GOOGL", "02079K305", "Alphabet Inc.", "edgar")

            # Read the file and verify content
            with open(live_db_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 3

                # Check all entries
                tickers = [row["ticker"] for row in rows]
                assert "AAPL" in tickers
                assert "MSFT" in tickers
                assert "GOOGL" in tickers

        finally:
            live_db_file.unlink()

    def test_append_to_db_handles_write_errors(self):
        """Test that write errors are handled gracefully."""
        # Use a path that doesn't exist or is read-only
        invalid_path = Path("/invalid/path/that/does/not/exist.csv")

        writer = CsvLiveDBWriter()

        # This should not raise an exception, just log a warning
        # We can't easily test the logging without complex mocking
        # but we can verify it doesn't crash
        try:
            writer.append_to_db(invalid_path, "TEST", "123456789", "Test Corp", "test")
            # If we get here without exception, the error handling worked
            assert True
        except Exception as e:
            # If an exception is raised, that's unexpected
            pytest.fail(f"append_to_db raised an unexpected exception: {e}")

    def test_append_to_db_empty_values(self):
        """Test appending entry with empty name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            live_db_file = Path(f.name)
            # Create header
            writer_csv = csv.writer(f)
            writer_csv.writerow(["ticker", "cusip", "name", "source"])

        try:
            writer = CsvLiveDBWriter()
            writer.append_to_db(live_db_file, "TEST", "123456789", "", "manual")

            # Read the file and verify content
            with open(live_db_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                assert rows[0]["ticker"] == "TEST"
                assert rows[0]["cusip"] == "123456789"
                assert rows[0]["name"] == ""  # Empty name should be preserved
                assert rows[0]["source"] == "manual"

        finally:
            live_db_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
