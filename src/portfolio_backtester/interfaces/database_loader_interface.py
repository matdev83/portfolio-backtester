"""
Database loading interfaces for CUSIP mapping persistence operations.

This module provides polymorphic interfaces to implement Dependency Inversion Principle
for database loading operations in the CUSIP mapping system.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple


class ISeedLoader(ABC):
    """
    Interface for loading seed data into the CUSIP mapping cache.

    This interface abstracts the seed file loading functionality,
    enabling dependency inversion for the CusipMappingDB class.
    """

    @abstractmethod
    def load_seeds(self, seed_files: List[Path], cache: Dict[str, Tuple[str, str]]) -> None:
        """
        Load seed data from files into the cache.

        Parameters
        ----------
        seed_files : List[Path]
            List of paths to seed files to load
        cache : Dict[str, Tuple[str, str]]
            Cache dictionary to populate with ticker -> (cusip, name) mappings
        """
        pass


class ILiveDBLoader(ABC):
    """
    Interface for loading live database data into the CUSIP mapping cache.

    This interface abstracts the live database loading functionality,
    enabling dependency inversion for the CusipMappingDB class.
    """

    @abstractmethod
    def load_live_db(self, live_db_file: Path, cache: Dict[str, Tuple[str, str]]) -> None:
        """
        Load live database data from file into the cache.

        Parameters
        ----------
        live_db_file : Path
            Path to the live database file to load
        cache : Dict[str, Tuple[str, str]]
            Cache dictionary to populate with ticker -> (cusip, name) mappings
        """
        pass


class CsvSeedLoader(ISeedLoader):
    """Concrete implementation for loading seed data from CSV files."""

    def load_seeds(self, seed_files: List[Path], cache: Dict[str, Tuple[str, str]]) -> None:
        """Load seed data from CSV files into the cache."""
        import csv
        import logging

        logger = logging.getLogger(__name__)

        for path in seed_files:
            if not path.exists():
                continue
            try:
                with path.open() as fh:
                    reader = csv.reader(fh)
                    for row in reader:
                        if len(row) < 2:
                            continue
                        cusip = row[0]
                        ticker = row[1]
                        name = row[2] if len(row) >= 3 else ""
                        ticker = ticker.strip().upper()
                        cusip = cusip.strip()
                        name = name.strip()
                        if ticker and cusip and 8 <= len(cusip) <= 9:
                            cache.setdefault(ticker, (cusip, name))
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Could not read seed %s: %s", path, e)


class CsvLiveDBLoader(ILiveDBLoader):
    """Concrete implementation for loading live database data from CSV files."""

    def load_live_db(self, live_db_file: Path, cache: Dict[str, Tuple[str, str]]) -> None:
        """Load live database data from CSV file into the cache."""
        import csv
        import logging

        logger = logging.getLogger(__name__)

        try:
            with live_db_file.open() as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    ticker = row["ticker"].upper()
                    cusip = row["cusip"]
                    name = row.get("name", "")
                    if ticker and cusip:
                        cache[ticker] = (cusip, name)
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Could not read live DB: %s", e)


class ILiveDBWriter(ABC):
    """
    Interface for writing data to the live database.

    This interface abstracts the live database writing functionality,
    enabling dependency inversion for the CusipMappingDB class.
    """

    @abstractmethod
    def append_to_db(
        self, live_db_file: Path, ticker: str, cusip: str, name: str, source: str
    ) -> None:
        """
        Append a new CUSIP mapping to the live database.

        Parameters
        ----------
        live_db_file : Path
            Path to the live database file
        ticker : str
            Stock ticker symbol
        cusip : str
            CUSIP identifier
        name : str
            Company name
        source : str
            Source of the mapping (e.g., 'openfigi', 'edgar', 'manual')
        """
        pass


class CsvLiveDBWriter(ILiveDBWriter):
    """Concrete implementation for writing to CSV live database."""

    def append_to_db(
        self, live_db_file: Path, ticker: str, cusip: str, name: str, source: str
    ) -> None:
        """Append a new CUSIP mapping to the CSV live database."""
        import csv
        import logging

        logger = logging.getLogger(__name__)

        try:
            with live_db_file.open("a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([ticker, cusip, name, source])
        except Exception as e:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning("Could not write to DB: %s", e)


class DatabaseLoaderFactory:
    """Factory for creating database loader instances."""

    @staticmethod
    def create_seed_loader() -> ISeedLoader:
        """Create a seed loader instance."""
        return CsvSeedLoader()

    @staticmethod
    def create_live_db_loader() -> ILiveDBLoader:
        """Create a live database loader instance."""
        return CsvLiveDBLoader()

    @staticmethod
    def create_live_db_writer() -> ILiveDBWriter:
        """Create a live database writer instance."""
        return CsvLiveDBWriter()


def create_seed_loader() -> ISeedLoader:
    """Factory function to create a seed loader instance."""
    return DatabaseLoaderFactory.create_seed_loader()


def create_live_db_loader() -> ILiveDBLoader:
    """Factory function to create a live database loader instance."""
    return DatabaseLoaderFactory.create_live_db_loader()


def create_live_db_writer() -> ILiveDBWriter:
    """Factory function to create a live database writer instance."""
    return DatabaseLoaderFactory.create_live_db_writer()
