"""Test fixtures for universe configuration tests."""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

from portfolio_backtester.universe_loader import clear_universe_cache


class UniverseTestFixture:
    """Test fixture for universe configuration tests."""

    def __init__(self):
        self.temp_dir = None
        self.universes_dir_patcher = None

    def setup(self):
        """Set up the test fixture."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.universes_dir_patcher = patch(
            "portfolio_backtester.universe_loader.UNIVERSES_DIR", self.temp_dir
        )
        self.universes_dir_patcher.start()
        clear_universe_cache()

    def teardown(self):
        """Clean up the test fixture."""
        if self.universes_dir_patcher:
            self.universes_dir_patcher.stop()
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        clear_universe_cache()

    def create_universe(self, name: str, content: str) -> Path:
        """Create a test universe file."""
        if not self.temp_dir:
            raise RuntimeError("Fixture not set up. Call setup() first.")

        universe_file: Path = self.temp_dir / f"{name}.txt"
        universe_file.write_text(content)
        return universe_file

    def create_standard_universes(self) -> Dict[str, Any]:
        """Create a set of standard test universes."""
        universes = {}

        # Tech stocks universe
        universes["tech"] = self.create_universe("tech", "AAPL\nMSFT\nGOOGL\nNVDA\n")

        # Finance universe
        universes["finance"] = self.create_universe("finance", "JPM\nBAC\nWFC\nGS\n")

        # Small test universe
        universes["small"] = self.create_universe("small", "AAPL\nMSFT\n")

        # Universe with comments and formatting
        universes["formatted"] = self.create_universe(
            "formatted",
            """# Test universe
AAPL
MSFT  # Microsoft
# Comment line
GOOGL

AMZN
""",
        )

        return universes


def create_universe_fixture():
    """Factory function to create a universe test fixture."""
    return UniverseTestFixture()
