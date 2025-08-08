"""
Legacy persistence functions - DEPRECATED.

These functions have been moved to the interface-based system:
- ISeedLoader interface with CsvSeedLoader implementation
- ILiveDBLoader interface with CsvLiveDBLoader implementation
- ILiveDBWriter interface with CsvLiveDBWriter implementation

This file is kept for potential backward compatibility but should not be used.
All database operations now go through the interface system in
src/portfolio_backtester/interfaces/database_loader_interface.py
"""

import logging

logger = logging.getLogger(__name__)

# Legacy functions removed - use interface-based system instead
# This ensures no dual-path exists and forces use of proper DI pattern
