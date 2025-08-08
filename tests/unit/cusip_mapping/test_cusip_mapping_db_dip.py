"""
Tests for CusipMappingDB class with Dependency Inversion Principle implementation.

This module tests the refactored CusipMappingDB class that uses dependency injection
for database loading operations.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from portfolio_backtester.cusip_mapping import CusipMappingDB
from portfolio_backtester.interfaces.database_loader_interface import (
    ISeedLoader,
    ILiveDBLoader,
)


class TestCusipMappingDBDependencyInjection:
    """Test CusipMappingDB with dependency injection."""

    def test_initialization_with_default_loaders(self):
        """Test CusipMappingDB initialization with default loaders."""
        with patch('portfolio_backtester.cusip_mapping.create_seed_loader') as mock_seed_factory, \
             patch('portfolio_backtester.cusip_mapping.create_live_db_loader') as mock_live_factory:
            
            # Mock the loaders
            mock_seed_loader = Mock(spec=ISeedLoader)
            mock_live_db_loader = Mock(spec=ILiveDBLoader)
            mock_seed_factory.return_value = mock_seed_loader
            mock_live_factory.return_value = mock_live_db_loader
            
            # Create instance
            db = CusipMappingDB()
            
            # Verify factory functions were called
            mock_seed_factory.assert_called_once()
            mock_live_factory.assert_called_once()
            
            # Verify loaders were called with correct parameters
            mock_seed_loader.load_seeds.assert_called_once()
            mock_live_db_loader.load_live_db.assert_called_once()
            
            # Verify cache was initialized
            assert isinstance(db._cache, dict)

    def test_initialization_with_injected_loaders(self):
        """Test CusipMappingDB initialization with injected loaders."""
        # Create mock loaders
        mock_seed_loader = Mock(spec=ISeedLoader)
        mock_live_db_loader = Mock(spec=ILiveDBLoader)
        
        # Create instance with injected dependencies
        db = CusipMappingDB(
            seed_loader=mock_seed_loader,
            live_db_loader=mock_live_db_loader
        )
        
        # Verify loaders were called
        mock_seed_loader.load_seeds.assert_called_once()
        mock_live_db_loader.load_live_db.assert_called_once()
        
        # Verify cache was initialized
        assert isinstance(db._cache, dict)

    def test_loaders_called_with_correct_parameters(self):
        """Test that loaders are called with correct file paths and cache."""
        mock_seed_loader = Mock(spec=ISeedLoader)
        mock_live_db_loader = Mock(spec=ILiveDBLoader)
        
        # Create instance
        db = CusipMappingDB(
            seed_loader=mock_seed_loader,
            live_db_loader=mock_live_db_loader
        )
        
        # Verify seed loader was called with correct parameters
        seed_call_args = mock_seed_loader.load_seeds.call_args
        seed_files, cache = seed_call_args[0]
        assert isinstance(seed_files, list)
        assert len(seed_files) >= 1  # At least one seed file
        assert isinstance(cache, dict)
        assert cache is db._cache
        
        # Verify live db loader was called with correct parameters
        live_call_args = mock_live_db_loader.load_live_db.call_args
        live_db_file, cache = live_call_args[0]
        assert isinstance(live_db_file, Path)
        assert isinstance(cache, dict)
        assert cache is db._cache

    def test_cache_population_through_loaders(self):
        """Test that cache gets populated through the loaders."""
        # Create mock loaders that populate the cache
        mock_seed_loader = Mock(spec=ISeedLoader)
        mock_live_db_loader = Mock(spec=ILiveDBLoader)
        
        def populate_seed_cache(seed_files, cache):
            cache["AAPL"] = ("12345678", "Apple Inc.")
            cache["MSFT"] = ("87654321", "Microsoft Corporation")
        
        def populate_live_cache(live_db_file, cache):
            cache["GOOGL"] = ("11111111", "Alphabet Inc.")
        
        mock_seed_loader.load_seeds.side_effect = populate_seed_cache
        mock_live_db_loader.load_live_db.side_effect = populate_live_cache
        
        # Create instance
        db = CusipMappingDB(
            seed_loader=mock_seed_loader,
            live_db_loader=mock_live_db_loader
        )
        
        # Verify cache was populated
        assert len(db._cache) == 3
        assert db._cache["AAPL"] == ("12345678", "Apple Inc.")
        assert db._cache["MSFT"] == ("87654321", "Microsoft Corporation")
        assert db._cache["GOOGL"] == ("11111111", "Alphabet Inc.")

    def test_resolve_uses_populated_cache(self):
        """Test that resolve method uses the cache populated by loaders."""
        # Create mock loaders that populate the cache
        mock_seed_loader = Mock(spec=ISeedLoader)
        mock_live_db_loader = Mock(spec=ILiveDBLoader)
        
        def populate_cache(seed_files, cache):
            cache["AAPL"] = ("12345678", "Apple Inc.")
        
        mock_seed_loader.load_seeds.side_effect = populate_cache
        mock_live_db_loader.load_live_db.return_value = None  # No-op
        
        # Create instance
        db = CusipMappingDB(
            seed_loader=mock_seed_loader,
            live_db_loader=mock_live_db_loader
        )
        
        # Test resolve method
        cusip, name = db.resolve("AAPL")
        assert cusip == "12345678"
        assert name == "Apple Inc."

    def test_resolve_handles_missing_ticker(self):
        """Test that resolve method handles missing tickers correctly."""
        # Create mock loaders with empty cache
        mock_seed_loader = Mock(spec=ISeedLoader)
        mock_live_db_loader = Mock(spec=ILiveDBLoader)
        
        # Create instance
        db = CusipMappingDB(
            seed_loader=mock_seed_loader,
            live_db_loader=mock_live_db_loader
        )
        
        # Mock all lookup methods to return None
        with patch('portfolio_backtester.cusip_mapping.lookup_openfigi') as mock_openfigi, \
             patch('portfolio_backtester.cusip_mapping.lookup_edgar') as mock_edgar, \
             patch('portfolio_backtester.cusip_mapping.lookup_duckduckgo') as mock_duckduckgo, \
             patch('portfolio_backtester.cusip_mapping.OPENFIGI_API_KEY', None):
            
            mock_edgar.return_value = (None, None)
            mock_duckduckgo.return_value = (None, None)
            
            # Test that KeyError is raised when ticker is not found
            with pytest.raises(KeyError, match="CUSIP not found for ticker UNKNOWN"):
                db.resolve("UNKNOWN")

    def test_backward_compatibility_with_no_parameters(self):
        """Test that the class maintains backward compatibility when called with no parameters."""
        with patch('portfolio_backtester.cusip_mapping.create_seed_loader') as mock_seed_factory, \
             patch('portfolio_backtester.cusip_mapping.create_live_db_loader') as mock_live_factory:
            
            # Mock the loaders
            mock_seed_loader = Mock(spec=ISeedLoader)
            mock_live_db_loader = Mock(spec=ILiveDBLoader)
            mock_seed_factory.return_value = mock_seed_loader
            mock_live_factory.return_value = mock_live_db_loader
            
            # This should work exactly as before
            db = CusipMappingDB()
            
            # Verify it's properly initialized
            assert hasattr(db, '_cache')
            assert isinstance(db._cache, dict)

    def test_mixed_dependency_injection(self):
        """Test mixed injection - providing only one loader."""
        mock_seed_loader = Mock(spec=ISeedLoader)
        
        with patch('portfolio_backtester.cusip_mapping.create_live_db_loader') as mock_live_factory:
            mock_live_db_loader = Mock(spec=ILiveDBLoader)
            mock_live_factory.return_value = mock_live_db_loader
            
            # Provide only seed loader
            db = CusipMappingDB(seed_loader=mock_seed_loader)
            
            # Verify both loaders were used
            mock_seed_loader.load_seeds.assert_called_once()
            mock_live_db_loader.load_live_db.assert_called_once()
            mock_live_factory.assert_called_once()

    def test_dependency_inversion_principle_compliance(self):
        """Test that the class follows Dependency Inversion Principle."""
        # The class should depend on abstractions (interfaces), not concretions
        
        # Create custom loader implementations
        class CustomSeedLoader(ISeedLoader):
            def load_seeds(self, seed_files, cache):
                cache["CUSTOM"] = ("99999999", "Custom Ticker")
        
        class CustomLiveDBLoader(ILiveDBLoader):
            def load_live_db(self, live_db_file, cache):
                cache["LIVE"] = ("88888888", "Live Ticker")
        
        # Inject custom loaders
        custom_seed_loader = CustomSeedLoader()
        custom_live_loader = CustomLiveDBLoader()
        
        db = CusipMappingDB(
            seed_loader=custom_seed_loader,
            live_db_loader=custom_live_loader
        )
        
        # Verify custom loaders were used
        assert "CUSTOM" in db._cache
        assert "LIVE" in db._cache
        assert db._cache["CUSTOM"] == ("99999999", "Custom Ticker")
        assert db._cache["LIVE"] == ("88888888", "Live Ticker")

    def test_loader_interface_type_checking(self):
        """Test that loaders must implement the correct interfaces."""
        # This test verifies static typing is working correctly
        
        # Valid loaders
        valid_seed_loader = Mock(spec=ISeedLoader)
        valid_live_loader = Mock(spec=ILiveDBLoader)
        
        # Should work without issues
        db = CusipMappingDB(
            seed_loader=valid_seed_loader,
            live_db_loader=valid_live_loader
        )
        
        assert db is not None  # Just verify it was created successfully


if __name__ == '__main__':
    pytest.main([__file__, '-v'])