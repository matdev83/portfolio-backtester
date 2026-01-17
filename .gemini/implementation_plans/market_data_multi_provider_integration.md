# Implementation Plan: Integrate market-data-multi-provider into portfolio-backtester

## Executive Summary

This plan details the steps to replace local data downloaders in `portfolio-backtester` with the centralized `market-data-multi-provider` package. This will provide:

- **Unified data management** across all stock-market projects
- **Multi-provider fallback** (Stooq → yfinance → TradingView)
- **Consistent caching** via canonical parquet storage
- **Additional data sources** (CBOE, FRED, SqueezeMetrics, SpotGamma)
- **Coverage validation** with trading calendars

---

## Current State Analysis

### portfolio-backtester Data Sources
| Component | File | Purpose |
|-----------|------|---------|
| `BaseDataSource` | `data_sources/base_data_source.py` | Abstract base class |
| `StooqDataSource` | `data_sources/stooq_data_source.py` | Stooq via pandas_datareader |
| `YFinanceDataSource` | `data_sources/yfinance_data_source.py` | yfinance downloads |
| `HybridDataSource` | `data_sources/hybrid_data_source.py` | Stooq + yfinance with failover |
| `MemoryDataSource` | `data_sources/memory_data_source.py` | Testing data source |
| `IDataSourceFactory` | `interfaces/data_source_interface.py` | Factory pattern |

### Hardcoded Symbols Found
| Location | Symbols | Context |
|----------|---------|---------|
| `optimization/evaluator.py:163` | `SPY, TLT, GLD, VTI, QQQ` | Default fallback universe |
| `optimization/evaluator.py:216` | `SPY` | Default benchmark |
| `interfaces/evaluation_strategy.py:106` | `SPY` | Benchmark in metrics calc |
| `backtester_logic/constraint_logic.py:30` | `SPY` | Global benchmark default |
| `config/parameters.yaml:65` | `SPY` | Global benchmark config |
| `testing/strategies/dummy_signal_strategy.py:40` | `SPY` | Test strategy default |
| `strategies/builtins/signal/uvxy_rsi_signal_strategy.py:37,66,70` | `SPY` | Non-universe data |

### Universe Sources
| Type | Location | How Resolved |
|------|----------|--------------|
| **Named universes** | `config/universes/*.txt` | `universe_loader.py` |
| **Dynamic (SPY holdings)** | `universe_data/spy_holdings.py` | `get_top_weight_sp500_components()` |
| **Fixed (YAML)** | Scenario configs | `universe_resolver.py` |
| **Single symbol** | Scenario configs | `universe_resolver.py` |

### Existing Universe Files
| File | Symbols Count |
|------|---------------|
| `sp500_top50.txt` | 50 tickers |
| `dow_jones.txt` | 30 tickers |
| `nasdaq_top20.txt` | 20 tickers |

---

## Phase 1: Foundation Setup (Low Risk)

### 1.1 Add market-data-multi-provider as Local Dependency

**File:** `pyproject.toml`

```toml
dependencies = [
    # ... existing deps ...
    "market-data-multi-provider @ file:../market-data-multi-provider",
]
```

Alternative for development:
```bash
pip install -e ../market-data-multi-provider
```

### 1.2 Create Symbol Mapping Registry

**Purpose:** Map portfolio-backtester ticker format (e.g., `SPY`, `^GSPC`) to market-data-multi-provider canonical IDs (e.g., `AMEX:SPY`, `SP:SPX`).

**New File:** `src/portfolio_backtester/data_sources/symbol_mapper.py`

```python
"""Symbol mapping between portfolio-backtester and market-data-multi-provider formats."""

from typing import Optional
from functools import lru_cache

# Direct mapping for common symbols
_SYMBOL_MAP: dict[str, str] = {
    # Index symbols
    "^GSPC": "SP:SPX",
    "SPX": "SP:SPX",
    "^VIX": "CBOE:VIX",
    "VIX": "CBOE:VIX",
    "^VVIX": "CBOE:VVIX",
    "^SKEW": "CBOE:SKEW",
    "^TNX": "TVC:TNX",
    "^TYX": "TVC:TYX",
    "^IRX": "TVC:IRX",
    
    # Reverse mapping for common ETFs (already match)
    # SPY, QQQ, IWM, etc. will be auto-mapped
}

# Exchange prefixes for U.S. stocks
_EXCHANGE_PREFIXES = {
    "NASDAQ": ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", 
               "AVGO", "COST", "ADBE", "NFLX", "PEP", "CSCO", "CMCSA", "INTC",
               "TXN", "QCOM", "AMGN", "HON"],  # Major NASDAQ stocks
    "AMEX": ["SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "XLB", "XLE", "XLF",
             "XLI", "XLP", "XLU", "XLV", "XLK", "XLY", "XLRE", "RSP", "VTI"],  # Major ETFs
}


@lru_cache(maxsize=4096)
def to_canonical_id(ticker: str) -> str:
    """Convert local ticker to market-data-multi-provider canonical ID.
    
    Args:
        ticker: Local ticker symbol (e.g., "SPY", "^GSPC")
        
    Returns:
        Canonical ID (e.g., "AMEX:SPY", "SP:SPX")
    """
    # Check direct mapping first
    ticker_upper = ticker.upper()
    if ticker_upper in _SYMBOL_MAP:
        return _SYMBOL_MAP[ticker_upper]
    
    # Check if already in canonical format
    if ":" in ticker:
        return ticker
    
    # Auto-detect exchange prefix
    for exchange, symbols in _EXCHANGE_PREFIXES.items():
        if ticker_upper in symbols:
            return f"{exchange}:{ticker_upper}"
    
    # Default: assume NYSE for unknown symbols
    # Most individual stocks without explicit exchange are NYSE
    return f"NYSE:{ticker_upper}"


@lru_cache(maxsize=4096)
def from_canonical_id(canonical_id: str) -> str:
    """Convert canonical ID back to local ticker format.
    
    Args:
        canonical_id: Canonical ID (e.g., "AMEX:SPY")
        
    Returns:
        Local ticker (e.g., "SPY")
    """
    # Reverse mapping for special symbols
    for local, canonical in _SYMBOL_MAP.items():
        if canonical == canonical_id:
            return local
    
    # Extract ticker from canonical format
    if ":" in canonical_id:
        return canonical_id.split(":", 1)[1]
    
    return canonical_id


def clear_cache() -> None:
    """Clear the mapping cache."""
    to_canonical_id.cache_clear()
    from_canonical_id.cache_clear()
```

---

## Phase 2: Data Source Adapter (Medium Risk)

### 2.1 Create MarketDataMultiProviderDataSource

**New File:** `src/portfolio_backtester/data_sources/mdmp_data_source.py`

```python
"""Market Data Multi-Provider data source adapter."""

import logging
from typing import List, Optional
from pathlib import Path

import pandas as pd

from .base_data_source import BaseDataSource
from .symbol_mapper import to_canonical_id, from_canonical_id

logger = logging.getLogger(__name__)


class MarketDataMultiProviderDataSource(BaseDataSource):
    """Data source using market-data-multi-provider package.
    
    This adapter wraps MarketDataClient to provide compatibility with
    portfolio-backtester's BaseDataSource interface.
    """

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        cache_expiry_hours: int = 24,  # For compatibility, not used internally
    ) -> None:
        """Initialize the data source.
        
        Args:
            data_dir: Directory for data storage (uses MDMP default if None)
            cache_expiry_hours: Ignored - MDMP handles caching internally
        """
        try:
            from market_data_multi_provider import MarketDataClient
        except ImportError as e:
            raise ImportError(
                "market-data-multi-provider is not installed. "
                "Install with: pip install -e ../market-data-multi-provider"
            ) from e
        
        self.client = MarketDataClient(data_dir=data_dir)
        self._cache_expiry_hours = cache_expiry_hours  # Stored for interface compat
        
        logger.debug("MarketDataMultiProviderDataSource initialized")

    def get_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with MultiIndex columns (Ticker, Field)
        """
        from datetime import datetime
        
        if not tickers:
            logger.warning("No tickers provided")
            return pd.DataFrame()
        
        logger.info(f"Fetching {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Convert dates
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Convert tickers to canonical IDs
        symbol_map = {to_canonical_id(t): t for t in tickers}
        canonical_ids = list(symbol_map.keys())
        
        # Fetch data using MDMP
        results = self.client.fetch_many(
            canonical_ids,
            start=start,
            end=end,
            use_cache=True,
            use_canonical=True,
        )
        
        # Process results into expected MultiIndex format
        all_ticker_data = []
        successful = 0
        failed = 0
        
        for outcome in results:
            original_ticker = symbol_map.get(outcome.symbol_id, 
                                             from_canonical_id(outcome.symbol_id))
            
            if outcome.data is None or outcome.data.empty:
                logger.warning(f"No data for {original_ticker} ({outcome.symbol_id})")
                failed += 1
                continue
            
            df = outcome.data.copy()
            
            # Ensure standard OHLCV columns exist
            expected_cols = ["Open", "High", "Low", "Close", "Volume"]
            available_cols = [c for c in expected_cols if c in df.columns]
            
            if not available_cols:
                logger.warning(f"No OHLCV columns for {original_ticker}")
                failed += 1
                continue
            
            # Create MultiIndex columns
            ticker_df = df[available_cols].copy()
            ticker_df.columns = pd.MultiIndex.from_product(
                [[original_ticker], ticker_df.columns],
                names=["Ticker", "Field"]
            )
            
            all_ticker_data.append(ticker_df)
            successful += 1
        
        logger.info(f"Fetched {successful} tickers successfully, {failed} failed")
        
        if not all_ticker_data:
            return pd.DataFrame()
        
        # Combine all ticker data
        result = pd.concat(all_ticker_data, axis=1)
        return result
```

### 2.2 Register New Data Source in Factory

**File:** `src/portfolio_backtester/interfaces/data_source_interface.py`

Add to `ConcreteDataSourceFactory.create_data_source()`:

```python
from ..data_sources.mdmp_data_source import MarketDataMultiProviderDataSource

data_source_map = {
    "stooq": StooqDataSource,
    "yfinance": YFinanceDataSource,
    "hybrid": HybridDataSource,
    "memory": MemoryDataSource,
    "test": MemoryDataSource,
    "mdmp": MarketDataMultiProviderDataSource,  # NEW
    "market-data-multi-provider": MarketDataMultiProviderDataSource,  # Alias
}

# ... in create_data_source():
elif ds_name in ("mdmp", "market-data-multi-provider"):
    data_dir = global_config.get("data_dir")
    return cast(IDataSource, MarketDataMultiProviderDataSource(data_dir=data_dir))
```

---

## Phase 3: Symbol Registry Updates (Medium Risk)

### 3.1 Register All Required Symbols in market-data-multi-provider

**File:** `market-data-multi-provider/src/market_data_multi_provider/resources/symbols.json`

Add entries for all symbols used in portfolio-backtester:

```json
{
  "NYSE:AAPL": {
    "preferred_provider": "stooq",
    "calendar_id": "NYSE",
    "description": "Apple Inc.",
    "providers": {
      "stooq": "aapl.us",
      "yfinance": "AAPL",
      "tradingview": "NASDAQ:AAPL"
    },
    "aliases": ["AAPL"]
  },
  "NYSE:MSFT": {
    "preferred_provider": "stooq",
    "calendar_id": "NYSE",
    "description": "Microsoft Corporation",
    "providers": {
      "stooq": "msft.us",
      "yfinance": "MSFT",
      "tradingview": "NASDAQ:MSFT"
    },
    "aliases": ["MSFT"]
  }
  // ... add all stocks from universes
}
```

### 3.2 Symbols to Register

**From Universe Files:**
- `sp500_top50.txt`: 50 tickers (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK.B, UNH, JNJ, etc.)
- `dow_jones.txt`: 30 tickers (AAPL, AMGN, AXP, BA, CAT, CRM, CSCO, CVX, etc.)
- `nasdaq_top20.txt`: 20 tickers (with overlap to above)

**From Hardcoded References:**
- `SPY` - Already in symbols.json ✓
- `QQQ` - Already in symbols.json ✓
- `GLD` - Needs to be added
- `TLT` - Needs to be added
- `VTI` - Needs to be added
- `IWM` - Already in symbols.json ✓
- `^GSPC`/`SPX` - Already in symbols.json ✓ (as SP:SPX)
- `UVXY` - Needs to be added

**Priority for provider ordering (for common stocks):**
1. `stooq` - Best historical data, no rate limits
2. `yfinance` - Good fallback, widely available
3. `tradingview` - Requires credentials, last resort

---

## Phase 4: Configuration Updates (Low Risk)

### 4.1 Update Default Data Source

**File:** `config/parameters.yaml`

```yaml
# Data source configuration
data_source: "mdmp"  # Changed from "hybrid"
data_dir: null  # Use MDMP default (market-data-multi-provider/data/)

# Legacy options (for backward compatibility)
prefer_stooq: true  # Ignored when using mdmp
```

### 4.2 Environment Variable Support

**Documentation:** Update README.md to include:

```markdown
### Market Data Configuration

The backtester uses `market-data-multi-provider` for data fetching. Configure via:

- `MARKET_DATA_CACHE_DIR`: Override data directory
- Data is shared across all projects using MDMP
```

---

## Phase 5: Remove Legacy Data Sources (High Risk)

### 5.1 Deprecation Path

1. **Phase 5a** (Optional): Keep legacy sources, use MDMP as default
2. **Phase 5b** (After validation): Remove legacy sources

### 5.2 Files to Eventually Remove
| File | Replacement |
|------|------------|
| `data_sources/stooq_data_source.py` | MDMP StooqProvider |
| `data_sources/yfinance_data_source.py` | MDMP YFinanceProvider |
| `data_sources/hybrid_data_source.py` | MDMP with fallback chain |

### 5.3 Update Imports Throughout Codebase

After MDMP is validated, update all imports:
```python
# Before
from ..data_sources.hybrid_data_source import HybridDataSource

# After
from ..data_sources.mdmp_data_source import MarketDataMultiProviderDataSource
```

---

## Phase 6: Testing & Validation (Critical)

### 6.1 Create Integration Tests

**New File:** `tests/integration/test_mdmp_data_source.py`

```python
"""Integration tests for MarketDataMultiProviderDataSource."""

import pytest
import pandas as pd
from datetime import date

from portfolio_backtester.data_sources.mdmp_data_source import MarketDataMultiProviderDataSource
from portfolio_backtester.data_sources.symbol_mapper import to_canonical_id


class TestMdmpDataSource:
    """Test MDMP data source integration."""
    
    @pytest.fixture
    def data_source(self):
        return MarketDataMultiProviderDataSource()
    
    def test_fetch_single_ticker(self, data_source):
        """Test fetching a single ticker."""
        result = data_source.get_data(["SPY"], "2024-01-01", "2024-01-31")
        assert not result.empty
        assert ("SPY", "Close") in result.columns
    
    def test_fetch_multiple_tickers(self, data_source):
        """Test fetching multiple tickers."""
        tickers = ["SPY", "QQQ", "IWM"]
        result = data_source.get_data(tickers, "2024-01-01", "2024-01-31")
        assert not result.empty
        for ticker in tickers:
            assert (ticker, "Close") in result.columns
    
    def test_symbol_mapping(self):
        """Test symbol mapping works correctly."""
        assert to_canonical_id("SPY") == "AMEX:SPY"
        assert to_canonical_id("^GSPC") == "SP:SPX"
        assert to_canonical_id("AAPL") == "NASDAQ:AAPL"
```

### 6.2 Backward Compatibility Test

Compare output between old and new data sources:
```python
def test_output_compatibility():
    """Ensure MDMP output matches HybridDataSource format."""
    hybrid = HybridDataSource()
    mdmp = MarketDataMultiProviderDataSource()
    
    tickers = ["SPY", "QQQ"]
    start, end = "2024-01-01", "2024-01-31"
    
    hybrid_data = hybrid.get_data(tickers, start, end)
    mdmp_data = mdmp.get_data(tickers, start, end)
    
    # Check structure matches
    assert hybrid_data.columns.equals(mdmp_data.columns)
    
    # Check data is similar (within tolerance for provider differences)
    pd.testing.assert_frame_equal(
        hybrid_data, mdmp_data, 
        check_exact=False, 
        rtol=0.01  # 1% tolerance
    )
```

---

## Implementation Order

### Sprint 1: Foundation (Est. 2-3 hours)
- [ ] 1.1 Add dependency to pyproject.toml
- [ ] 1.2 Create symbol_mapper.py
- [ ] 2.1 Create mdmp_data_source.py
- [ ] 2.2 Register in factory

### Sprint 2: Symbol Registry (Est. 3-4 hours)
- [ ] 3.1 Inventory all symbols from universe files
- [ ] 3.2 Add missing symbols to symbols.json (with stooq→yfinance→tradingview order)
- [ ] 3.3 Validate symbol resolution works

### Sprint 3: Integration Testing (Est. 2-3 hours)
- [ ] 6.1 Create integration tests
- [ ] 6.2 Create compatibility tests
- [ ] Run full test suite

### Sprint 4: Configuration & Rollout (Est. 1-2 hours)
- [ ] 4.1 Update default config
- [ ] 4.2 Update documentation
- [ ] Test with real backtesting scenarios

### Sprint 5: Cleanup (Optional, Est. 2-3 hours)
- [ ] 5.1 Deprecate legacy sources (add warnings)
- [ ] 5.2 (Future) Remove legacy sources after validation period

---

## Rollback Plan

If issues arise, revert to legacy sources by:

1. Change `data_source` in `parameters.yaml` back to `"hybrid"`
2. Keep legacy data source files intact during transition
3. No code changes needed if factory pattern is used correctly

---

## Success Criteria

1. ✅ All existing tests pass with MDMP data source
2. ✅ Backtesting results are numerically equivalent (within tolerance)
3. ✅ Data is cached in shared location (`market-data-multi-provider/data/`)
4. ✅ Provider fallback chain works (stooq → yfinance → tradingview)
5. ✅ Universe resolution works with all tickers
6. ✅ No regressions in backtest performance

---

## Appendix: Complete Symbol List for symbols.json

### ETFs (AMEX)
| Symbol | Description | Priority |
|--------|-------------|----------|
| GLD | SPDR Gold Trust | stooq, yfinance, tradingview |
| TLT | iShares 20+ Year Treasury Bond | stooq, yfinance, tradingview |
| VTI | Vanguard Total Stock Market | stooq, yfinance, tradingview |
| UVXY | ProShares Ultra VIX Short-Term Futures | yfinance, tradingview |

### Major Stocks (from universe files)
All S&P 500 top 50, Dow Jones 30, and NASDAQ 20 stocks need entries with:
- `preferred_provider`: `"stooq"`  
- Provider order: `stooq` → `yfinance` → `tradingview`
- Stooq format: `{ticker}.us` (e.g., `aapl.us`)
- Calendar: `"NYSE"`

Total symbols to add: ~80 (after deduplication)
